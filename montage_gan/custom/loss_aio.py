from typing import List, Optional

import numpy as np
import torch

from custom_utils.image_utils import make_batch_for_pos_estimator, alpha_composite_pytorch, normalize_minus11, \
    normalize_zero1, alpha_composite, calc_psnr, convert_translate_to_2x3
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix


class Loss:
    def accumulate_gradients(self, phase, real_img, gen_z, sync, gain):  # to be overridden by subclass
        raise NotImplementedError()


class StyleGAN2Loss(Loss):
    def __init__(self, layer_name, device, mapping_network, local_G, local_D, augment_pipe=None,
                 style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        """
        :param device:
        :param layer_name: Local GAN's name
        :param mapping_network: Mapping Network, latent code z -> w
        :param local_G: Local Generator, w -> image
        :param local_D: Local Discriminator
        :param augment_pipe:
        :param style_mixing_prob:
        :param r1_gamma:
        :param pl_batch_shrink:
        :param pl_decay:
        :param pl_weight:
        """
        super().__init__()
        self.layer_name = layer_name
        self.device = device
        self.mapping_network = mapping_network
        self.local_G = local_G
        self.local_D = local_D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

    def run_G(self, z, c, sync):
        """
        :param z: All local generators take the same z as input
        :param sync: sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1), passed from training_loop
                The loss get accumulated for multiple rounds.
                batch_size=4,  # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus
                batch_gpu=4,  # Number of samples processed at a time by one GPU.
                num_gpus=1,  # Number of GPUs participating in the training.
                sync is a boolean
                NOTE:
                1. Has no effect with 1 GPU setup
                2. Otherwise, the module is decorated with no_sync() when sync is False
                Documentation: A context manager to disable gradient synchronizations across DDP processes. Within this context, gradients will be accumulated on module variables, which will later be synchronized in the first forward-backward pass exiting the context.
                https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        :param c
        :return:
        """
        with misc.ddp_sync(self.mapping_network, sync):
            ws = self.mapping_network(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function(f'style_mixing_{self.layer_name}'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                                         torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.mapping_network(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.local_G, sync):
            num_ws = self.local_G.num_ws
            ws = ws[:, :num_ws]
            img = self.local_G(ws)
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.local_D, sync):
            logits = self.local_D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, gen_z, sync, gain, real_c=None, gen_c=None):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function(f'Gmain_forward_{self.layer_name}'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl))  # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report(f'{self.layer_name}/Loss/scores/fake', gen_logits)
                training_stats.report(f'{self.layer_name}/Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
                training_stats.report(f'{self.layer_name}/Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function(f'Gmain_backward_{self.layer_name}'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function(f'Gpl_forward_{self.layer_name}'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size] if gen_c else gen_c, sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function(
                        f'pl_grads_{self.layer_name}'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = \
                        torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True,
                                            only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report(f'{self.layer_name}/Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report(f'{self.layer_name}/Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function(f'Gpl_backward_{self.layer_name}'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function(f'Dgen_forward_{self.layer_name}'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False)  # Gets synced by loss_Dreal.
                training_stats.report(f'{self.layer_name}/Loss/scores/fake', gen_logits)
                training_stats.report(f'{self.layer_name}/Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function(f'Dgen_backward_{self.layer_name}'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + f'_forward_{self.layer_name}'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report(f'{self.layer_name}/Loss/scores/real', real_logits)
                training_stats.report(f'{self.layer_name}/Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report(f'{self.layer_name}/Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function(
                            f'r1_grads_{self.layer_name}'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = \
                            torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True,
                                                only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report(f'{self.layer_name}/Loss/r1_penalty', r1_penalty)
                    training_stats.report(f'{self.layer_name}/Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + f'_backward_{self.layer_name}'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()


def theta_constrain_loss(theta: torch.Tensor):
    device = theta.device
    img_layers = theta.shape[-3]  # B,L,2,3
    # Constrain the theta parameters' range
    upper_bound = convert_translate_to_2x3(torch.tensor([[1., 1.]] * img_layers, device=device))
    lower_bound = convert_translate_to_2x3(torch.tensor([[-1., -1.]] * img_layers, device=device))
    clamp_theta = torch.max(torch.min(theta, upper_bound), lower_bound)
    return torch.norm(theta - clamp_theta, 2)


class MontageGANLoss:
    def __init__(self, device,
                 mapping_network: torch.nn.Module,
                 layer_names: List[str],
                 local_G_list: List[torch.nn.Module],
                 augment_pipe_list: List[Optional[torch.nn.Module]],
                 local_D_list: List[torch.nn.Module],
                 pos_estimator: torch.nn.Module,
                 renderer: Optional[torch.nn.Module],
                 criterion_renderer: Optional[torch.nn.Module],
                 global_augment_pipe: Optional[torch.nn.Module],
                 global_D: torch.nn.Module,
                 global_r1_gamma=10,
                 **sg2loss_kwargs):
        """
        :param device:
        :param mapping_network: Mapping Network
        :param local_G_list: List of local generators
        :param local_D_list: List of local discriminators
        :param pos_estimator: Layer position estimator
        :param global_D: Global discriminator
        :param augment_pipe_list: List of augment pipes
        :param global_r1_gamma: R1 gamma for global GAN
        :param base_resolution: Base resolution of training set
        :param sg2loss_kwargs: kwargs for StyleGAN2Loss
        """
        self.device = device
        assert len(local_G_list) == len(local_D_list) == len(augment_pipe_list)

        # Local
        self.layer_names = layer_names
        self.local_GAN_loss_list = [
            StyleGAN2Loss(layer_name, device, mapping_network, local_G, local_D, augment_pipe, **sg2loss_kwargs)
            for layer_name, local_G, local_D, augment_pipe in
            zip(layer_names, local_G_list, local_D_list, augment_pipe_list)]

        # Global
        self.pos_estimator = pos_estimator
        self.renderer = renderer
        self.criterion_renderer = criterion_renderer
        self.global_augment_pipe = global_augment_pipe
        self.global_D = global_D
        self.global_r1_gamma = global_r1_gamma

    def run_global_G(self, z, c, sync):
        local_G_output_list = [loss.run_G(z, c, sync)[0] for loss in self.local_GAN_loss_list]
        fake_layer = make_batch_for_pos_estimator(local_G_output_list, pad_value=-1)  # B,L,C,H,W [-1,1]
        with misc.ddp_sync(self.pos_estimator, sync):
            transformed_fake_layer, theta = self.pos_estimator(fake_layer)  # B,L,C,H,W [-1,1]
        return transformed_fake_layer, theta

    def run_global_D(self, blchw, c, sync):
        if self.renderer is not None:
            # Using renderer
            output_blended = self.renderer(blchw)  # [-1,1]
        else:
            # Using alpha composite implemented with PyTorch
            output_blended = normalize_minus11(alpha_composite_pytorch(normalize_zero1(blchw)))  # [-1,1]
        # Continue to SG2ADA routine
        if self.global_augment_pipe is not None:
            output_blended = self.global_augment_pipe(output_blended)
        with misc.ddp_sync(self.global_D, sync):
            logits = self.global_D(output_blended, c)
        return logits

    def accumulate_gradients(self, phase, real_blchw, real_list_of_bchw, gen_z, sync, gain, real_c=None, gen_c=None):
        # print(phase)
        do_local = phase.startswith("local_")
        do_global = phase.startswith("global_")
        do_Renderer = phase == "Renderer"
        assert do_local or do_global or do_Renderer

        if do_local:
            # Format: local_[G|D][both|main|reg]_{layer_name}, layer_name may contains "_"
            _, base_phase, *layer_name = phase.split("_")
            layer_name = "_".join(layer_name)
            assert base_phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
            assert layer_name in self.layer_names
            layer_idx = self.layer_names.index(layer_name)
            real_img = real_list_of_bchw[layer_idx]
            loss = self.local_GAN_loss_list[layer_idx]
            loss.accumulate_gradients(base_phase, real_img, gen_z, sync, gain, real_c, gen_c)

        if do_global:
            # Format: global_[G|D][both|main|reg]
            _, base_phase = phase.split("_")
            assert base_phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
            do_Gmain = (base_phase in ['Gmain', 'Gboth'])
            do_Dmain = (base_phase in ['Dmain', 'Dboth'])
            do_Dr1 = (base_phase in ['Dreg', 'Dboth']) and (self.global_r1_gamma != 0)

            # Gmain: Maximize logits for generated layers.
            if do_Gmain:
                with torch.autograd.profiler.record_function(f'Gmain_forward_global'):
                    transformed_fake_layer, theta = self.run_global_G(gen_z, gen_c, sync=sync)
                    gen_logits = self.run_global_D(transformed_fake_layer, gen_c, sync=False)
                    training_stats.report(f'global/Loss/scores/fake', gen_logits)
                    training_stats.report(f'global/Loss/signs/fake', gen_logits.sign())
                    loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
                    training_stats.report(f'global/Loss/G/loss', loss_Gmain)
                    loss_theta_constrain = theta_constrain_loss(theta)
                    training_stats.report(f'global/Loss/STN/theta_constrain', loss_theta_constrain)
                with torch.autograd.profiler.record_function(f'Gmain_backward_global'):
                    loss_Gmain.mean().mul(gain).backward(retain_graph=True)
                    loss_theta_constrain.mul(gain).backward()

            # Dmain: Minimize logits for generated layers.
            loss_Dgen = 0
            if do_Dmain:
                with torch.autograd.profiler.record_function(f'Dgen_forward_global'):
                    transformed_fake_layer, _ = self.run_global_G(gen_z, gen_c, sync=False)
                    gen_logits = self.run_global_D(transformed_fake_layer, gen_c, sync=False)
                    training_stats.report(f'global/Loss/scores/fake', gen_logits)
                    training_stats.report(f'global/Loss/signs/fake', gen_logits.sign())
                    loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
                with torch.autograd.profiler.record_function(f'Dgen_backward_global'):
                    loss_Dgen.mean().mul(gain).backward()

            # Dmain: Maximize logits for real layers.
            if do_Dmain or do_Dr1:
                name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
                with torch.autograd.profiler.record_function(name + f'_forward_global'):
                    real_layer_tmp = real_blchw.detach().requires_grad_(do_Dr1)
                    real_logits = self.run_global_D(real_layer_tmp, real_c, sync=sync)
                    training_stats.report(f'global/Loss/scores/real', real_logits)
                    training_stats.report(f'global/Loss/signs/real', real_logits.sign())

                    loss_Dreal = 0
                    if do_Dmain:
                        loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                        training_stats.report(f'global/Loss/D/loss', loss_Dgen + loss_Dreal)

                    loss_Dr1 = 0
                    if do_Dr1:
                        with torch.autograd.profiler.record_function(
                                f'r1_grads_global'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = \
                                torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_layer_tmp],
                                                    create_graph=True,
                                                    only_inputs=True)[0]
                        r1_penalty = r1_grads.square().sum([1, 2, 3])
                        loss_Dr1 = r1_penalty * (self.global_r1_gamma / 2)
                        training_stats.report(f'Loss/r1_penalty', r1_penalty)
                        training_stats.report(f'Loss/D/reg', loss_Dr1)

                with torch.autograd.profiler.record_function(name + f'_backward_global'):
                    (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

        # Renderer: Retrain renderer.
        if do_Renderer:
            assert self.renderer is not None
            assert self.criterion_renderer is not None
            with torch.autograd.profiler.record_function('renderer_forward'):
                # Renderer with generated samples
                x, _ = self.run_global_G(gen_z, gen_c, sync=False)  # B,L,C,H,W
                output_renderer = self.renderer(x)  # [-1,1]
                target = alpha_composite(normalize_zero1(x.detach()))  # [0,1]
                target = target.to(self.device)
                loss_renderer_gen = self.criterion_renderer(normalize_zero1(output_renderer), target)
                training_stats.report('Renderer/loss_gen', loss_renderer_gen.item())
                training_stats.report('Renderer/psnr_gen',
                                      calc_psnr(normalize_zero1(output_renderer.detach()), target.detach()))

                # Renderer with real samples
                x = real_blchw  # B,L,C,H,W
                output_renderer = self.renderer(x)  # [-1,1]
                target = alpha_composite(normalize_zero1(x.detach()))  # [0,1]
                target = target.to(self.device)
                loss_renderer_real = self.criterion_renderer(normalize_zero1(output_renderer), target)
                training_stats.report('Renderer/loss_real', loss_renderer_real.item())
                training_stats.report('Renderer/psnr_real',
                                      calc_psnr(normalize_zero1(output_renderer.detach()), target.detach()))

            with torch.autograd.profiler.record_function('renderer_backward'):
                loss_renderer_gen.backward()
                loss_renderer_real.backward()
