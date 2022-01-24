import copy
import json
import os
import time
from itertools import chain
from typing import Optional, Union, List, Dict

import numpy as np
import psutil
import torch
import torch.backends.cudnn
import torch.utils.data
from torch.optim import Adam
from torchvision.utils import save_image

import dnnlib
from custom.dataset_aio import DatasetAIO
from custom.networks_aio import MappingNetwork, SynthesisNetwork, Discriminator
from custom_utils.image_utils import alpha_composite, make_batch_for_pos_estimator, normalize_zero1, \
    make_batch_for_local_d, normalize_minus11
from fukuwarai.networks import STNv2c as STN
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from training.augment import AugmentPipe
from metrics import metric_main

"""
Renderer module
"""
# Note:
# Loss: MSE, Variant: Tanh
# diff_rendering/211120-1956-output-tanh/renderer032000.pth.tar

# Loss: L1, Variant: Subpixel
# diff_rendering/211210-1834-output-subpixel/renderer032000.pth.tar

renderer_config = {
    "img_resolution": 256,
    "img_channels": 4,
    "img_layers": 9,
    "device": "cuda",
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "amsgrad": True,  # use amsgrad for Tanh variant
    "loss_type": "mse",  # l1/mse
    # "loss_type": "l1",  # l1/mse
    # "renderer_pth_path": None,  # pth to resume from
    "renderer_pth_path": "pretrained/diff_rendering/211120-1956-output-tanh/renderer032000.pth.tar",
    "renderer_type": "tanh",  # sigmoid/tanh/subpixel
    # "renderer_type": "subpixel",  # sigmoid/tanh/subpixel
    "bypass_renderer": False,
}

config = {
    # Control the number of convolution layer of GANs
    "conv_base_index": 3,  # SG2ada default is 2. The larger the index, the less conv. layer will be used.

    # Training specific
    # "train_global": False, # Step 1 (Pretrain local GANs)
    "train_global": True,  # Step 2 (Train global GAN)
    "train_renderer": True,  # Enable renderer training?
    "local_noaug": False,  # `noaug` override for local GANs (Should be `False` during step 1)
    "global_noaug": False,  # `noaug` override for global GAN
    "local_fixaug": True,  # Not implemented yet
    "local_augment_p": 0.3,  # Not implemented yet

    # Loss specific
    "global_d_real_use_renderer": True,
    "renderer_retrain_use_real": True,

    # Report specific
    "debug": False  # Print debug message
}

if not config["train_global"]:
    print("Global GAN training is disabled!!!")
    config["train_renderer"] = False
    renderer_config["bypass_renderer"] = True

if not config["train_renderer"]:
    print("Renderer training is disabled!!!")

if not renderer_config["bypass_renderer"]:
    is_range_tanh = True  # whether the data range is [0,1] or [-1,1]
    if renderer_config["renderer_type"] == "sigmoid":
        from diff_rendering.networks import Renderer as Renderer  # Sigmoid variant

        is_range_tanh = False
    elif renderer_config["renderer_type"] == "tanh":
        from diff_rendering.networks import RendererTanh as Renderer  # Tanh variant
    elif renderer_config["renderer_type"] == "subpixel":
        from diff_rendering.networks import RendererSubPixelConv as Renderer  # Sub-pixel conv variant
    else:
        raise RuntimeError(f"Unknown renderer type {renderer_config['renderer_type']}")

    renderer = Renderer(img_resolution=renderer_config["img_resolution"], img_channels=renderer_config["img_channels"],
                        img_layers=renderer_config["img_layers"]).to(renderer_config["device"])

    optimizer_renderer = Adam(renderer.parameters(), lr=renderer_config["lr"], betas=renderer_config["betas"],
                              amsgrad=renderer_config["amsgrad"])

    if renderer_config["renderer_pth_path"]:
        saved = torch.load(renderer_config["renderer_pth_path"])
        if isinstance(saved, dict):
            # New pickle format
            renderer.load_state_dict(saved["renderer"])
        else:
            # Old pickle format that contain only the model, deprecated!
            renderer.load_state_dict(saved)

    if renderer_config["loss_type"] == "mse":
        criterion_renderer = torch.nn.MSELoss()
    elif renderer_config["loss_type"] == "l1":
        criterion_renderer = torch.nn.L1Loss()
    else:
        raise RuntimeError(f"Unknown loss type {renderer_config['loss_type']}")
else:
    print("Renderer is bypassed!!!")

"""
Training loop
"""


def training_loop(
        # General
        rank=0,  # Rank of the current process in [0, num_gpus].
        run_dir=".",  # Output directory.
        random_seed=0,  # Global random seed.
        num_gpus=1,  # Number of GPUs participating in the training.
        kimg_per_tick=4,  # Progress snapshot interval.
        image_snapshot_ticks=50,  # How often to save image snapshots? None = disable.
        network_snapshot_ticks=50,  # How often to save network snapshots? None = disable.

        # Networks
        mapping_kwargs=None,  # Options for mapping network.
        local_G_kwargs=None,  # Options for local generator network.
        local_D_kwargs=None,  # Options for local discriminator network.
        pos_estimator_kwargs=None,  # Options for pos estimator network.
        global_D_kwargs=None,  # Options for global discriminator network.

        # Dataset, Dataloader
        training_set_kwargs=None,  # Options for training set.
        data_loader_kwargs=None,  # Options for torch.utils.data.DataLoader.

        # Training
        total_kimg=25000,  # Total length of the training, measured in thousands of real images.
        batch_size=4,  # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
        batch_gpu=4,  # Number of samples processed at a time by one GPU.
        ema_kimg=10,  # Half-life of the exponential moving average (EMA) of generator weights.
        ema_rampup=None,  # EMA ramp-up coefficient.
        G_reg_interval=4,  # How often to perform regularization for G? None = disable lazy regularization.
        D_reg_interval=16,  # How often to perform regularization for D? None = disable lazy regularization.

        # Optimizers
        local_G_opt_kwargs=None,  # Options for local generator optimizer.
        local_D_opt_kwargs=None,  # Options for local discriminator optimizer.
        global_G_opt_kwargs=None,  # Options for pos estimator optimizer.
        global_D_opt_kwargs=None,  # Options for global discriminator optimizer.

        # Loss
        loss_kwargs=None,  # Options for loss function.
        metrics=None,  # Metrics to evaluate during training.

        # Augment
        augment_kwargs=None,  # Options for augmentation pipeline. None = disable.
        augment_p=0,  # Initial value of augmentation probability.
        ada_target=None,  # ADA target value. None = fixed p.
        ada_interval=4,  # How often to perform ADA adjustment?
        ada_kimg=500,
        # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.

        # Resume
        resume_pkl=None,  # Network pickle to resume training from.
        resume_kimg=0.0,  # Assumed training progress at the beginning. Affects reporting and training schedule.

        # Misc
        abort_fn=None,
        # Callback function for determining whether to abort training. Must return consistent results across ranks.
        progress_fn=None,  # Callback function for updating training progress. Called for all ranks.
):
    # Ensure that all required kwargs are defined.
    assert all([
        # Networks
        mapping_kwargs,
        local_G_kwargs,
        local_D_kwargs,
        pos_estimator_kwargs,
        global_D_kwargs,
        # Dataset
        training_set_kwargs,
        # Optimizers
        local_G_opt_kwargs,
        local_D_opt_kwargs,
        global_G_opt_kwargs,
        global_D_opt_kwargs,
        # Loss
        loss_kwargs,
    ])

    # Apply conv_base_index parameter.
    local_G_kwargs.conv_base_index = local_D_kwargs.conv_base_index = global_D_kwargs.conv_base_index = \
        training_set_kwargs.conv_base_index = config["conv_base_index"]

    # Apply config to loss kwargs.
    loss_kwargs.global_d_real_use_renderer = config["global_d_real_use_renderer"]
    loss_kwargs.renderer_retrain_use_real = config["renderer_retrain_use_real"]

    # Initialize.
    start_time = time.time()
    device = torch.device("cuda", rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    cudnn_benchmark = True  # Enabled by default in SG2ada
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    allow_tf32 = False  # Disabled by default in SG2ada
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.

    # Load training set.
    training_set: DatasetAIO = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus,
                                                seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler,
                                                             batch_size=batch_size // num_gpus, **data_loader_kwargs))

    # Construct networks.
    def init_module(m):
        # .requires_grad_(False) at first (to conserve GPU memory)
        return m.train().requires_grad_(False).to(device)

    layer_names = training_set.layer_names
    layer_sizes = [training_set.base_size_layer(layer_name) for layer_name in layer_names]
    num_channels = training_set.num_channels
    num_layers = training_set.num_layers
    local_G_list, local_D_list = [], []
    global_G_ema = {
        "mapping_network": None,
        "local_G_list": [],
        "pos_estimator": None,
    }
    for layer_name in layer_names:
        init_res_layer = training_set.init_res_layer(layer_name)
        res_layer = training_set.resolution_layer(layer_name)
        local_kwargs = {
            "img_resolution": res_layer,
            "img_channels": num_channels,
            "init_res": init_res_layer,
        }
        local_G: SynthesisNetwork = init_module(dnnlib.util.construct_class_by_name(**local_G_kwargs, **local_kwargs))
        local_G_list.append(local_G)
        global_G_ema["local_G_list"].append(copy.deepcopy(local_G).eval())

        local_D: Discriminator = init_module(
            dnnlib.util.construct_class_by_name(**local_D_kwargs, **local_kwargs))
        local_D_list.append(local_D)

    max_ws_num = max([local_G.num_ws for local_G in local_G_list])
    mapping_network: MappingNetwork = init_module(
        dnnlib.util.construct_class_by_name(**mapping_kwargs, c_dim=0, num_ws=max_ws_num))
    global_G_ema["mapping_network"] = copy.deepcopy(mapping_network).eval()

    init_res, res_log2, res = training_set.init_res, training_set.res_log2, training_set.resolution
    stn_kwargs = {
        "img_resolution": res,
        "img_channels": num_channels,
        "img_layers": num_layers,
    }

    if config["train_global"]:
        pos_estimator: STN = init_module(
            dnnlib.util.construct_class_by_name(**pos_estimator_kwargs, **stn_kwargs))
        global_G_ema["pos_estimator"] = copy.deepcopy(pos_estimator).eval()
        global_kwargs = {"img_resolution": res,
                         "img_channels": num_channels,
                         "init_res": init_res,
                         }
        global_D: Discriminator = init_module(dnnlib.util.construct_class_by_name(**global_D_kwargs, **global_kwargs))
    else:
        pos_estimator = None
        global_D = None

    def copy_params_and_buffers(modules_src: Union[torch.nn.Module, List[torch.nn.Module]],
                                modules_dst: Union[torch.nn.Module, List[torch.nn.Module]]):
        assert type(modules_src) == type(modules_dst)
        is_list = isinstance(modules_src, list)

        if is_list:
            assert len(modules_src) == len(modules_dst)
            _ = [misc.copy_params_and_buffers(m_src, m_dst, require_all=False) for m_src, m_dst in
                 zip(modules_src, modules_dst)]
        else:
            misc.copy_params_and_buffers(modules_src, modules_dst, require_all=False)

    def load_state_dicts(modules: Union[torch.nn.Module, List[torch.nn.Module]],
                         state_dicts: Union[Dict, List[Dict]]):
        is_list = isinstance(modules, list)

        if is_list:
            assert len(modules) == len(state_dicts)
            _ = [module.load_state_dict(state_dict) for module, state_dict in zip(modules, state_dicts)]
        else:
            modules.load_state_dict(state_dicts)

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = torch.load(f)
        modules = [("mapping_network", mapping_network)]
        modules += [("local_G_list", local_G_list)]
        modules += [("local_D_list", local_D_list)]
        if not renderer_config["bypass_renderer"]:
            modules += [("renderer", renderer)]
        modules += [("mapping_network_ema", global_G_ema["mapping_network"])]
        modules += [("local_G_ema", global_G_ema["local_G_list"])]
        if config["train_global"]:
            modules += [("pos_estimator", pos_estimator)]
            modules += [("pos_estimator_ema", global_G_ema["pos_estimator"])]
            modules += [("global_D", global_D)]
        for name, module in modules:
            data = resume_data.get(name, None)
            if data is None:
                print(f"Skip resuming {name} as no data is found.")
                continue
            load_state_dicts(module, resume_data[name])
            # Deprecated in favor of PyTorch default load/save function
            # copy_params_and_buffers(resume_data[name], module)

    # Setup augmentation.
    # Local augment pipeline.
    augment_pipe_list = []
    ada_stats_list = []
    for layer_name in layer_names:
        augment_pipe = None
        ada_stats = None
        if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None) and not config["local_noaug"]:
            augment_pipe: Optional[AugmentPipe] = init_module(dnnlib.util.construct_class_by_name(**augment_kwargs))
            augment_pipe.p.copy_(torch.as_tensor(augment_p))
            if ada_target is not None:
                ada_stats = training_stats.Collector(regex=f'{layer_name}/Loss/signs/real')
        augment_pipe_list.append(augment_pipe)
        ada_stats_list.append(ada_stats)
    # Global augment pipeline.
    global_augment_pipe = None
    global_ada_stats = None
    if config["train_global"]:
        if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None) and not config["global_noaug"]:
            global_augment_pipe = init_module(dnnlib.util.construct_class_by_name(**augment_kwargs))
            global_augment_pipe.p.copy_(torch.as_tensor(augment_p))
            if ada_target is not None:
                global_ada_stats = training_stats.Collector(regex=f'global/Loss/signs/real')

    # Distribute across GPUs.
    ddp_modules = dict()
    modules = [("mapping_network", mapping_network)]
    modules += [("local_G_list", local_G_list)]
    modules += [("local_D_list", local_D_list)]
    if not renderer_config["bypass_renderer"]:
        modules += [("renderer", renderer)]
    modules += [(None, global_G_ema["mapping_network"])]
    modules += [(None, global_G_ema["local_G_list"])]
    modules += [("augment_pipe_list", augment_pipe_list)]
    if config["train_global"]:
        modules += [("pos_estimator", pos_estimator)]
        modules += [(None, global_G_ema["pos_estimator"])]
        modules += [("global_D", global_D)]
        modules += [("global_augment_pipe", global_augment_pipe)]

    def distribute_across_gpus(modules: Union[torch.nn.Module, List[torch.nn.Module]]):
        is_list = isinstance(modules, list)

        def process(module: torch.nn.Module):
            if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
                module.requires_grad_(True)
                module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
                module.requires_grad_(False)

        if is_list:
            _ = [process(module) for module in modules]
        else:
            process(modules)

    for name, module in modules:
        if name is None:
            # global_G_ema modules, exclude from the dpp_modules list.
            distribute_across_gpus(module)
        else:
            distribute_across_gpus(module)
            ddp_modules[name] = module

    phases = []
    if renderer_config["bypass_renderer"]:
        loss = dnnlib.util.construct_class_by_name(device=device,
                                                   layer_names=layer_names,
                                                   **ddp_modules,
                                                   **loss_kwargs)  # subclass of training.loss.Loss
    else:
        loss = dnnlib.util.construct_class_by_name(device=device,
                                                   layer_names=layer_names,
                                                   criterion_renderer=criterion_renderer,
                                                   **ddp_modules,
                                                   **loss_kwargs)  # subclass of training.loss.Loss
        # Append renderer update as the first phase
        if config["train_renderer"]:
            phases += [dnnlib.EasyDict(name="Renderer", module=renderer, opt=optimizer_renderer, interval=1)]

    # Local GANs update phases
    for layer_name, local_G, local_D in zip(layer_names, local_G_list, local_D_list):
        for name, local_module, local_opt_kwargs, local_reg_interval in [
            ("local_G", [mapping_network, local_G], local_G_opt_kwargs, G_reg_interval),
            ("local_D", local_D, local_D_opt_kwargs, D_reg_interval)]:

            if name == "local_G":
                local_parameters = chain(*[m.parameters() for m in local_module])
            else:
                local_parameters = local_module.parameters()

            if local_reg_interval is None:
                opt = dnnlib.util.construct_class_by_name(params=local_parameters,
                                                          **local_opt_kwargs
                                                          )  # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name + f'both_{layer_name}', module=local_module, opt=opt, interval=1)]
            else:  # Lazy regularization.
                mb_ratio = local_reg_interval / (local_reg_interval + 1)
                opt_kwargs = dnnlib.EasyDict(local_opt_kwargs)
                opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                opt = dnnlib.util.construct_class_by_name(local_parameters,
                                                          **opt_kwargs)  # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name + f'main_{layer_name}', module=local_module, opt=opt, interval=1)]
                phases += [
                    dnnlib.EasyDict(name=name + f'reg_{layer_name}', module=local_module, opt=opt,
                                    interval=local_reg_interval)]

    if config["train_global"]:
        # Global GAN update after local GANs' phase
        for name, global_module, global_opt_kwargs, global_reg_interval in [
            ('global_G', [mapping_network, *local_G_list, pos_estimator], global_G_opt_kwargs, G_reg_interval),
            ('global_D', global_D, global_D_opt_kwargs, D_reg_interval)]:

            if name == "global_G":
                global_parameters = chain(*[m.parameters() for m in global_module])
            else:
                global_parameters = global_module.parameters()

            if global_reg_interval is None:
                opt = dnnlib.util.construct_class_by_name(params=global_parameters,
                                                          **global_opt_kwargs)  # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name + 'both', module=global_module, opt=opt, interval=1)]
            else:  # Lazy regularization.
                mb_ratio = global_reg_interval / (global_reg_interval + 1)
                opt_kwargs = dnnlib.EasyDict(global_opt_kwargs)
                opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                opt = dnnlib.util.construct_class_by_name(global_parameters,
                                                          **opt_kwargs)  # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name + 'main', module=global_module, opt=opt, interval=1)]
                phases += [
                    dnnlib.EasyDict(name=name + 'reg', module=global_module, opt=opt, interval=global_reg_interval)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_z = None
    grid_size = 32
    if rank == 0:
        rnd = np.random.RandomState(0)
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(grid_size)]
        # Load data.
        batch = torch.stack([training_set[i] for i in grid_indices])  # B,L,C,H,W [0,1]
        images = alpha_composite(batch)  # B,C,H,W [0,1]
        save_image(images, os.path.join(run_dir, 'reals.png'))
        # Generate grid_z for the first time.
        grid_z = torch.randn([grid_size, mapping_network.z_dim], device=device).split(batch_gpu)

    # Initialize logs.
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    def set_requires_grad_(modules: Union[torch.nn.Module, List[torch.nn.Module]], requires_grad: bool):
        if isinstance(modules, list):
            _ = [m.requires_grad_(requires_grad) for m in modules]
        else:
            modules.requires_grad_(requires_grad)

    def param_grad_nan_to_num(modules: Union[torch.nn.Module, List[torch.nn.Module]]):
        is_list = isinstance(modules, list)

        def process(module: torch.nn.Module):
            for param in module.parameters():
                if param.grad is not None:
                    misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        if is_list:
            _ = [process(module) for module in modules]
        else:
            process(modules)

    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            data = next(training_set_iterator)  # B,L,C,H,W
            phase_real_blchw = normalize_minus11(data.detach().clone()).to(device).split(batch_gpu)
            phase_real_list_of_bchw = [
                make_batch_for_local_d(blchw, layer_size_list=layer_sizes, to_minus11=True) for blchw in
                data.split(batch_gpu)]

            all_gen_z = torch.randn([len(phases) * batch_size, mapping_network.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z in zip(phases, all_gen_z):
            if config["debug"]:
                print(phase.name)
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            set_requires_grad_(phase.module, True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (real_layer, gen_z, real_list_of_bchw) in enumerate(
                    zip(phase_real_blchw, phase_gen_z, phase_real_list_of_bchw)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, real_blchw=real_layer,
                                          real_list_of_bchw=real_list_of_bchw, gen_z=gen_z,
                                          sync=sync, gain=gain)

            # Update weights.
            set_requires_grad_(phase.module, False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                param_grad_nan_to_num(phase.module)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

            # Manually empty GPU cache (Hopefully it can reduce some GPU memory usage?)
            torch.cuda.empty_cache()

        def update_ema(modules_ema: Union[torch.nn.Module, List[torch.nn.Module]],
                       modules: Union[torch.nn.Module, List[torch.nn.Module]], ema_beta):
            assert type(modules_ema) == type(modules)
            is_list = isinstance(modules_ema, list)

            def process(module_ema: torch.nn.Module, module: torch.nn.Module):
                for p_ema, p in zip(module_ema.parameters(), module.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(module_ema.buffers(), module.buffers()):
                    b_ema.copy_(b)

            if is_list:
                assert len(modules_ema) == len(modules)
                _ = [process(module_ema, module) for module_ema, module in zip(modules_ema, modules)]
            else:
                process(modules_ema, modules)

        # Update global_G_ema.
        with torch.autograd.profiler.record_function('global_G_ema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            update_ema(global_G_ema["mapping_network"], mapping_network, ema_beta)
            update_ema(global_G_ema["local_G_list"], local_G_list, ema_beta)
            if config["train_global"]:
                update_ema(global_G_ema["pos_estimator"], pos_estimator, ema_beta)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        for layer_name, ada_stats, augment_pipe in zip(layer_names + ["global"], ada_stats_list + [global_ada_stats],
                                                       augment_pipe_list + [global_augment_pipe]):
            if (ada_stats is not None) and (batch_idx % ada_interval == 0):
                ada_stats.update()
                adjust = np.sign(ada_stats[f'{layer_name}/Loss/signs/real'] - ada_target) * (
                        batch_size * ada_interval) / (
                                 ada_kimg * 1000)
                # Ensure that p doesn't go up too far.
                augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)).min(
                    misc.constant(0.6, device=device)))
                # augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2 ** 30):<6.2f}"]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2 ** 30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        for layer_name, augment_pipe in zip(layer_names + ["global"], augment_pipe_list + [global_augment_pipe]):
            fields += [
                f"aug_{layer_name} {training_stats.report0(f'Progress/augment_{layer_name}', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        def generate_sample_ema(z):
            mapping_network_ema = global_G_ema["mapping_network"]
            local_G_list_ema = global_G_ema["local_G_list"]
            if config["train_global"]:
                pos_estimator_ema = global_G_ema["pos_estimator"]

            ws = mapping_network_ema(z=z, c=torch.empty(size=(len(z), 0)))

            local_G_output_list = [G_ema(ws=ws[:, :G_ema.num_ws], noise_mode='const') for G_ema in local_G_list_ema]

            fake_layer = make_batch_for_pos_estimator(local_G_output_list, pad_value=-1)  # B,L,C,H,W [-1,1]
            if not config["train_global"]:
                return fake_layer
            transformed_fake_layer, _ = pos_estimator_ema(fake_layer)  # B,L,C,H,W [-1,1]
            return transformed_fake_layer

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            batch = torch.cat([generate_sample_ema(z=z).cpu() for z in grid_z])  # B,L,C,H,W
            # SG2ada's generator might exceed the data range a little bit, therefore applying a clip before output
            batch = torch.clip(batch, -1, 1)  # B,L,C,H,W [-1,1]
            batch = normalize_zero1(batch)  # B,L,C,H,W [0,1]
            # Save each layer
            b, l, c, h, w = batch.shape
            save_image(torch.reshape(batch, (b * l, c, h, w))[:4 * renderer_config["img_layers"]],
                       os.path.join(run_dir, f'fakes-layer{cur_nimg // 1000:06d}.png'),
                       nrow=renderer_config["img_layers"])
            # Save blended
            images = alpha_composite(batch)  # B,C,H,W [0,1]
            save_image(images, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}.png'))

        # Save network snapshot.
        def get_module_for_snapshot(modules: Union[torch.nn.Module, List[torch.nn.Module]]):
            is_list = isinstance(modules, list)

            def process(module: torch.nn.Module):
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    return copy.deepcopy(module).eval().requires_grad_(False).cpu()

            if is_list:
                return [process(module) for module in modules]
            else:
                return process(modules)

        def get_state_dicts(modules: Union[torch.nn.Module, List[torch.nn.Module]]):
            is_list = isinstance(modules, list)

            def process(module: torch.nn.Module):
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    return copy.deepcopy(module.state_dict())

            if is_list:
                return [process(module) for module in modules]
            else:
                return process(modules)

        snapshot_pth = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            modules = [("mapping_network", mapping_network)]
            modules += [("local_G_list", local_G_list)]
            modules += [("local_D_list", local_D_list)]
            if not renderer_config["bypass_renderer"]:
                modules += [("renderer", renderer)]
            modules += [("mapping_network_ema", global_G_ema["mapping_network"])]
            modules += [("local_G_ema", global_G_ema["local_G_list"])]
            modules += [("augment_pipe_list", augment_pipe_list)]
            if config["train_global"]:
                modules += [("pos_estimator", pos_estimator)]
                modules += [("pos_estimator_ema", global_G_ema["pos_estimator"])]
                modules += [("global_augment_pipe", global_augment_pipe)]
                modules += [("global_D", global_D)]

            for name, module in modules:
                snapshot_data[name] = get_state_dicts(module)
                # Deprecated in favor of PyTorch default load/save function
                # module = get_module_for_snapshot(module)
                # snapshot_data[name] = module
                # del module  # conserve memory
            snapshot_pth = os.path.join(run_dir, f'network-snapshot-{cur_nimg // 1000:06d}.pth')
            if rank == 0:
                torch.save(snapshot_data, snapshot_pth)

            del snapshot_data  # conserve memory

        snapshot_data = dict()
        if config["train_global"] and (len(metrics) > 0):
            modules = [("mapping_network_ema", global_G_ema["mapping_network"])]
            modules += [("local_G_ema", global_G_ema["local_G_list"])]
            modules += [("pos_estimator_ema", global_G_ema["pos_estimator"])]
            for name, module in modules:
                snapshot_data[name] = get_module_for_snapshot(module)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data,
                                                      dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank,
                                                      device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pth)
                stats_metrics.update(result_dict.results)

        del snapshot_data  # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')
