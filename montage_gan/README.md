# Notes
The `train_aio.py` is the entry point of MontageGAN. Configuration is needed to start the training.

Besides, there are more configurations in the `training_loop_aio.py` and `dataset_aio.py`. Make sure you check them as well.

MontageGAN utilizes modules from SG2ada with modifications. You can find most of the modified and custom-made modules in the `custom`, `custom_utils`, `diff_rendering`, and `fukuwarai` directories. Also, the `pretrained` directory holds the pretrained model of our renderer module.

We have no plan of publishing our dataset due to copyright reasons. However, you can collect your dataset using the tool we made, available at https://github.com/jeffshee/live2d.
