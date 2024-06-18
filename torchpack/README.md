# torchpack (Deprecated! Please use [mmcv](https://github.com/open-mmlab/mmcv) instead.)

[![PyPI Version](https://img.shields.io/pypi/v/torchpack.svg)](https://pypi.python.org/pypi/torchpack)

Torchpack is a set of interfaces to simplify the usage of PyTorch.

Documentation is ongoing.


## Installation

- Install with pip. 
```
pip install torchpack
```
- Install from source.
```
git clone https://github.com/hellock/torchpack.git
cd torchpack
python setup.py install
```

**Note**: If you want to use tensorboard to visualize the training process, you need to
install tensorflow([`installation guide`](https://www.tensorflow.org/install/install_linux)) and tensorboardX(`pip install tensorboardX`).

## What can torchpack do

Torchpack aims to help users to start training with less code, while stays
flexible and configurable. It provides a `Runner` with lots of `Hooks`.

## Example

```python
######################## file1: config.py #######################
work_dir = './demo'  # dir to save log file and checkpoints
optimizer = dict(
    algorithm='SGD', args=dict(lr=0.001, momentum=0.9, weight_decay=5e-4))
workflow = [('train', 2), ('val', 1)]  # train 2 epochs and then validate 1 epochs, iteratively
max_epoch = 16
lr_policy = dict(policy='step', step=12)  # decrese learning rate by 10 every 12 epochs
checkpoint_cfg = dict(interval=1)  # save checkpoint at every epoch
log_cfg = dict(
    # log at every 50 iterations
    interval=50,
    # two logging hooks, one for printing in terminal and one for tensorboard visualization
    hooks=[
        ('TextLoggerHook', {}),
        ('TensorboardLoggerHook', dict(log_dir=work_dir + '/log'))
    ])

######################### file2: main.py ########################
import torch
from torchpack import Config, Runner
from collections import OrderedDict

# define how to process a batch and return a dict
def batch_processor(model, data, train_mode):
    img, label = data
    label = label.cuda(non_blocking=True)
    pred = model(img)
    loss = F.cross_entropy(pred, label)
    accuracy = get_accuracy(pred, label_var)
    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()
    log_vars['accuracy'] = accuracy.item()
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=img.size(0))
    return outputs

cfg = Config.from_file('config.py')  # or config.yaml/config.json
model = resnet18()
runner = Runner(model, cfg.optimizer, batch_processor, cfg.work_dir)
runner.register_default_hooks(lr_config=cfg.lr_policy,
                              checkpoint_config=cfg.checkpoint_cfg,
                              log_config=cfg.log_cfg)

runner.run([train_loader, val_loader], cfg.workflow, cfg.max_epoch)
```

For a full example of training on ImageNet, please see `examples/train_imagenet.py`.

```shell
python examples/train_imagenet.py examples/config.py
```
