import os
from collections import OrderedDict

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls


def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    own_state = module.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            unexpected_keys.append(name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        try:
            own_state[name].copy_(param)
        except Exception:
            raise RuntimeError('While copying the parameter named {}, '
                               'whose dimensions in the model are {} and '
                               'whose dimensions in the checkpoint are {}.'
                               .format(name, own_state[name].size(),
                                       param.size()))
    missing_keys = set(own_state.keys()) - set(state_dict.keys())

    err_msgs = []
    if unexpected_keys:
        err_msgs.append('unexpected key in source state_dict: {}\n'.format(
            ', '.join(unexpected_keys)))
    if missing_keys:
        err_msgs.append('missing keys in source state_dict: {}\n'.format(
            ', '.join(missing_keys)))
    msg = '\n'.join(err_msgs)
    if msg:
        if strict:
            raise RuntimeError(msg)
        elif logger is not None:
            logger.warn(msg)
        else:
            print(msg)


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None):
    # load checkpoint from modelzoo or file or url
    if filename.startswith('modelzoo://'):
        model_name = filename[11:]
        checkpoint = model_zoo.load_url(model_urls[model_name])
    elif filename.startswith(('http://', 'https://')):
        checkpoint = model_zoo.load_url(filename)
    else:
        if not os.path.isfile(filename):
            raise IOError('{} is not a checkpoint file'.format(filename))
        checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    # load state_dict
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def make_link(filename, link):
    if os.path.islink(link):
        os.remove(link)
    os.symlink(filename, link)


def model_weights_to_cpu(state_dict):
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def save_checkpoint(model,
                    epoch,
                    num_iters,
                    out_dir,
                    filename_tmpl='epoch_{}.pth',
                    optimizer=None,
                    is_best=False):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module
    filename = os.path.join(out_dir, filename_tmpl.format(epoch))
    checkpoint = {
        'epoch': epoch,
        'num_iters': num_iters,
        'state_dict': model_weights_to_cpu(model.state_dict())
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    torch.save(checkpoint, filename)
    latest_link = os.path.join(out_dir, 'latest.pth')
    make_link(filename, latest_link)
    if is_best:
        best_link = os.path.join(out_dir, 'best.pth')
        make_link(filename, best_link)
