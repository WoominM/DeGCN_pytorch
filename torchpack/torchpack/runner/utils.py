import functools
from collections import defaultdict
from getpass import getuser
from socket import gethostname

import torch.distributed as dist


def get_host_info():
    return '{}@{}'.format(getuser(), gethostname())


def get_dist_info():
    if dist._initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = defaultdict(float)
        self.sum = defaultdict(float)
        self.avg = defaultdict(float)
        self.count = defaultdict(int)
        self.reset()

    def reset(self, keys=None):
        if isinstance(keys, str):
            keys = [keys]
        elif keys is None:
            keys = self.val.keys()
        for k in keys:
            self.val[k] = 0
            self.sum[k] = 0
            self.avg[k] = 0
            self.count[k] = 0

    def update(self, pairs, n=1):
        for k, v in pairs.items():
            self.val[k] = v
            self.sum[k] += v * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]
