import logging
import os
import time

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from torchpack.io import load_checkpoint, save_checkpoint
from torchpack.runner.hooks import (Hook, LrUpdaterHook, CheckpointSaverHook,
                                    TimerHook, OptimizerStepperHook)
from torchpack.runner.log_buffer import LogBuffer
from torchpack.runner.utils import get_dist_info, get_host_info


class Runner(object):

    def __init__(self,
                 model,
                 optimizer,
                 batch_processor,
                 work_dir=None,
                 log_level=logging.INFO):
        self.model = model
        self.optimizer = self.set_optimizer(optimizer)
        assert callable(batch_processor)
        self.batch_processor = batch_processor

        self.rank, self.world_size = get_dist_info()

        if isinstance(work_dir, str):
            self.work_dir = os.path.abspath(work_dir)
            if not os.path.isdir(self.work_dir):
                os.makedirs(self.work_dir)
        elif work_dir is None:
            self.work_dir = work_dir
        else:
            raise TypeError('"work_dir" must be a str or None')

        self.logger = self.init_logger(work_dir, log_level)

        if isinstance(self.model, (DataParallel, DistributedDataParallel)):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self.log_buffer = LogBuffer()
        self.hooks = []
        self.max_epoch = 0
        self.max_iter = 0
        self.epoch = 0
        self.num_iters = 0
        self.num_epoch_iters = 0
        self.mode = None

    @property
    def model_name(self):
        return self._model_name

    def set_optimizer(self, optimizer):
        if isinstance(optimizer, dict):
            optim_cls = getattr(torch.optim, optimizer['algorithm'])
            optimizer = optim_cls(self.model.parameters(), **optimizer['args'])
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                '"optimizer" must be either an Optimizer object or a dict')
        return optimizer

    def init_logger(self, log_dir=None, level=logging.INFO):
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s', level=level)
        logger = logging.getLogger(__name__)
        if log_dir:
            filename = '{}_{}.log'.format(
                time.strftime('%Y%m%d_%H%M%S', time.localtime()), self.rank)
            log_file = os.path.join(log_dir, filename)
            file_handler = logging.FileHandler(log_file, 'w')
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
        return logger

    def current_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def register_hook(self, hook, hook_type=None, priority=50):
        assert isinstance(priority, int) and priority >= 0 and priority <= 100
        if hook_type is None:
            assert isinstance(hook, Hook)
        else:
            if isinstance(hook, dict):
                hook = hook_type(**hook)
            elif not isinstance(hook, hook_type):
                raise TypeError('hook must be a {} object or a dict'.format(
                    hook_type.__name__))
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self.hooks) - 1, -1, -1):
            if priority >= self.hooks[i].priority:
                self.hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self.hooks.insert(0, hook)

    def call_hook(self, fn_name):
        for hook in self.hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict,
                               self.logger)

    def save_checkpoint(self, out_dir, filename_tmpl='epoch_{}.pth'):
        save_checkpoint(
            self.model,
            self.epoch + 1,
            self.num_iters,
            out_dir=out_dir,
            filename_tmpl=filename_tmpl)

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self.max_iter = self.max_epoch * len(data_loader)
        self.call_hook('before_train_epoch')
        for i, data_batch in enumerate(data_loader):
            self.num_epoch_iters = i
            self.call_hook('before_train_iter')
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=True, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_train_iter')
            self.num_iters += 1
        self.call_hook('after_train_epoch')
        self.epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        for i, data_batch in enumerate(data_loader):
            self.num_epoch_iters = i
            self.call_hook('before_val_iter')
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=False, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_val_iter')
        self.call_hook('after_val_epoch')

    def resume(self, checkpoint, resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)
        self.epoch = checkpoint['epoch']
        self.num_iters = checkpoint['num_iters']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info('resumed epoch %d, iter %d', self.epoch,
                         self.num_iters)

    def run(self, data_loaders, workflow, max_epoch, **kwargs):
        assert isinstance(data_loaders, list)
        self.max_epoch = max_epoch
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epoch)
        self.call_hook('before_run')
        while self.epoch < max_epoch:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run an epoch'.
                            format(mode))
                    epoch_runner = getattr(self, mode)
                elif callable(mode):
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(
                                        type(mode)))
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= max_epoch:
                        return
                    epoch_runner(data_loaders[i], **kwargs)
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def register_lr_hooks(self, lr_config):
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config)
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            from .hooks import lr_updater
            hook_name = lr_config['policy'].title() + 'LrUpdaterHook'
            if not hasattr(lr_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lr_updater, hook_name)
            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError('"lr_config" must be either a LrUpdaterHook object'
                            ' or dict, not {}'.format(type(lr_config)))

    def register_logger_hooks(self, log_config):
        self.register_hook(TimerHook())
        log_interval = log_config['interval']
        from . import hooks
        for logger_name, args in log_config['hooks']:
            if isinstance(logger_name, str):
                logger_cls = getattr(hooks, logger_name)
            elif isinstance(logger_name, type):
                logger_cls = logger_name
            else:
                raise TypeError(
                    'logger name must be a string of hook type, not {}'.format(
                        logger_name))
            kwargs = args.copy()
            if 'interval' not in kwargs:
                kwargs['interval'] = log_interval
            self.register_hook(logger_cls(**kwargs), priority=60)

    def register_default_hooks(self,
                               lr_config,
                               grad_clip_config=None,
                               checkpoint_config=None,
                               log_config=None):
        """Register several default hooks"""
        if grad_clip_config is None:
            grad_clip_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}
        self.register_lr_hooks(lr_config)
        self.register_hook(grad_clip_config, OptimizerStepperHook)
        self.register_hook(checkpoint_config, CheckpointSaverHook)
        if log_config is not None:
            self.register_logger_hooks(log_config)
