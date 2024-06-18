from __future__ import division

from .hook import Hook


class LrUpdaterHook(Hook):

    def __init__(self,
                 by_epoch=True,
                 warm_up=None,
                 warm_up_iters=0,
                 warm_up_ratio=0.1,
                 **kwargs):
        # validate the "warm_up" argument
        if warm_up is not None:
            if warm_up not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    '"{}" is not a supported type for warming up, valid types'
                    ' are "constant" and "linear"'.format(warm_up))
        if warm_up is not None:
            assert warm_up_iters > 0, \
                '"warm_up_iters" must be a positive integer'
            assert 0 < warm_up_ratio <= 1.0, \
                '"warm_up_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warm_up = warm_up
        self.warm_up_iters = warm_up_iters
        self.warm_up_ratio = warm_up_ratio

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed

    def _set_lr(self, runner, lr_groups):
        for param_group, lr in zip(runner.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr

    def get_lr(self, runner, base_lr):
        raise NotImplementedError

    def get_regular_lr(self, runner):
        return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):
        if self.warm_up == 'constant':
            warmup_lr = [_lr * self.warm_up_ratio for _lr in self.regular_lr]
        elif self.warm_up == 'linear':
            k = (1 - cur_iters / self.warm_up_iters) * (1 - self.warm_up_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
        elif self.warm_up == 'exp':
            k = self.warm_up_ratio**(1 - cur_iters / self.warm_up_iters)
            warmup_lr = [_lr * k for _lr in self.regular_lr]
        return warmup_lr

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        for group in runner.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [
            group['initial_lr'] for group in runner.optimizer.param_groups
        ]

    def before_train_epoch(self, runner):
        if not self.by_epoch:
            return
        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner):
        cur_iters = runner.num_iters
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warm_up is None or cur_iters >= self.warm_up_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iters)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            if self.warm_up is None or cur_iters > self.warm_up_iters:
                return
            elif cur_iters == self.warm_up_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iters)
                self._set_lr(runner, warmup_lr)


class FixedLrUpdaterHook(LrUpdaterHook):

    def __init__(self, **kwargs):
        super(FixedLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        return base_lr


class StepLrUpdaterHook(LrUpdaterHook):

    def __init__(self, step, gamma=0.1, **kwargs):
        assert isinstance(step, (list, int))
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int) and s > 0
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        super(StepLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.num_iters

        if isinstance(self.step, int):
            return base_lr * (self.gamma**(progress // self.step))

        exp = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break
        return base_lr * self.gamma**exp


class ExpLrUpdaterHook(LrUpdaterHook):

    def __init__(self, gamma, **kwargs):
        self.gamma = gamma
        super(ExpLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.num_iters
        return base_lr * self.gamma**progress


class PolyLrUpdaterHook(LrUpdaterHook):

    def __init__(self, power=1., **kwargs):
        self.power = power
        super(PolyLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epoch
        else:
            progress = runner.num_iters
            max_progress = runner.max_iter
        return base_lr * (1 - progress / max_progress)**self.power


class InvLrUpdaterHook(LrUpdaterHook):

    def __init__(self, gamma, power=1., **kwargs):
        self.gamma = gamma
        self.power = power
        super(InvLrUpdaterHook, self).__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.num_iters
        return base_lr * (1 + self.gamma * progress)**(-self.power)
