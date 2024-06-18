from torchpack.io import save_checkpoint
from .hook import Hook
from ..utils import master_only


class CheckpointSaverHook(Hook):

    def __init__(self,
                 interval=-1,
                 save_optimizer=True,
                 out_dir=None,
                 **kwargs):
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.args = kwargs

    @master_only
    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        if not self.out_dir:
            self.out_dir = runner.work_dir
        optimizer = runner.optimizer if self.save_optimizer else None
        save_checkpoint(
            model=runner.model,
            epoch=runner.epoch + 1,
            num_iters=runner.num_iters,
            out_dir=self.out_dir,
            optimizer=optimizer,
            **self.args)
