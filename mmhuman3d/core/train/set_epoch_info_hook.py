# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        max_epochs = runner._max_epochs
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.set_epoch(epoch, max_epochs)