from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SetIterInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_iter(self, runner):
        num_iter = runner._inner_iter
        max_iters = runner._max_iters
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        model.set_iter(num_iter, max_iters)