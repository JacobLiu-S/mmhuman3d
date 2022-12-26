# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import Registry

from .expose_head import ExPoseBodyHead, ExPoseFaceHead, ExPoseHandHead
from .hmr_head import HMRHead
from .hybrik_head import HybrIKHead
from .pare_head import PareHead
from .domain_cls import DomainClassifier

HEADS = Registry('heads')

HEADS.register_module(name='HybrIKHead', module=HybrIKHead)
HEADS.register_module(name='HMRHead', module=HMRHead)
HEADS.register_module(name='PareHead', module=PareHead)
HEADS.register_module(name='ExPoseBodyHead', module=ExPoseBodyHead)
HEADS.register_module(name='ExPoseHandHead', module=ExPoseHandHead)
HEADS.register_module(name='ExPoseFaceHead', module=ExPoseFaceHead)
HEADS.register_module(name='DomainClassifier', module=DomainClassifier)


def build_head(cfg):
    """Build head."""
    if cfg is None:
        return None
    return HEADS.build(cfg)
