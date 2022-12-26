import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from mmcv.runner.base_module import BaseModule


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DomainClassifier(BaseModule):

    def __init__(self,
                 feat_dim,
                #  smpl_mean_params=None,
                #  npose=144,
                #  nbeta=10,
                #  ncam=3,
                 hdim=1024,
                 init_cfg=None):
        super(DomainClassifier, self).__init__(init_cfg=init_cfg)
        
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(feat_dim, hdim))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(hdim))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(hdim, hdim))
        self.domain_classifier.add_module('d_bn2', nn.BatchNorm1d(hdim))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc3', nn.Linear(hdim, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))


    def forward(self, x, alpha):

        # hmr head only support one layer feature
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[-1]

        output_seq = False
        if len(x.shape) == 4:
            # use feature from the last layer of the backbone
            # apply global average pooling on the feature map
            x = x.mean(dim=-1).mean(dim=-1)
        elif len(x.shape) == 3:
            # temporal feature
            output_seq = True
            B, T, L = x.shape
            x = x.view(-1, L)

        batch_size = x.shape[0]
        reverse_feature = ReverseLayerF.apply(x, alpha)
        domain_output = self.domain_classifier(reverse_feature)
        return domain_output
