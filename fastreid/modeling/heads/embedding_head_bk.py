# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from torch import nn
import torch
from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class EmbeddingHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        with_bnneck = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type = cfg.MODEL.HEADS.NORM

        if pool_type == 'fastavgpool':
            self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':
            self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':
            self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':
            self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':
            self.pool_layer = GeneralizedMeanPooling()
        elif pool_type == "avgmaxpool":
            self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool':
            self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == "identity":
            self.pool_layer = nn.Identity()
        elif pool_type == "flatten":
            self.pool_layer = Flatten()
        else:
            raise KeyError(f"{pool_type} is not supported!")
        # fmt: on

        self.neck_feat = neck_feat

        bottleneck = []
        if embedding_dim > 0:
            bottleneck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            bottleneck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck = nn.Sequential(*bottleneck)

        # identity classification layer
        # fmt: off
        if cls_type == 'linear':
            self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':
            self.classifier = ArcSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'circleSoftmax':
            self.classifier = CircleSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'cosSoftmax':
            self.classifier = CosSoftmax(cfg, feat_dim, num_classes)
        else:
            raise KeyError(f"{cls_type} is not supported!")
        # fmt: on

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None, fc_dim=None, pass_head=False):
        """
        See :class:`ReIDHeads.forward`.
        """

        if not pass_head:
            global_feat = self.pool_layer(features)

            if fc_dim != None:
                if fc_dim <= global_feat.size(1):
                    t_global_feat = global_feat[:, :fc_dim, :, :]
                else:
                    z = torch.zeros(global_feat.size(0), fc_dim - global_feat.size(1), 1, 1)
                    z = z.cuda() if torch.cuda.is_available() else z
                    t_global_feat = torch.cat((global_feat, z), 1)
            else:
                t_global_feat = global_feat

            bn_feat = self.bottleneck(t_global_feat)
            bn_feat = bn_feat[..., 0, 0]

            # Evaluation
            # fmt: off
            if not self.training: return bn_feat
            # fmt: on
        else:
            bn_feat = features
            if fc_dim != None:
                if fc_dim <= bn_feat.size(1):
                    bn_feat = bn_feat[:, :fc_dim]
                else:
                    z = torch.zeros(bn_feat.size(0), fc_dim - bn_feat.size(1))
                    z = z.cuda() if torch.cuda.is_available() else z
                    bn_feat = torch.cat((bn_feat, z), 1)

        # Training
        if self.classifier.__class__.__name__ == 'Linear':
            cls_outputs = self.classifier(bn_feat)
            pred_class_logits = F.linear(bn_feat, self.classifier.weight)
        else:
            cls_outputs = self.classifier(bn_feat, targets)
            pred_class_logits = self.classifier.s * F.linear(F.normalize(bn_feat),
                                                             F.normalize(self.classifier.weight))

        # fmt: off
        if self.neck_feat == "before" and not pass_head:
            feat = global_feat[..., 0, 0]
        elif self.neck_feat == "before" and pass_head:
            feat = bn_feat
        elif self.neck_feat == "after":
            feat = bn_feat
        else:
            raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": pred_class_logits,
            "features": feat,
            "bn_feat": bn_feat
        }


@REID_HEADS_REGISTRY.register()
class DisentangleHead(nn.Module):
    def __init__(self, cfg, cfg_old):
        super().__init__()
        # fmt: off
        feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        disentangle_dim1 = cfg.MODEL.HEADS.DISENTANGLE_DIM1
        disentangle_dim2 = cfg.MODEL.HEADS.DISENTANGLE_DIM2
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        with_bnneck = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type = cfg.MODEL.HEADS.NORM
        self.harf = cfg.MODEL.HEADS.DISENTANGLE_HARF
        old_num_classes = cfg_old.MODEL.HEADS.NUM_CLASSES

        if pool_type == 'fastavgpool':
            self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':
            self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':
            self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':
            self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':
            self.pool_layer = GeneralizedMeanPooling()
        elif pool_type == "avgmaxpool":
            self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool':
            self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == "identity":
            self.pool_layer = nn.Identity()
        elif pool_type == "flatten":
            self.pool_layer = Flatten()
        else:
            raise KeyError(f"{pool_type} is not supported!")
        # fmt: on

        self.neck_feat = neck_feat

        if disentangle_dim1 > 0:
            if self.harf: feat_dim1 = feat_dim//2
            else: feat_dim1 = feat_dim
            self.bottleneck_d1 = nn.Conv2d(feat_dim1, disentangle_dim1, 1, 1, bias=False)
            self.disentangle_conv1 = True
        else:
            self.disentangle_conv1 = False
            disentangle_dim1 = feat_dim
        if with_bnneck:
            self.bnneck_d1 = get_norm(norm_type, disentangle_dim1, bias_freeze=True)

        if disentangle_dim2 > 0:
            if self.harf: feat_dim2 = feat_dim//2
            else: feat_dim2 = feat_dim
            self.bottleneck_d2 = nn.Conv2d(feat_dim2, disentangle_dim2, 1, 1, bias=False)
            self.disentangle_conv2 = True
        else:
            self.disentangle_conv2 = False
            disentangle_dim2 = feat_dim
        if with_bnneck:
            self.bnneck_d2 = get_norm(norm_type, disentangle_dim2, bias_freeze=True)

        bottleneck = []
        if embedding_dim > 0:
            #bottleneck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            bottleneck.append(nn.Conv2d(disentangle_dim1 + disentangle_dim2, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim
        else:
            feat_dim = disentangle_dim1 + disentangle_dim2
        if with_bnneck:
            bottleneck.append(get_norm(norm_type, feat_dim, bias_freeze=True))
        self.bottleneck = nn.Sequential(*bottleneck)


        # identity classification layer
        # fmt: off
        if cls_type == 'linear':
            self.classifier_1 = nn.Linear(disentangle_dim1, old_num_classes, bias=False)
            self.classifier_2 = nn.Linear(disentangle_dim2, num_classes, bias=False)
            self.classifier_all = nn.Linear(disentangle_dim1+disentangle_dim2, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':
            self.classifier_1 = ArcSoftmax(cfg, disentangle_dim1, old_num_classes)
            self.classifier_2 = ArcSoftmax(cfg, disentangle_dim2, num_classes)
            self.classifier_all = ArcSoftmax(cfg, disentangle_dim1+disentangle_dim2, num_classes)
        elif cls_type == 'circleSoftmax':
            self.classifier_1 = CircleSoftmax(cfg, disentangle_dim1, old_num_classes)
            self.classifier_2 = CircleSoftmax(cfg, disentangle_dim2, num_classes)
            self.classifier_all = CircleSoftmax(cfg, disentangle_dim1+disentangle_dim2, num_classes)
        elif cls_type == 'cosSoftmax':
            self.classifier_1 = CosSoftmax(cfg, disentangle_dim1, old_num_classes)
            self.classifier_2 = CosSoftmax(cfg, disentangle_dim2, num_classes)
            self.classifier_all = CosSoftmax(cfg, disentangle_dim1+disentangle_dim2, num_classes)
        else:
            raise KeyError(f"{cls_type} is not supported!")
        # fmt: on

        self.bottleneck.apply(weights_init_kaiming)
        if self.disentangle_conv1:
            self.bottleneck_d1.apply(weights_init_kaiming)
            self.bottleneck_d2.apply(weights_init_kaiming)

        self.classifier_1.apply(weights_init_classifier)
        if self.disentangle_conv2:
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_all.apply(weights_init_classifier)


    def forward(self, features, targets=None, fc_dim=None, pass_head=False,compare_old=False):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        #print(global_feat.shape)
        if self.harf:
            global_feat1 = global_feat[: , :global_feat.size(1)//2,: ,: ]
            global_feat2 = global_feat[: , global_feat.size(1)//2:global_feat.size(1),: ,: ]
        else:
            global_feat1 = global_feat
            global_feat2 = global_feat
        if self.disentangle_conv1:
            d1_feat = self.bottleneck_d1(global_feat1)
        else:
            d1_feat = global_feat1
        d1_feat_bn = self.bnneck_d1(d1_feat)
        if self.disentangle_conv2:
            d2_feat = self.bottleneck_d2(global_feat2)
        else:
            d2_feat = global_feat2
        d2_feat_bn = self.bnneck_d2(d2_feat)
        #print(d1_feat.shape, d2_feat.shape, d1_feat_bn.shape)
        all_feat = torch.cat((d1_feat_bn, d2_feat_bn), 1)
        bn_feat = self.bottleneck(all_feat)

        d1_feat_bn = d1_feat_bn[..., 0, 0]
        d2_feat_bn = d2_feat_bn[..., 0, 0]
        all_feat = all_feat[..., 0, 0]
        bn_feat = bn_feat[..., 0, 0]
        #print('all_feat', all_feat)
        #print('bn_feat', bn_feat)
        #print('d1_feat_bn', d1_feat_bn)
        #print('d2_feat_bn', d2_feat_bn)
        #print('d1_feat', d1_feat[..., 0, 0])
        #print('d2_feat', d2_feat[..., 0, 0])

        # Evaluation
        # fmt: off
        if not self.training and not compare_old: return d2_feat_bn
        if not self.training and compare_old: return d1_feat_bn
        # fmt: on
        # Training
        if self.classifier_1.__class__.__name__ == 'Linear':
            cls_outputs_1 = self.classifier_1(d1_feat_bn)
            pred_class_logits_1 = F.linear(d1_feat_bn, self.classifier_1.weight)

            cls_outputs_2 = self.classifier_2(d2_feat_bn)
            pred_class_logits_2 = F.linear(d2_feat_bn, self.classifier_2.weight)

            cls_outputs_all = self.classifier_all(all_feat)
            pred_class_logits_all = F.linear(all_feat, self.classifier_all.weight)
        else:
            cls_outputs_1 = self.classifier_1(d1_feat_bn, targets)
            pred_class_logits_1 = self.classifier_1.s * F.linear(F.normalize(d1_feat_bn),
                                                             F.normalize(self.classifier_1.weight))
            cls_outputs_2 = self.classifier_2(d2_feat_bn, targets)
            pred_class_logits_2 = self.classifier_2.s * F.linear(F.normalize(d2_feat_bn),
                                                             F.normalize(self.classifier_2.weight))
            cls_outputs_all = self.classifier_all(all_feat, targets)
            pred_class_logits_all = self.classifier_all.s * F.linear(F.normalize(all_feat),
                                                             F.normalize(self.classifier_all.weight))

        return {
            "cls_outputs_1": cls_outputs_1,
            "cls_outputs_2": cls_outputs_2,
            "cls_outputs_all": cls_outputs_all,
            "pred_class_logits_1": pred_class_logits_1,
            "pred_class_logits_2": pred_class_logits_2,
            "pred_class_logits_all": pred_class_logits_all,
            "global_feat": global_feat[..., 0, 0],
            "bn_feat": bn_feat,
            "all_feat": all_feat,
            "d1_feat": d1_feat[..., 0, 0],
            "d2_feat": d2_feat[..., 0, 0],
            "d1_feat_bn": d1_feat_bn,
            "d2_feat_bn": d2_feat_bn,
        }

    def process_old(self, features, targets=None, fc_dim=None):
        d1_feat = features
        if fc_dim != None:
            if fc_dim <= d1_feat.size(1):
                bn_feat = d1_feat[:, :fc_dim]
            else:
                z = torch.zeros(d1_feat.size(0), fc_dim - d1_feat.size(1))
                z = z.cuda() if torch.cuda.is_available() else z
                bn_feat = torch.cat((d1_feat, z), 1)

        # Training
        if self.classifier_1.__class__.__name__ == 'Linear':
            cls_outputs_1 = self.classifier_1(d1_feat)
            pred_class_logits_1 = F.linear(d1_feat, self.classifier_1.weight)
        else:
            cls_outputs_1 = self.classifier_1(d1_feat, targets)
            pred_class_logits_1 = self.classifier_1.s * F.linear(F.normalize(d1_feat),
                                                                 F.normalize(self.classifier_1.weight))

        return {
            "cls_outputs_1": cls_outputs_1,
            "pred_class_logits_1": pred_class_logits_1,
            "features": d1_feat,
            "bn_feat": d1_feat
        }
