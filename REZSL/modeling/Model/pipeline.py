import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal as MVN
from .resnet import resnet101_features
from .transformer import ViT, ViT1
from .mainModel import *
from .utils import get_attributes_info, get_attr_group

from os.path import join
import pickle
from contrastive_learning.builder import MoCo, my_SimCLR,my_SimCLR2,my_SimCLR3,my_SimCLR4,my_SimCLR5


def build_BasicNet(cfg):
    dataset_name = cfg.DATASETS.NAME
    att_type = cfg.DATASETS.SEMANTIC_TYPE
    info = get_attributes_info(dataset_name, att_type)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    scls_num = cls_num - ucls_num

    attr_group = get_attr_group(dataset_name)
    img_size = cfg.DATASETS.IMAGE_SIZE
    # res101 feature size
    hid_dim = cfg.MODEL.HID_DIM
    scale = cfg.MODEL.SCALE

    pretrained = cfg.MODEL.BACKBONE.PRETRAINED
    ft_flag = cfg.MODEL.BACKBONE.FINETUNE
    model_dir = cfg.PRETRAINED_MODELS

    backbone_type = cfg.MODEL.BACKBONE.TYPE
    if backbone_type == 'resnet':
        # res101 feature size
        c, w, h = 2048, img_size // 32, img_size // 32
        backbone = resnet101_features(pretrained=pretrained, model_dir=model_dir)
    elif backbone_type == 'vit':
        # vit feature size
        c, w, h = 768, img_size // 16, img_size // 16
        # c, w, h = 1024, img_size // 16, img_size // 16 #for vit_large_patch16_224_in21k
        if img_size == 224:
            backbone = ViT(model_name="vit_base_patch16_224", pretrained=pretrained)
            # backbone = ViT(model_name="vit_large_patch16_224_in21k", pretrained=pretrained)
        else:  # img_size == 384
            backbone = ViT(model_name="vit_base_patch16_384", pretrained=pretrained)

    device = torch.device(cfg.MODEL.DEVICE)

    return BasicNet(backbone=backbone, backbone_type=backbone_type, ft_flag=ft_flag, img_size=img_size, hid_dim=hid_dim,
                    c=c, w=w, h=h, scale=scale, attritube_num=attritube_num, cls_num=cls_num, ucls_num=ucls_num,
                    device=device)


def build_AttentionNet(cfg, contrastive_learning=False):
    dataset_name = cfg.DATASETS.NAME
    att_type = cfg.DATASETS.SEMANTIC_TYPE
    info = get_attributes_info(dataset_name, att_type)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    scls_num = cls_num - ucls_num

    attr_group = get_attr_group(dataset_name)

    img_size = cfg.DATASETS.IMAGE_SIZE
    dataset_name = cfg.DATASETS.NAME

    hid_dim = cfg.MODEL.HID_DIM
    scale = cfg.MODEL.SCALE
    pretrained = cfg.MODEL.BACKBONE.PRETRAINED
    ft_flag = cfg.MODEL.BACKBONE.FINETUNE
    model_dir = cfg.PRETRAINED_MODELS
    backbone_type = cfg.MODEL.BACKBONE.TYPE
    print(backbone_type)
    if backbone_type == 'resnet':
        # res101 feature size
        c, w, h = 2048, img_size // 32, img_size // 32
        backbone = resnet101_features(pretrained=pretrained, model_dir=model_dir)
    elif backbone_type == 'vit':
        # vit feature size
        # c, w, h = 768, img_size // 16, img_size // 16
        # c, w, h = 1024, img_size // 16, img_size // 16
        c, w, h = 768, img_size // 16, img_size // 16  # for vit_large_patch16_224_in21k
        model_name = "google/vit-large-patch16-224-in21k"
        model_name = 'google/vit-base-patch16-224'
        if model_name == 'google/vit-large-patch16-224-in21k':
            c, w, h = 1024, img_size // 16, img_size // 16
        if img_size == 224:
            # backbone = ViT(model_name="vit_base_patch16_224", pretrained=pretrained)
            # backbone = ViT(model_name="vit_large_patch16_224_in21k", pretrained=pretrained)
            # backbone = ViT1(model_name="google/vit-large-patch16-224-in21k", pretrained=pretrained)
            backbone = ViT1(model_name=model_name, pretrained=pretrained)
        else:  # img_size == 384
            backbone = ViT(model_name="vit_base_patch16_384", pretrained=pretrained)

    w2v_file = dataset_name + "_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device(cfg.MODEL.DEVICE)

    return AttentionNet1(backbone=backbone, backbone_type=backbone_type, ft_flag=ft_flag, img_size=img_size,
                         hid_dim=hid_dim,
                         c=c, w=w, h=h, scale=scale,
                         attritube_num=attritube_num,
                         attr_group=attr_group, w2v=w2v,
                         cls_num=cls_num, ucls_num=ucls_num, device=device, Contrastive_Learning=contrastive_learning)

def build_AttentionNet2(cfg, contrastive_learning=False):
    dataset_name = cfg.DATASETS.NAME
    att_type = cfg.DATASETS.SEMANTIC_TYPE
    info = get_attributes_info(dataset_name, att_type)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]
    scls_num = cls_num - ucls_num

    attr_group = get_attr_group(dataset_name)

    img_size = cfg.DATASETS.IMAGE_SIZE
    dataset_name = cfg.DATASETS.NAME

    hid_dim = cfg.MODEL.HID_DIM
    scale = cfg.MODEL.SCALE
    pretrained = cfg.MODEL.BACKBONE.PRETRAINED
    ft_flag = cfg.MODEL.BACKBONE.FINETUNE
    model_dir = cfg.PRETRAINED_MODELS
    backbone_type = cfg.MODEL.BACKBONE.TYPE
    print(backbone_type)
    if backbone_type == 'resnet':
        # res101 feature size
        c, w, h = 2048, img_size // 32, img_size // 32
        backbone = resnet101_features(pretrained=pretrained, model_dir=model_dir)
    elif backbone_type == 'vit':
        # vit feature size
        # c, w, h = 768, img_size // 16, img_size // 16
        # c, w, h = 1024, img_size // 16, img_size // 16
        c, w, h = 768, img_size // 16, img_size // 16  # for vit_large_patch16_224_in21k
        if img_size == 224:
            # backbone = ViT(model_name="vit_base_patch16_224", pretrained=pretrained)
            # backbone = ViT(model_name="vit_large_patch16_224_in21k", pretrained=pretrained)
            # backbone = ViT1(model_name="google/vit-large-patch16-224-in21k", pretrained=pretrained)
            backbone = ViT1(model_name="google/vit-base-patch16-224", pretrained=pretrained)
        else:  # img_size == 384
            backbone = ViT(model_name="vit_base_patch16_384", pretrained=pretrained)

    w2v_file = dataset_name + "_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device(cfg.MODEL.DEVICE)

    return AttentionNet2(backbone=backbone, backbone_type=backbone_type, ft_flag=ft_flag, img_size=img_size,
                         hid_dim=hid_dim,
                         c=c, w=w, h=h, scale=scale,
                         attritube_num=attritube_num,
                         attr_group=attr_group, w2v=w2v,
                         cls_num=cls_num, ucls_num=ucls_num, device=device, Contrastive_Learning=contrastive_learning)


def build_GEMNet(cfg):
    dataset_name = cfg.DATASETS.NAME
    info = get_attributes_info(dataset_name)
    attritube_num = info["input_dim"]
    cls_num = info["n"]
    ucls_num = info["m"]

    attr_group = get_attr_group(dataset_name)

    img_size = cfg.DATASETS.IMAGE_SIZE

    # res101 feature size
    c, w, h = 2048, img_size // 32, img_size // 32

    scale = cfg.MODEL.SCALE

    pretrained = cfg.MODEL.BACKBONE.PRETRAINED
    model_dir = cfg.PRETRAINED_MODELS

    res101 = resnet101_features(pretrained=pretrained, model_dir=model_dir)
    ft_flag = cfg.MODEL.BACKBONE.FINETUNE

    w2v_file = dataset_name + "_attribute.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)

    device = torch.device(cfg.MODEL.DEVICE)

    return GEMNet(res101=res101, ft_flag=ft_flag, img_size=img_size,
                  c=c, w=w, h=h, scale=scale,
                  attritube_num=attritube_num,
                  attr_group=attr_group, w2v=w2v,
                  cls_num=cls_num, ucls_num=ucls_num,
                  device=device)


def build_MoCo(cfg):
    return MoCo(build_AttentionNet, cfg)


def build_SimCLR(cfg):
    return my_SimCLR(build_AttentionNet, cfg)

def build_SimCLR2(cfg):
    return my_SimCLR2(build_AttentionNet, cfg)

def build_SimCLR3(cfg):
    return my_SimCLR3(build_AttentionNet, cfg)

def build_SimCLR4(cfg):
    return my_SimCLR4(build_AttentionNet, cfg)

def build_SimCLR5(cfg):
    return my_SimCLR5(build_AttentionNet, cfg)

_ZSL_META_ARCHITECTURES = {
    "BasicNet": build_BasicNet,
    "AttentionNet": build_AttentionNet,
    "GEMNet": build_GEMNet,
    "MoCo": build_MoCo,
    "SimCLR": build_SimCLR,
    "SimCLR2": build_SimCLR2,
    "SimCLR3": build_SimCLR3,
    "AttentionNet2": build_AttentionNet2,
    "SimCLR4": build_SimCLR4,
    "SimCLR5": build_SimCLR5,
}


def build_zsl_pipeline(cfg, ):
    meta_arch = _ZSL_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
