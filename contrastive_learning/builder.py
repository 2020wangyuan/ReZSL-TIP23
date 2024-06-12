# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from .util import Sampler as my_sampler
from .util import pad_tensor_list_to_uniform_length, pad_tensor_list_to_uniform_length2
import numpy as np

CUB_att = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9],  # has_bill_shape
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],  # has_wing_color
    [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],  # has_upperparts_color
    [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],  # has_underparts_color
    [55, 56, 57, 58],  # has_breast_pattern
    [59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73],  # has_back_color
    [74, 75, 76, 77, 78, 79],  # has_tail_shape
    [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94],  # has_upper_tail_color
    [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105],  # has_head_pattern
    [106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],  # has_breast_color
    [121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135],  # has_throat_color
    [136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149],  # has_eye_color
    [150, 151, 152],  # has_bill_length
    [153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167],  # has_forehead_color
    [168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182],  # has_under_tail_color
    [183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197],  # has_nape_color
    [198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212],  # has_belly_color
    [213, 214, 215, 216, 217],  # has_wing_shape
    [218, 219, 220, 221, 222],  # has_size
    [223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236],  # has_shape
    [237, 238, 239, 240],  # has_back_pattern
    [241, 242, 243, 244],  # has_tail_pattern
    [245, 246, 247, 248],  # has_belly_pattern
    [249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263],  # has_primary_color
    [264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278],  # has_leg_color
    [279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293],  # has_bill_color
    [294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308],  # has_crown_color
    [309, 310, 311, 312]  # has_wing_pattern
]
CUB_att = [[item - 1 for item in sublist] for sublist in CUB_att]

cub_group_att_name = [
    "bill_shape",
    "wing_color",
    "upperparts_color",
    "underparts_color",
    "breast_pattern",
    "back_color",
    "tail_shape",
    "upper_tail_color",
    "head_pattern",
    "breast_color",
    "throat_color",
    "eye_color",
    "bill_length",
    "forehead_color",
    "under_tail_color",
    "nape_color",
    "belly_color",
    "wing_shape",
    "size",
    "shape",
    "back_pattern",
    "tail_pattern",
    "belly_pattern",
    "primary_color",
    "leg_color",
    "bill_color",
    "crown_color",
    "wing_pattern"
]

AWA_att = [
    [1, 2, 3, 4, 5, 6, 7, 8],  # Colors: black, white, blue, brown, gray, orange, red, yellow
    [9, 10, 11],  # Patterns: patches, spots, stripes
    [12, 13, 14],  # Skin/Fur: furry, hairless, toughskin
    [15, 16],  # Size: big, small
    [17, 18],  # Body Type: bulbous, lean
    [19, 20, 21, 22, 23, 24, 25, 26],  # Limbs: flippers, hands, hooves, pads, paws, longleg, longneck, tail
    [27, 28, 29, 30],  # Teeth: chewteeth, meatteeth, buckteeth, strainteeth
    [31, 32, 33],  # Horns/Claws: horns, claws, tusks
    [34],  # Smell: smelly
    [35, 36, 37, 38, 39, 40, 41],  # Movement: flys, hops, swims, tunnels, walks, fast, slow
    [42, 43, 44],  # Strength: strong, weak, muscle
    [45, 46],  # Posture: bipedal, quadrapedal
    [47, 48, 49, 50, 51],  # Activity Level: active, inactive, nocturnal, hibernate, agility
    [52, 53, 54, 55, 56],  # Diet: fish, meat, plankton, vegetation, insects
    [57, 58, 59, 60, 61, 62],  # Foraging Behavior: forager, grazer, hunter, scavenger, skimmer, stalker
    [63, 64],  # Geographic Origin: newworld, oldworld
    [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78],
    # Habitat: arctic, coastal, desert, bush, plains, forest, fields, jungle, mountains, ocean, ground, water, tree, cave
    [79, 80],  # Temperament: fierce, timid
    [81],  # Intelligence: smart
    [82, 83],  # Social Behavior: group, solitary
    [84],  # Nesting: nestspot
    [85]  # Domesticity: domestic
]
AWA_att = [[item - 1 for item in sublist] for sublist in AWA_att]
awa_group_att_name = [
    "Colors",  # Colors: black, white, blue, brown, gray, orange, red, yellow
    "Patterns",  # Patterns: patches, spots, stripes
    "Skin/Fur",  # Skin/Fur: furry, hairless, toughskin
    "Size",  # Size: big, small
    "Body Type",  # Body Type: bulbous, lean
    "Limbs",  # Limbs: flippers, hands, hooves, pads, paws, longleg, longneck, tail
    "Teeth",  # Teeth: chewteeth, meatteeth, buckteeth, strainteeth
    "Horns/Claws",  # Horns/Claws: horns, claws, tusks
    "Smell",  # Smell: smelly
    "Movement",  # Movement: flys, hops, swims, tunnels, walks, fast, slow
    "Strength",  # Strength: strong, weak, muscle
    "Posture",  # Posture: bipedal, quadrapedal
    "Activity Level",  # Activity Level: active, inactive, nocturnal, hibernate, agility
    "Diet",  # Diet: fish, meat, plankton, vegetation, insects
    "Foraging Behavior",  # Foraging Behavior: forager, grazer, hunter, scavenger, skimmer, stalker
    "Geographic Origin",  # Geographic Origin: newworld, oldworld
    "Habitat",
    # Habitat: arctic, coastal, desert, bush, plains, forest, fields, jungle, mountains, ocean, ground, water, tree, cave
    "Temperament",  # Temperament: fierce, timid
    "Intelligence",  # Intelligence: smart
    "Social Behavior",  # Social Behavior: group, solitary
    "Nesting",  # Nesting: nestspot
    "Domesticity"  # Domesticity: domestic
]


def get_w2v_group_att(data_type):
    if data_type == "CUB":
        return None


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, build_AttentionNet, cfg,
                 dim=128, K=2048, m=0.70,
                 T=0.09,
                 mlp=False, contrastive_learning=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: contrastive_learning momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = build_AttentionNet(cfg, contrastive_learning=contrastive_learning)
        self.encoder_k = build_AttentionNet(cfg, contrastive_learning=contrastive_learning)

        self.cosine_dis = self.encoder_q.cosine_dis

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # 创建两个队列，一个包含编码后的特征，一个包含对应的类别标签
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("label_queue", torch.zeros(K))
        self.register_buffer("label_queue_prt", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
                self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    # 这段代码定义了一个方法 _dequeue_and_enqueue，它似乎是一个类的一部分，
    # 因为它具有 self 参数。它的作用似乎是将一组键（keys）添加到一个队列中，
    # 然后从队列中取出一些键，以便队列保持一定的大小。
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, k_labels):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        k_labels = concat_all_gather(k_labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        self.label_queue[ptr: ptr + batch_size] = k_labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
        self.label_queue_prt[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, x, support_att, target_img=None, masked_one_hot=None, selected_layer=0, q_labels=None,
                sampler=None):
        """
        在forward函数的基础上加入自己的代理任务
        Input:
            im_q即x: a batch of query images
            q_att_binary: im_q对应的二进制属性
            q_labels: im_q对应的标签
        Output:
            logits, targets
        """

        pos_cat_neg_samples = None
        valid_q = None
        logits_all = None
        labels = None

        # compute query features

        if sampler is not None:

            neg_samples_list, pos_sample_list = sampler.samples(q_labels, self.label_queue[:self.label_queue_prt],
                                                                self.label_queue_prt)
            v2s, reconstruct_x, reconstruct_loss, q = self.encoder_q(x=x, target_img=target_img,
                                                                     support_att=support_att,
                                                                     masked_one_hot=masked_one_hot,
                                                                     selected_layer=selected_layer,
                                                                     sampled_atts=sampler.target_att)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)
            legal_samples_list_index = [i for i in range(len(pos_sample_list)) if
                                        len(neg_samples_list[i]) != 0 and len(pos_sample_list[i]) != 0]
            if len(legal_samples_list_index) != 0:
                pos_cat_neg_samples = []
                valid_q = q[legal_samples_list_index]
                valid_q = torch.tensor(valid_q)
                for i in legal_samples_list_index:
                    sample_list = torch.cat((pos_sample_list[i], neg_samples_list[i]), dim=0)

                    cat_samples = []
                    for j in sample_list:
                        j = int(j)
                        sample = self.queue[:, j].clone().detach()
                        cat_samples.append(sample)
                    cat_samples = torch.stack(cat_samples, dim=1)

                    pos_cat_neg_samples.append(cat_samples)

                logits_all = []
                for i in range(len(legal_samples_list_index)):
                    logits = torch.einsum("c,ck->k", q[legal_samples_list_index[i]], pos_cat_neg_samples[i])
                    logits_all.append(logits)

                logits_all = pad_tensor_list_to_uniform_length(logits_all)

                logits_all /= self.T

                labels = torch.zeros(logits_all.shape[0], dtype=torch.long).cuda()

            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(x)

                _, k = self.encoder_k(x=im_k, target_img=target_img,
                                      support_att=support_att,
                                      selected_layer=selected_layer, sampled_atts=sampler.target_att)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)

                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            # dequeue and enqueue

            self._dequeue_and_enqueue(k, q_labels)
            return v2s, reconstruct_x, reconstruct_loss, logits_all, labels

        else:
            v2s = self.encoder_q(x=x,
                                 support_att=support_att,

                                 )  # queries: NxC

            return v2s


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class my_SimCLR(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, build_AttentionNet, cfg,
                 T=0.12,
                 contrastive_learning=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: contrastive_learning momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(my_SimCLR, self).__init__()

        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = build_AttentionNet(cfg, contrastive_learning=contrastive_learning)

        self.CLproject = []
        for i in range(312):
            layer = nn.Sequential(nn.Linear(312, 1024), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(1024, 2048), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(2048, 128), nn.ReLU(),
                                  ).to('cuda')
            for i in [0, 3, 6, ]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

        """
        self.CLproject = [nn.Sequential(nn.Linear(312, 128), nn.ReLU()).to('cuda') for i in range(312)]
        for layer in self.CLproject:
            torch.nn.init.normal_(layer[0].weight, mean=0.0, std=0.01)
            torch.nn.init.constant_(layer[0].bias, 0.0)  # 初始化偏置为常数，这里设为0
        """
        self.cosine_dis = self.encoder_q.cosine_dis

    def forward(self, x, support_att, target_img=None, masked_one_hot=None, selected_layer=0, q_labels=None,
                sampler=None):
        """
        在forward函数的基础上加入自己的代理任务
        Input:
            im_q即x: a batch of query images
            q_att_binary: im_q对应的二进制属性
            q_labels: im_q对应的标签
        Output:
            logits, targets
        """

        pos_cat_neg_samples = None
        logits_all = None
        labels = None

        # compute query features

        if sampler is not None:

            tar_atts, pos_sample_list, neg_samples_list = sampler.sample_from_batch(q_labels)
            sample_index_list = [i + j for i, j in zip(pos_sample_list, neg_samples_list)]
            v2s, reconstruct_x, reconstruct_loss, _ = self.encoder_q(x=x, target_img=target_img,
                                                                     support_att=support_att,
                                                                     masked_one_hot=masked_one_hot,
                                                                     selected_layer=selected_layer,
                                                                     sampled_atts=sampler.target_att)  # queries: NxC
            # q = nn.functional.normalize(q, dim=1)

            pos_cat_neg_samples = []
            for l in sample_index_list:
                cat_sample = v2s.detach().clone()[l]
                pos_cat_neg_samples.append(cat_sample)

            pos_cat_neg_samples = pad_tensor_list_to_uniform_length2(pos_cat_neg_samples)
            pos_cat_neg_samples = [self.CLproject[tar_atts[i]](pos_cat_neg_samples[i].detach().clone()) for i in
                                   range(len(pos_cat_neg_samples))]

            query = [self.CLproject[tar_atts[i]](v2s[i].detach().clone()) for i in range(len(tar_atts))]

            logits_all = []
            for i in range(len(pos_cat_neg_samples)):
                logits = torch.einsum("c,kc->k", query[i], pos_cat_neg_samples[i])
                logits_all.append(logits)

            logits_all = torch.stack(logits_all, dim=0)

            logits_all /= self.T

            labels = torch.zeros(logits_all.shape[0], dtype=torch.long).cuda()

            return v2s, reconstruct_x, reconstruct_loss, logits_all, labels

        else:
            v2s = self.encoder_q(x=x,
                                 support_att=support_att,

                                 )  # queries: NxC

            return v2s


class my_SimCLR2(nn.Module):

    def __init__(self, build_AttentionNet, cfg,
                 T=0.12,
                 contrastive_learning=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: contrastive_learning momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(my_SimCLR2, self).__init__()

        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = build_AttentionNet(cfg, contrastive_learning=contrastive_learning)
        self.scls_num = self.encoder_q.scls_num
        self.attritube_num = self.encoder_q.attritube_num
        self.dim = self.encoder_q.feat_channel
        self.CLproject = []
        for i in range(1):
            layer = nn.Sequential(nn.Linear(self.attritube_num, 1024), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(1024, 2048), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(2048, 128), nn.ReLU(),
                                  ).to('cuda')
            for i in [0, 3, 6, ]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

            layer = nn.Sequential(nn.Linear(1, 1024), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(1024, 2048), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(2048, self.attritube_num), nn.ReLU(),
                                  ).to('cuda')

            for i in [0, 3, 6, ]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

        """
        self.CLproject = [nn.Sequential(nn.Linear(312, 128), nn.ReLU()).to('cuda') for i in range(312)]
        for layer in self.CLproject:
            torch.nn.init.normal_(layer[0].weight, mean=0.0, std=0.01)
            torch.nn.init.constant_(layer[0].bias, 0.0)  # 初始化偏置为常数，这里设为0
        """
        self.cosine_dis = self.encoder_q.cosine_dis

    def forward(self, x, support_att, target_img=None, masked_one_hot=None, selected_layer=0, q_labels=None,
                sampler=None):
        """
        在forward函数的基础上加入自己的代理任务
        Input:
            im_q即x: a batch of query images
            q_att_binary: im_q对应的二进制属性
            q_labels: im_q对应的标签
        Output:
            logits, targets
        """

        pos_cat_neg_samples = None
        logits_all = None
        labels = None

        # compute query features

        if sampler is not None:

            tar_atts, pos_sample_list, neg_samples_list = sampler.sample_from_batch(q_labels)
            sample_index_list = [i + j for i, j in zip(pos_sample_list, neg_samples_list)]
            v2s, reconstruct_x, reconstruct_loss, _ = self.encoder_q(x=x, target_img=target_img,
                                                                     support_att=support_att,
                                                                     masked_one_hot=masked_one_hot,
                                                                     selected_layer=selected_layer,
                                                                     sampled_atts=sampler.target_att)  # queries: NxC
            # q = nn.functional.normalize(q, dim=1)

            pos_cat_neg_samples = []
            for l in sample_index_list:
                cat_sample = v2s.detach().clone()[l]
                pos_cat_neg_samples.append(cat_sample)

            pos_cat_neg_samples = pad_tensor_list_to_uniform_length2(pos_cat_neg_samples)
            pos_cat_neg_samples = [
                self.CLproject[0](pos_cat_neg_samples[i].detach().clone() + self.CLproject[1](tar_atts[i].float())) for
                i in range(len(pos_cat_neg_samples))]
            query = [self.CLproject[0](v2s[i].detach().clone() + self.CLproject[1](tar_atts[i].float())) for i in
                     range(len(tar_atts))]

            logits_all = []
            for i in range(len(pos_cat_neg_samples)):
                logits = torch.einsum("c,kc->k", query[i], pos_cat_neg_samples[i])
                logits_all.append(logits)

            logits_all = torch.stack(logits_all, dim=0)

            logits_all /= self.T

            labels = torch.zeros(logits_all.shape[0], dtype=torch.long).cuda()

            return v2s, reconstruct_x, reconstruct_loss, logits_all, labels

        else:
            v2s = self.encoder_q(x=x,
                                 support_att=support_att,

                                 )  # queries: NxC

            return v2s


class my_SimCLR3(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, build_AttentionNet, cfg,
                 T=0.12,
                 contrastive_learning=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: contrastive_learning momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(my_SimCLR3, self).__init__()

        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = build_AttentionNet(cfg, contrastive_learning=contrastive_learning)
        self.scls_num = self.encoder_q.scls_num
        self.attritube_num = self.encoder_q.attritube_num
        self.dim = self.encoder_q.feat_channel

        self.CLproject = []
        for i in range(1):
            layer = nn.Sequential(nn.Linear(312, 1024), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(1024, 2048), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(2048, 128), nn.ReLU(),
                                  ).to('cuda')
            for i in [0, 3, 6, ]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

            layer = nn.Sequential(nn.Linear(1, 1024), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(1024, 2048), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(2048, self.attritube_num), nn.ReLU(),
                                  ).to('cuda')

            for i in [0, 3, 6, ]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

        self.register_buffer('ema', torch.zeros(self.scls_num, self.attritube_num, self.dim))
        self.decay = 0.99

        self.cosine_dis = self.encoder_q.cosine_dis

    def forward(self, x, support_att, target_img=None, masked_one_hot=None, selected_layer=0,
                q_labels=None,
                sampler=None):
        """
        在forward函数的基础上加入自己的代理任务
        Input:
            im_q即x: a batch of query images
            q_att_binary: im_q对应的二进制属性
            q_labels: im_q对应的标签
        Output:
            logits, targets
        """

        pos_cat_neg_samples = None
        logits_all = None

        # compute query features

        if sampler is not None:

            tar_atts, pos_sample_list, neg_samples_list = sampler.sample_from_batch(q_labels)
            sample_index_list = [i + j for i, j in zip(pos_sample_list, neg_samples_list)]
            v2s, reconstruct_x, reconstruct_loss, _ = self.encoder_q(x=x, target_img=target_img,
                                                                     support_att=support_att,
                                                                     masked_one_hot=masked_one_hot,
                                                                     selected_layer=selected_layer,
                                                                     sampled_atts=sampler.target_att)  # queries: NxC

            batch_part_feature = self.encoder_q.part_feature
            self.ema_update(batch_part_feature.clone().detach(), q_labels)

            # 开始算part contrastive learning 的 logit
            cache_part_feature = self.ema[q_labels].clone().detach()
            part_CL_logits = torch.einsum('bij,bkl->bik', cache_part_feature, batch_part_feature)
            part_CL_label = torch.arange(0, self.attritube_num).to('cuda')
            part_CL_label = part_CL_label.repeat(part_CL_logits.shape[0], 1)

            pos_cat_neg_samples = []
            for l in sample_index_list:
                cat_sample = v2s.detach().clone()[l]
                pos_cat_neg_samples.append(cat_sample)

            pos_cat_neg_samples = pad_tensor_list_to_uniform_length2(pos_cat_neg_samples)
            pos_cat_neg_samples = [
                self.CLproject[0](pos_cat_neg_samples[i].detach().clone() + self.CLproject[1](tar_atts[i].float())) for
                i in range(len(pos_cat_neg_samples))]
            query = [self.CLproject[0](v2s[i].detach().clone() + self.CLproject[1](tar_atts[i].float())) for i in
                     range(len(tar_atts))]

            logits_all = []
            for i in range(len(pos_cat_neg_samples)):
                logits = torch.einsum("c,kc->k", query[i], pos_cat_neg_samples[i])
                logits_all.append(logits)

            logits_all = torch.stack(logits_all, dim=0)

            logits_all /= self.T

            labels = torch.zeros(logits_all.shape[0], dtype=torch.long).cuda()

            return v2s, reconstruct_x, reconstruct_loss, logits_all, labels, part_CL_logits, part_CL_label

        else:
            v2s = self.encoder_q(x=x,
                                 support_att=support_att,
                                 )  # queries: NxC

            return v2s

    def ema_update(self, batch_part_feature, batch_labels):

        batch_part_feature = batch_part_feature.clone().detach()
        for i in range(len(batch_labels)):
            self.ema[batch_labels[i]] = ((1.0 - self.decay) * self.ema[batch_labels[i]].clone().detach()
                                         + self.decay * batch_part_feature[i])


class my_SimCLR4(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, build_AttentionNet, cfg,
                 T=0.12, decay=0.8,
                 contrastive_learning=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: contrastive_learning momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(my_SimCLR4, self).__init__()

        self.T = T

        self.CLproject = []
        for i in range(1):
            layer = nn.Sequential(nn.LayerNorm(312), nn.Linear(312, 1024), nn.LayerNorm(1024), nn.LeakyReLU(),
                                  nn.Dropout(0.1),
                                  nn.Linear(1024, 2048), nn.LayerNorm(2048), nn.LeakyReLU(), nn.Dropout(0.1),
                                  nn.Linear(2048, 128), nn.LeakyReLU(), nn.Dropout(0.1),
                                  ).to('cuda')
            for i in [1, 5, 9, ]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

            layer = nn.Sequential(nn.Linear(1, 1024), nn.LayerNorm(1024), nn.LeakyReLU(), nn.Dropout(0.1),
                                  nn.Linear(1024, 2048), nn.LayerNorm(2048), nn.LeakyReLU(), nn.Dropout(0.1),
                                  nn.Linear(2048, 312), nn.LeakyReLU(), nn.Dropout(0.1),
                                  ).to('cuda')

            for i in [0, 4, 8, ]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = build_AttentionNet(cfg, contrastive_learning=contrastive_learning)

        self.scls_num = self.encoder_q.scls_num
        self.attritube_num = self.encoder_q.attritube_num
        self.dim = self.encoder_q.feat_channel

        self.sequence = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, 128), nn.LeakyReLU(),
                                      nn.Dropout(0.1)).to('cuda')

        params = torch.randn([self.scls_num, self.attritube_num, 128], requires_grad=False)
        params = nn.functional.normalize(params, dim=2)
        self.ema = params.clone().detach().to('cuda')
        self.decay = decay

        self.cosine_dis = self.encoder_q.cosine_dis

    def forward(self, x, support_att, labels=None, target_img=None, masked_one_hot=None, selected_layer=0,
                q_labels=None,
                sampler=None):
        """
        在forward函数的基础上加入自己的代理任务
        Input:
            im_q即x: a batch of query images
            q_att_binary: im_q对应的二进制属性
            q_labels: im_q对应的标签
        Output:
            logits, targets
        """

        pos_cat_neg_samples = None
        logits_all = None

        # compute query features

        if sampler is not None:

            tar_atts, pos_sample_list, neg_samples_list = sampler.sample_from_batch(q_labels)
            sample_index_list = [i + j for i, j in zip(pos_sample_list, neg_samples_list)]
            v2s, reconstruct_x, reconstruct_loss, _ = self.encoder_q(x=x, target_img=target_img,
                                                                     support_att=support_att,
                                                                     masked_one_hot=masked_one_hot,
                                                                     selected_layer=selected_layer,
                                                                     sampled_atts=sampler.target_att)  # queries: NxC

            batch_part_feature = self.encoder_q.part_feature
            batch_part_feature = self.sequence(batch_part_feature)
            self.ema_update(batch_part_feature.clone().detach(), labels)

            # 开始算part contrastive learning 的 logit
            cache_part_feature = self.ema[labels].clone().detach()
            part_CL_logits = torch.einsum('bij,bkl->bik', cache_part_feature, batch_part_feature)
            part_CL_label = torch.arange(0, 312).to('cuda')
            part_CL_label = part_CL_label.repeat(part_CL_logits.shape[0], 1)

            pos_cat_neg_samples = []
            for l in sample_index_list:
                cat_sample = v2s.detach().clone()[l]
                pos_cat_neg_samples.append(cat_sample)

            pos_cat_neg_samples = pad_tensor_list_to_uniform_length2(pos_cat_neg_samples)

            pos_cat_neg_samples = [
                self.CLproject[0](pos_cat_neg_samples[i].detach().clone() + self.CLproject[1](tar_atts[i].float())) for
                i in range(len(pos_cat_neg_samples))]
            query = [self.CLproject[0](v2s[i].detach().clone() + self.CLproject[1](tar_atts[i].float())) for i in
                     range(len(tar_atts))]

            logits_all = []
            for i in range(len(pos_cat_neg_samples)):
                logits = torch.einsum("c,kc->k", query[i], pos_cat_neg_samples[i])
                logits_all.append(logits)

            logits_all = torch.stack(logits_all, dim=0)

            logits_all /= self.T

            labels = torch.zeros(logits_all.shape[0], dtype=torch.long).cuda()

            return v2s, reconstruct_x, reconstruct_loss, logits_all, labels, part_CL_logits, part_CL_label

        else:
            v2s = self.encoder_q(x=x,
                                 support_att=support_att,
                                 )  # queries: NxC

            return v2s

    def ema_update(self, batch_part_feature, batch_labels):
        batch_labels = batch_labels.to('cuda')
        batch_part_feature = batch_part_feature.clone().detach()
        self.ema[batch_labels] = ((1.0 - self.decay) * self.ema[batch_labels].clone().detach()
                                  + self.decay * batch_part_feature)


class my_SimCLR5(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, build_AttentionNet, cfg,
                 T=0.12, decay=0.8,
                 contrastive_learning=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: contrastive_learning momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(my_SimCLR5, self).__init__()

        self.T = T

        self.CLproject = []
        for i in range(1):
            layer = nn.Sequential(nn.Linear(312, 128), nn.LeakyReLU(),
                                  ).to('cuda')
            for i in [0]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

            layer = nn.Sequential(nn.Linear(1, 312), nn.LeakyReLU(),
                                  ).to('cuda')

            for i in [0, ]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = build_AttentionNet(cfg, contrastive_learning=contrastive_learning)

        self.scls_num = self.encoder_q.scls_num
        self.attritube_num = self.encoder_q.attritube_num
        self.dim = self.encoder_q.feat_channel

        self.CLproject = []
        for i in range(1):
            layer = nn.Sequential(nn.Linear(self.attritube_num, 128), nn.LeakyReLU(),
                                  ).to('cuda')
            for i in [0]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

            layer = nn.Sequential(nn.Linear(1, self.attritube_num), nn.LeakyReLU(),
                                  ).to('cuda')

            for i in [0, ]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

        self.part_feature_dim = 256
        self.reduction = nn.Linear(self.dim, self.part_feature_dim)
        torch.nn.init.normal_(self.reduction.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.reduction.bias, 0.0)

        self.register_buffer('ema', torch.zeros(self.scls_num, self.attritube_num, self.part_feature_dim))
        self.decay = decay

        self.cosine_dis = self.encoder_q.cosine_dis

    def forward(self, x, support_att, labels=None, target_img=None, masked_one_hot=None, selected_layer=0,
                q_labels=None,
                sampler=None):
        """
        在forward函数的基础上加入自己的代理任务
        Input:
            im_q即x: a batch of query images
            q_att_binary: im_q对应的二进制属性
            q_labels: im_q对应的标签
        Output:
            logits, targets
        """

        pos_cat_neg_samples = None
        logits_all = None

        # compute query features

        if sampler is not None:

            tar_atts, pos_sample_list, neg_samples_list = sampler.sample_from_batch(q_labels)
            sample_index_list = [i + j for i, j in zip(pos_sample_list, neg_samples_list)]
            v2s, reconstruct_x, reconstruct_loss, _ = self.encoder_q(x=x, target_img=target_img,
                                                                     support_att=support_att,
                                                                     masked_one_hot=masked_one_hot,
                                                                     selected_layer=selected_layer,
                                                                     sampled_atts=sampler.target_att)  # queries: NxC

            # batch_part_feature = self.encoder_q.part_feature
            batch_part_feature = self.encoder_q.part_feature_another_style
            batch_part_feature = self.reduction(batch_part_feature)
            self.ema_update(batch_part_feature, labels)

            # 开始算part contrastive learning 的 logit
            cache_part_feature = self.ema[labels].clone().detach()
            part_CL_logits = torch.einsum('bij,bkl->bik', cache_part_feature, batch_part_feature)
            part_CL_label = torch.arange(0, self.attritube_num).to('cuda')
            part_CL_label = part_CL_label.repeat(part_CL_logits.shape[0], 1)

            pos_cat_neg_samples = []
            for l in sample_index_list:
                cat_sample = v2s.detach().clone()[l]
                pos_cat_neg_samples.append(cat_sample)

            pos_cat_neg_samples = pad_tensor_list_to_uniform_length2(pos_cat_neg_samples)
            pos_cat_neg_samples = [
                self.CLproject[0](pos_cat_neg_samples[i].detach().clone() + self.CLproject[1](tar_atts[i].float())) for
                i in range(len(pos_cat_neg_samples))]
            query = [self.CLproject[0](v2s[i].detach().clone() + self.CLproject[1](tar_atts[i].float())) for i in
                     range(len(tar_atts))]

            logits_all = []
            for i in range(len(pos_cat_neg_samples)):
                logits = torch.einsum("c,kc->k", query[i], pos_cat_neg_samples[i])
                logits_all.append(logits)

            logits_all = torch.stack(logits_all, dim=0)

            logits_all /= self.T

            labels = torch.zeros(logits_all.shape[0], dtype=torch.long).cuda()

            return v2s, reconstruct_x, reconstruct_loss, logits_all, labels, part_CL_logits, part_CL_label

        else:
            v2s = self.encoder_q(x=x,
                                 support_att=support_att,
                                 )  # queries: NxC

            return v2s

    def ema_update(self, batch_part_feature, batch_labels):

        batch_part_feature = batch_part_feature.clone().detach()
        for i in range(len(batch_labels)):
            self.ema[batch_labels[i]] = ((1.0 - self.decay) * self.ema[batch_labels[i]].clone().detach()
                                         + self.decay * batch_part_feature[i])


class my_SimCLR6(nn.Module):

    ###在这修改T 和 T2
    def __init__(self, build_AttentionNet, cfg,
                 T=0.12, T2=0.5,
                 contrastive_learning=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: contrastive_learning momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(my_SimCLR6, self).__init__()

        self.T = T
        self.T2 = T2

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = build_AttentionNet(cfg, contrastive_learning=contrastive_learning)
        self.scls_num = self.encoder_q.scls_num
        self.attritube_num = self.encoder_q.attritube_num
        self.dim = self.encoder_q.feat_channel

        self.CLproject = []
        for i in range(1):
            layer = nn.Sequential(nn.Linear(self.attritube_num, 1024), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(1024, 2048), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(2048, 128), nn.ReLU(),
                                  ).to('cuda')
            for i in [0, 3, 6, ]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

            layer = nn.Sequential(nn.Linear(1, 1024), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(1024, 2048), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(2048, self.attritube_num), nn.ReLU(),
                                  ).to('cuda')

            for i in [0, 3, 6, ]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

        self.cosine_dis = self.encoder_q.cosine_dis

        # 经过融合后的part feature
        self.part_feature_dim_after_fusing = 128
        self.fuse_model = fuse_to_get_part_feature(self.dim, cfg, self.part_feature_dim_after_fusing)
        self.bank = None

        if cfg.DATASETS.NAME == "CUB":
            ##################################################################################### 请在此处修改factor
            self.bank = MemoryBank(150, 28, self.part_feature_dim_after_fusing, factor=0.99)

    def forward(self, x, support_att, target_img=None, masked_one_hot=None, selected_layer=0, q_labels=None,
                sampler=None, labels=None):
        """
        在forward函数的基础上加入自己的代理任务
        Input:
            im_q即x: a batch of query images
            q_att_binary: im_q对应的二进制属性
            q_labels: im_q对应的标签
        Output:
            logits, targets
        """

        pos_cat_neg_samples = None
        logits_all = None
        labels = None

        # compute query features

        if sampler is not None:

            tar_atts, pos_sample_list, neg_samples_list = sampler.sample_from_batch(q_labels)
            sample_index_list = [i + j for i, j in zip(pos_sample_list, neg_samples_list)]
            v2s, reconstruct_x, reconstruct_loss, _ = self.encoder_q(x=x, target_img=target_img,
                                                                     support_att=support_att,
                                                                     masked_one_hot=masked_one_hot,
                                                                     selected_layer=selected_layer,
                                                                     sampled_atts=sampler.target_att)  # queries: NxC

            part_feature = self.encoder_q.part_feature_another_style
            part_feature = self.fuse_model(part_feature)
            self.bank.update(part_feature, q_labels)
            bank_feature = self.bank.get_memory_bank()[q_labels]
            part_CL_logits = torch.einsum('bij,bkl->bik', bank_feature, part_feature)
            part_CL_logits /= self.T2
            part_CL_label = torch.arange(0, 28).to('cuda')
            part_CL_label = part_CL_label.repeat(part_CL_logits.shape[0], 1)

            pos_cat_neg_samples = []
            for l in sample_index_list:
                cat_sample = v2s.detach().clone()[l]
                pos_cat_neg_samples.append(cat_sample)

            pos_cat_neg_samples = pad_tensor_list_to_uniform_length2(pos_cat_neg_samples)
            pos_cat_neg_samples = [
                self.CLproject[0](pos_cat_neg_samples[i].detach().clone() + self.CLproject[1](tar_atts[i].float())) for
                i in range(len(pos_cat_neg_samples))]
            query = [self.CLproject[0](v2s[i].detach().clone() + self.CLproject[1](tar_atts[i].float())) for i in
                     range(len(tar_atts))]

            logits_all = []
            for i in range(len(pos_cat_neg_samples)):
                logits = torch.einsum("c,kc->k", query[i], pos_cat_neg_samples[i])
                logits_all.append(logits)

            logits_all = torch.stack(logits_all, dim=0)

            logits_all /= self.T

            labels = torch.zeros(logits_all.shape[0], dtype=torch.long).cuda()

            return v2s, reconstruct_x, reconstruct_loss, logits_all, labels, part_CL_logits, part_CL_label

        else:
            v2s = self.encoder_q(x=x,
                                 support_att=support_att,

                                 )  # queries: NxC

            return v2s


class my_SimCLR7(nn.Module):

    def __init__(self, build_AttentionNet, cfg,
                 T=0.12,
                 contrastive_learning=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: contrastive_learning momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(my_SimCLR7, self).__init__()

        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = build_AttentionNet(cfg, contrastive_learning=contrastive_learning)
        self.scls_num = self.encoder_q.scls_num
        self.attritube_num = self.encoder_q.attritube_num
        self.dim = self.encoder_q.feat_channel

        self.CLproject = []
        for i in range(1):
            layer = nn.Sequential(nn.Linear(self.attritube_num, 1024), nn.ReLU(), nn.LeakyReLU(),
                                  nn.Linear(1024, 2048), nn.ReLU(), nn.LeakyReLU(),
                                  nn.Linear(2048, 128), nn.ReLU(),
                                  ).to('cuda')
            for i in [0, 3, 6, ]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

            layer = nn.Sequential(nn.Linear(1, 1024), nn.ReLU(), nn.LeakyReLU(),
                                  nn.Linear(1024, 2048), nn.ReLU(), nn.LeakyReLU(),
                                  nn.Linear(2048, self.attritube_num), nn.ReLU(),
                                  ).to('cuda')

            for i in [0, 3, 6, ]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

        self.cosine_dis = self.encoder_q.cosine_dis

        # 经过融合后的part feature
        self.part_feature_dim_after_fusing = 128
        self.fuse_model = fuse_to_get_part_feature(self.dim, cfg, self.part_feature_dim_after_fusing)
        self.bank = None

        if cfg.DATASETS.NAME == "CUB":
            ##################################################################################### 请在此处修改factor
            self.bank = MemoryBank(150, 28, self.part_feature_dim_after_fusing, factor=0.90)

    def forward(self, x, support_att, target_img=None, masked_one_hot=None, selected_layer=0, q_labels=None,
                sampler=None, labels=None):
        """
        在forward函数的基础上加入自己的代理任务
        Input:
            im_q即x: a batch of query images
            q_att_binary: im_q对应的二进制属性
            q_labels: im_q对应的标签
        Output:
            logits, targets
        """

        pos_cat_neg_samples = None
        logits_all = None
        labels = None

        # compute query features

        if sampler is not None:

            tar_atts, pos_sample_list, neg_samples_list = sampler.sample_from_batch(q_labels)
            sample_index_list = [i + j for i, j in zip(pos_sample_list, neg_samples_list)]
            v2s, reconstruct_x, reconstruct_loss, _ = self.encoder_q(x=x, target_img=target_img,
                                                                     support_att=support_att,
                                                                     masked_one_hot=masked_one_hot,
                                                                     selected_layer=selected_layer,
                                                                     sampled_atts=sampler.target_att)  # queries: NxC

            part_feature = self.encoder_q.part_feature_another_style
            part_feature = self.fuse_model(part_feature)
            self.bank.update(part_feature, q_labels)
            bank_feature = self.bank.get_memory_bank()[q_labels]
            part_CL_logits = torch.einsum('bij,bkl->bik', bank_feature, part_feature)

            part_CL_label = torch.arange(0, 28).to('cuda')
            part_CL_label = part_CL_label.repeat(part_CL_logits.shape[0], 1)

            pos_cat_neg_samples = []
            for l in sample_index_list:
                cat_sample = v2s.detach().clone()[l]
                pos_cat_neg_samples.append(cat_sample)

            pos_cat_neg_samples = pad_tensor_list_to_uniform_length2(pos_cat_neg_samples)
            pos_cat_neg_samples = [
                self.CLproject[0](pos_cat_neg_samples[i].detach().clone() + self.CLproject[1](tar_atts[i].float())) for
                i in range(len(pos_cat_neg_samples))]
            query = [self.CLproject[0](v2s[i].detach().clone() + self.CLproject[1](tar_atts[i].float())) for i in
                     range(len(tar_atts))]

            logits_all = []
            for i in range(len(pos_cat_neg_samples)):
                logits = torch.einsum("c,kc->k", query[i], pos_cat_neg_samples[i])
                logits_all.append(logits)

            logits_all = torch.stack(logits_all, dim=0)

            logits_all /= self.T

            labels = torch.zeros(logits_all.shape[0], dtype=torch.long).cuda()

            return v2s, reconstruct_x, reconstruct_loss, logits_all, labels, part_CL_logits, part_CL_label

        else:
            v2s = self.encoder_q(x=x,
                                 support_att=support_att,

                                 )  # queries: NxC

            return v2s


import torch
import torch.nn as nn
import torch.nn.functional as F


class fuse_to_get_part_feature(nn.Module):
    def __init__(self, input_dim, cfg, output_dim=128, w2v_group_att_path="/mnt/mydisk1/wangyuan/project/ReZSL/token"):
        super(fuse_to_get_part_feature, self).__init__()

        self.data_type = cfg.DATASETS.NAME
        self.group_index = None
        w2v_group_att = None
        if self.data_type == "CUB":
            self.group_index = CUB_att

            result = [[0] * 312 for _ in range(len(self.group_index))]
            for i, sublist in enumerate(CUB_att):
                for index in sublist:
                    result[i][index] = 1  # 减去1因为索引从0开始

            self.group_index = result
            self.group_index = np.array(self.group_index)
            self.group_index = torch.from_numpy(self.group_index).float()

            w2v_group_att = torch.load(w2v_group_att_path + "/cub_group_att_name_w2v.pt")


        elif self.data_type == "AWA":
            self.group_index = AWA_att

            result = [[0] * 85 for _ in range(len(self.group_index))]
            for i, sublist in enumerate(AWA_att):
                for index in sublist:
                    result[i][index] = 1

            self.group_index = result
            self.group_index = np.array(self.group_index)
            self.group_index = torch.from_numpy(self.group_index).float()

            w2v_group_att = torch.load(w2v_group_att_path + "/awa_group_att_name_w2v.pt")

        self.group_index = self.group_index.to("cuda")
        self.w2v_group_att = w2v_group_att.to("cuda")
        self.input_dim = input_dim

        # 定义线性层或其他必要的层
        self.him_feature = 2048
        self.qW = nn.Linear(w2v_group_att.shape[1], self.him_feature)
        self.kvW = nn.Linear(input_dim, self.him_feature)

        self.norm = nn.LayerNorm(self.him_feature)

        self.MLP = nn.Sequential(
            nn.Linear(self.him_feature, 1024), nn.LeakyReLU(), nn.LayerNorm(1024),
            nn.Linear(1024, 512), nn.LeakyReLU(), nn.LayerNorm(512),
            nn.Linear(512, output_dim), nn.LeakyReLU(),
        )

    def forward(self, x):
        # 使用wav_group_att计算注意力权重

        q = self.qW(self.w2v_group_att)
        k = self.kvW(x)
        v = self.kvW(x)

        q = self.norm(q)
        k = self.norm(k)
        v = self.norm(v)

        q = q.unsqueeze(0).repeat(len(k), 1, 1)

        # 计算注意力权重
        att_weights = F.softmax(torch.bmm(q, k.transpose(1, 2)) / (q.shape[1] ** 0.5), dim=2)

        # 将注意力权重应用于输入特征
        att_weights = att_weights * self.group_index
        hidden_feature = torch.bmm(att_weights, v)

        # hidden_feature += v

        part_feature = self.norm(hidden_feature)
        part_feature = self.MLP(part_feature)

        # part_feature += hidden_feature

        return part_feature


import torch

import torch


class MemoryBank:
    """
    Memory bank for part contrastive learning
    """

    def __init__(self, class_num, att_group_num, feature_dim, device='cuda', factor=0.99):
        self.class_num = class_num
        self.att_group_num = att_group_num
        self.feature_dim = feature_dim
        self.device = device
        self.memory_bank = torch.zeros(class_num, att_group_num, feature_dim, device=device)
        self.factor = factor

    def update(self, part_features, labels):
        part_features = part_features.detach()
        labels = labels.detach()

        # One-hot encode the labels
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.class_num).float()

        # Expand dimensions to match part_features
        labels_one_hot_expanded = labels_one_hot.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.att_group_num,
                                                                                  self.feature_dim)

        # Calculate the contribution of each part_feature
        update_contributions = labels_one_hot_expanded * part_features.unsqueeze(1)

        # Sum contributions across the batch
        summed_contributions = update_contributions.sum(dim=0)

        # Calculate the count of each class in the batch
        label_counts = labels_one_hot.sum(dim=0).unsqueeze(1).unsqueeze(2).expand(self.class_num, self.att_group_num,
                                                                                  self.feature_dim)

        # Avoid division by zero by using a small epsilon
        epsilon = 1e-8
        label_counts = torch.maximum(label_counts, torch.tensor(epsilon, device=self.device))

        # Update memory bank
        self.memory_bank = (1 - self.factor) * self.memory_bank + self.factor * summed_contributions / label_counts

    def get_memory_bank(self):
        return self.memory_bank
