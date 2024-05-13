# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from .util import Sampler as my_sampler
from .util import pad_tensor_list_to_uniform_length,pad_tensor_list_to_uniform_length2


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
            layer = nn.Sequential(nn.Linear(312, 1024), nn.ReLU(),nn.Dropout(0.1),
                                  nn.Linear(1024, 2048), nn.ReLU(),nn.Dropout(0.1),
                                  nn.Linear(2048, 128), nn.ReLU(),
                                  ).to('cuda')
            for i in [0,3,6,]:
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
            pos_cat_neg_samples = [self.CLproject[tar_atts[i]](pos_cat_neg_samples[i].detach().clone()) for i in range(len(pos_cat_neg_samples))]

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
        super(my_SimCLR2, self).__init__()

        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = build_AttentionNet(cfg, contrastive_learning=contrastive_learning)


        self.CLproject = []
        for i in range(1):
            layer = nn.Sequential(nn.Linear(312, 1024), nn.ReLU(),nn.Dropout(0.1),
                                  nn.Linear(1024, 2048), nn.ReLU(),nn.Dropout(0.1),
                                  nn.Linear(2048, 128), nn.ReLU(),
                                  ).to('cuda')
            for i in [0,3,6,]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

            layer = nn.Sequential(nn.Linear(1, 1024), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(1024, 2048), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(2048, 312), nn.ReLU(),
                                  ).to('cuda')

            for i in [0,3,6,]:
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
            pos_cat_neg_samples = [self.CLproject[0](pos_cat_neg_samples[i].detach().clone()+self.CLproject[1](tar_atts[i].float())) for i in range(len(pos_cat_neg_samples))]
            query = [self.CLproject[0](v2s[i].detach().clone()+self.CLproject[1](tar_atts[i].float())) for i in range(len(tar_atts))]

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

        self.CLproject = []
        for i in range(1):
            layer = nn.Sequential(nn.Linear(312, 1024), nn.ReLU(),nn.Dropout(0.1),
                                  nn.Linear(1024, 2048), nn.ReLU(),nn.Dropout(0.1),
                                  nn.Linear(2048, 128), nn.ReLU(),
                                  ).to('cuda')
            for i in [0,3,6,]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)

            layer = nn.Sequential(nn.Linear(1, 1024), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(1024, 2048), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(2048, 312), nn.ReLU(),
                                  ).to('cuda')

            for i in [0,3,6,]:
                torch.nn.init.normal_(layer[i].weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(layer[i].bias, 0.0)  # 初始化偏置为常数，这里设为0
            self.CLproject.append(layer)


        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = build_AttentionNet(cfg, contrastive_learning=contrastive_learning)

        self.scls_num = self.encoder_q.scls_num
        self.attritube_num = self.encoder_q.attritube_num
        self.dim = self.encoder_q.feat_channel

        # create the queue
        self.register_buffer("queue", torch.randn(self.scls_num, self.attritube_num,self.dim))
        self.queue = nn.functional.normalize(self.queue, dim=2)



        self.cosine_dis = self.encoder_q.cosine_dis





    def forward(self, x, support_att, labels = None,target_img=None, masked_one_hot=None, selected_layer=0, q_labels=None,
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
            pos_cat_neg_samples = [self.CLproject[0](pos_cat_neg_samples[i].detach().clone()+self.CLproject[1](tar_atts[i].float())) for i in range(len(pos_cat_neg_samples))]
            query = [self.CLproject[0](v2s[i].detach().clone()+self.CLproject[1](tar_atts[i].float())) for i in range(len(tar_atts))]

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
