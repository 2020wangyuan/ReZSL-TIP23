# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from .util import Sampler as my_sampler
from .util import pad_tensor_list_to_uniform_length


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, build_AttentionNet, cfg,
                 dim=128, K=8192, m=0.999,
                 T=0.07,
                 mlp=False, contrastive_learning=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
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
    def _dequeue_and_enqueue(self, keys,k_labels):
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

    def forward1(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def forward(self, x,support_att, target_img=None, masked_one_hot=None, selected_layer=0, q_labels=None, sampler=None):
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
        CLloss = None


        # compute query features



        if sampler is not None:



            neg_samples_list, pos_sample_list = sampler.samples(q_labels, self.label_queue,
                                                                     self.label_queue_prt)
            v2s, reconstruct_x, reconstruct_loss, q = self.encoder_q(x=x, target_img=target_img,
                                                                     support_att=support_att,
                                                                     masked_one_hot=masked_one_hot,
                                                                     selected_layer=selected_layer,
                                                                     sampled_atts=sampler.target_att)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)
            legal_samples_list_index= [i for i in range(len(pos_sample_list)) if len(neg_samples_list[i]) !=0 and len(pos_sample_list[i]) !=0]
            if len(legal_samples_list_index) != 0 :
                pos_cat_neg_samples = []
                valid_q = q[legal_samples_list_index]
                valid_q = torch.tensor(valid_q)
                for i in legal_samples_list_index:
                    sample_list = pos_sample_list[i] + neg_samples_list[i]
                    cat_samples = []
                    for j in sample_list:
                        j = int(j)
                        sample = self.queue[:,j].clone().detach()
                        cat_samples.append(sample)
                    cat_samples = torch.stack(cat_samples, dim=1)


                    pos_cat_neg_samples.append(cat_samples)




            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(x)

                _, _, _, k = self.encoder_k(x=im_k, target_img=target_img,
                                            support_att=support_att, masked_one_hot=masked_one_hot,
                                            selected_layer=selected_layer, sampled_atts=sampler.target_att)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)

                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            if valid_q is not None:
                # negative logits: NxK
                logits_all = []
                for i in range(len(valid_q)):
                    logits = torch.einsum("c,ck->k", valid_q[i], pos_cat_neg_samples[i])
                    logits_all.append(logits)
                logits_all = pad_tensor_list_to_uniform_length(logits_all)

                logits_all /= self.T







                labels = torch.zeros(logits_all.shape[0], dtype=torch.long).cuda()
                CLloss = nn.CrossEntropyLoss()(logits_all, labels)
            # dequeue and enqueue

            self._dequeue_and_enqueue(k, q_labels)
            return v2s, reconstruct_x, reconstruct_loss, CLloss

        else:
            v2s= self.encoder_q(x=x,
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
