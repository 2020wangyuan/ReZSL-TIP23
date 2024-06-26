import torch
import numpy as np

import torch.distributed as dist
from REZSL.utils.comm import *
from .inferencer import eval_zs_gzsl
from REZSL.modeling import weighted_RegressLoss, ADLoss, CPTLoss, build_zsl_pipeline, computeCoefficient, recordError, \
    get_attributes_info, get_attr_group
from REZSL.data.transforms.data_transform import batch_random_mask
import random
from REZSL.utils import set_seed


def do_train(
        model,
        ReZSL,
        tr_dataloader,
        tu_loader,
        ts_loader,
        res,
        optimizer,
        scheduler,
        lamd,
        test_gamma,
        use_REZSL,
        RegNorm,
        RegType,
        scale,
        device,
        max_epoch,
        model_file_path,
        cfg,
        cl_sampler=None
):
    seed = cfg.SEED
    set_seed(seed)

    model.to(device)
    best_performance = [-0.1, -0.1, -0.1, -0.1, -0.1]  # ZSL, S, U, H, AUSUC
    best_epoch = -1
    att_all = res['att_all'].to(device)
    att_all_var = torch.var(att_all, dim=0)
    att_all_std = torch.sqrt(att_all_var + 1e-12)
    print(att_all_std)
    att_seen = res['att_seen'].to(device)
    support_att_seen = att_seen

    print("-----use " + RegType + " -----")
    Reg_loss = weighted_RegressLoss(RegNorm, RegType, device)
    CLS_loss = torch.nn.CrossEntropyLoss()
    contrastive_loss = torch.nn.CrossEntropyLoss()

    losses = []
    cls_losses = []
    reg_losses = []

    model.train()

    for epoch in range(0, max_epoch):
        print("lr: %.8f" % (optimizer.param_groups[0]["lr"]))

        loss_epoch = []
        cls_loss_epoch = []
        reg_loss_epoch = []
        reconstruct_loss_epoch = []
        contrastive_learning_loss_epoch = []
        part_CL_loss_epoch = []

        scheduler.step()

        num_steps = len(tr_dataloader)
        model_type = cfg.MODEL.META_ARCHITECTURE

        part_CL_loss = 0
        CL_loss = 1
        logit = 2
        part_CL_logits = 3
        part_CL_labels = 4
        labels = 5
        reconstruct_loss = 6
        reconstruct_x = 7

        for iteration, (batch_img, batch_att, batch_label) in enumerate(tr_dataloader):
            part_CL_loss = None
            CL_loss = None
            logit = None
            part_CL_logits = None
            part_CL_labels = None
            labels = None
            reconstruct_loss = None
            reconstruct_x = None

            # 选择用于重构的隐藏层的输出feature
            selected_layer = random.randint(0, 11)
            new_height = int(224 / 2 ** int(selected_layer / 3))
            new_width = int(224 / 2 ** int(selected_layer / 3))
            resized_image = torch.nn.functional.interpolate(batch_img, size=(new_height, new_width), mode='bilinear',
                                                            align_corners=False)

            batch_img = batch_img.to(device)

            # mask图片和embedding
            # batch_img, mask_one_hot = batch_random_mask(batch_img, mask_prob=0.1)

            # 只mask embedding
            _, mask_one_hot = batch_random_mask(batch_img, mask_prob=0.30)
            batch_att = batch_att.to(device)
            batch_label = batch_label.to(device)

            if iteration % 50 == 0:
                index = torch.argmax(ReZSL.running_weights_Matrix)
                att_dim = batch_att.shape[1]
                d1 = index // att_dim
                d2 = index % att_dim
                print('index: (%d, %d), max weight: %.4f, corresponding offset: %.4f, max offset: %.4f' % (
                    d1, d2, ReZSL.running_weights_Matrix[d1][d2], ReZSL.running_offset_Matrix[d1][d2],
                    torch.max(ReZSL.running_offset_Matrix)))

            if model_type == "BasicNet" or model_type == "AttentionNet" or "MoCo":
                # v2s = model(x=batch_img, support_att=support_att_seen)
                if model_type == 'AttentionNet' or model_type == "AttentionNet2" or model_type == "AttentionNet3":
                    v2s, reconstruct_x, reconstruct_loss = model(x=batch_img, target_img=resized_image,
                                                                 support_att=support_att_seen,
                                                                 masked_one_hot=mask_one_hot,
                                                                 selected_layer=selected_layer)
                elif model_type == 'SimCLR3':
                    v2s, reconstruct_x, reconstruct_loss, logit, labels, part_CL_logits, part_CL_labels = model(
                        x=batch_img, target_img=resized_image,
                        support_att=support_att_seen,
                        masked_one_hot=mask_one_hot,
                        selected_layer=selected_layer,
                        sampler=cl_sampler,
                        q_labels=batch_label)
                elif (model_type == 'SimCLR3' or model_type == "SimCLR4" or model_type == "SimCLR5"
                      or model_type == "SimCLR6" or model_type == "SimCLR7"):
                    v2s, reconstruct_x, reconstruct_loss, logit, labels, part_CL_logits, part_CL_labels = model(
                        x=batch_img, target_img=resized_image,
                        labels=batch_label,
                        support_att=support_att_seen,
                        masked_one_hot=mask_one_hot,
                        selected_layer=selected_layer,
                        sampler=cl_sampler,
                        q_labels=batch_label)

                else:
                    v2s, reconstruct_x, reconstruct_loss, logit, labels = model(x=batch_img, target_img=resized_image,
                                                                                support_att=support_att_seen,
                                                                                masked_one_hot=mask_one_hot,
                                                                                selected_layer=selected_layer,
                                                                                sampler=cl_sampler,
                                                                                q_labels=batch_label)

                if use_REZSL:
                    n = v2s.shape[0]
                    ReZSL.updateWeightsMatrix(v2s.detach(), batch_att.detach(), batch_label.detach())
                    weights = ReZSL.getWeights(n, att_dim,
                                               batch_label.detach()).detach()  # weights matrix does not need gradients
                else:
                    weights = None

                if model.module == None:
                    score, cos_dist = model.cosine_dis(pred_att=v2s, support_att=support_att_seen)
                else:
                    if model_type == "MoCo":
                        score, cos_dist = model.module.encoder_q.cosine_dis(pred_att=v2s, support_att=support_att_seen)
                        pass
                    else:
                        score, cos_dist = model.module.cosine_dis(pred_att=v2s, support_att=support_att_seen)

                Lreg = Reg_loss(v2s, batch_att, weights)
                Lcls = CLS_loss(score, batch_label)
                if model_type != 'AttentionNet' and logit is not None:
                    CL_loss = contrastive_loss(logit, labels)

                if model_type != 'AttentionNet' and part_CL_logits is not None:
                    part_CL_loss = contrastive_loss(part_CL_logits, part_CL_labels)

                loss = lamd[0] * Lcls + lamd[1] * Lreg + 1 * reconstruct_loss

                if CL_loss is not None:
                    loss += CL_loss * 0.2

                if part_CL_loss is not None:
                    loss += part_CL_loss * 0.1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if CL_loss is None:

                    log_info = 'epoch: %d, it: %d/%d  |  loss: %.4f, cls_loss: %.4f, reg_loss: %.4f, reconstruct_loss: %.4f, lr: %.10f' % \
                               (epoch + 1, iteration, num_steps, loss, Lcls, Lreg, reconstruct_loss,
                                optimizer.param_groups[0]["lr"])
                elif CL_loss is not None and part_CL_loss is not None:
                    log_info = 'epoch: %d, it: %d/%d  |  loss: %.4f, cls_loss: %.4f, reg_loss: %.4f, reconstruct_loss: %.4f, CL_loss: %.4f,part_CL_loss: %.4f, lr: %.10f' % \
                               (epoch + 1, iteration, num_steps, loss, Lcls, Lreg, reconstruct_loss, CL_loss,
                                part_CL_loss,
                                optimizer.param_groups[0]["lr"])
                else:
                    log_info = 'epoch: %d, it: %d/%d  |  loss: %.4f, cls_loss: %.4f, reg_loss: %.4f, reconstruct_loss: %.4f, CL_loss: %.4f, lr: %.10f' % \
                               (epoch + 1, iteration, num_steps, loss, Lcls, Lreg, reconstruct_loss, CL_loss,
                                optimizer.param_groups[0]["lr"])
                print(log_info)

            if model_type == "GEMNet":
                v2s, atten_v2s, atten_map, query = model(x=batch_img, support_att=support_att_seen)

                if use_REZSL:
                    n = v2s.shape[0]
                    ReZSL.updateWeightsMatrix(v2s.detach(), batch_att.detach(),
                                              batch_label.detach())  # or updateWeightsMatrix_inBatch
                    weights = ReZSL.getWeights(n, att_dim,
                                               batch_label.detach()).detach()  # weights matrix does not need gradients
                else:
                    weights = None

                if model.module == None:
                    score, cos_dist = model.cosine_dis(pred_att=v2s, support_att=support_att_seen)
                else:
                    score, cos_dist = model.module.cosine_dis(pred_att=v2s, support_att=support_att_seen)
                Lreg = Reg_loss(v2s, batch_att, weights)
                Lcls = CLS_loss(score, batch_label)

                attr_group = get_attr_group(cfg.DATASETS.NAME)

                Lad = ADLoss(query, attr_group)

                Lcpt = CPTLoss(atten_map, device)

                loss = lamd[0] * Lcls + lamd[1] * Lreg + lamd[2] * Lad + lamd[3] * Lcpt

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                log_info = 'epoch: %d, it: %d/%d  |  loss: %.4f, cls_loss: %.4f, reg_loss: %.4f, ad_loss: %.4f, cpt_loss: %.4f  lr: %.10f' % \
                           (epoch + 1, iteration, num_steps, loss, Lcls, Lreg, Lad, Lcpt,
                            optimizer.param_groups[0]["lr"])
                print(log_info)

            loss_epoch.append(loss.item())
            cls_loss_epoch.append(Lcls.item())
            reg_loss_epoch.append(Lreg.item())
            reconstruct_loss_epoch.append(reconstruct_loss.item())
            if CL_loss is not None:
                contrastive_learning_loss_epoch.append(CL_loss.item())
            if part_CL_loss is not None:
                part_CL_loss_epoch.append(part_CL_loss.item())

        if is_main_process():
            losses += loss_epoch
            cls_losses += cls_loss_epoch
            reg_losses += reg_loss_epoch

            loss_epoch_mean = sum(loss_epoch) / len(loss_epoch)
            cls_loss_epoch_mean = sum(cls_loss_epoch) / len(cls_loss_epoch)
            reg_loss_epoch_mean = sum(reg_loss_epoch) / len(reg_loss_epoch)
            reconstruct_loss_epoch_mean = sum(reconstruct_loss_epoch) / len(reconstruct_loss_epoch)

            if len(contrastive_learning_loss_epoch) == 0:
                contrastive_learning_loss_epoch_mean = 0
            else:
                contrastive_learning_loss_epoch_mean = sum(contrastive_learning_loss_epoch) / len(
                    contrastive_learning_loss_epoch)

            if len(part_CL_loss_epoch) == 0:
                part_CL_loss_epoch = 0;
            else:
                part_CL_loss_epoch = sum(part_CL_loss_epoch) / len(part_CL_loss_epoch)

            log_info = 'epoch: %d |  loss: %.4f, cls_loss: %.4f, reg_loss: %.4f, reconstruct_loss_epoch: %.4f, contrastive_learning_loss_epoch: %.4f,part_CL_loss_epoch: %.4f, lr: %.10f' % \
                       (epoch + 1, loss_epoch_mean, cls_loss_epoch_mean, reg_loss_epoch_mean,
                        reconstruct_loss_epoch_mean, contrastive_learning_loss_epoch_mean, part_CL_loss_epoch,
                        optimizer.param_groups[0]["lr"])
            print(log_info)
        mask = torch.gt(ReZSL.mean_value, 0.0)
        mean = torch.mean(torch.masked_select(ReZSL.mean_value, mask))
        std = torch.std(torch.masked_select(ReZSL.mean_value, mask))
        print('Train_mean_offset mean: ' + str(mean.item()) + '. std: ' + str(std.item()) + '.')

        synchronize()
        print('Current running_weights_Matrix: ')
        print(ReZSL.running_weights_Matrix)
        acc_seen, acc_novel, H, acc_zs, AUSUC, best_gamma = eval_zs_gzsl(
            tu_loader,
            ts_loader,
            res,
            model,
            test_gamma,
            ReZSL,
            device)

        synchronize()

        if is_main_process():
            print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f, AUSUC=%.4f, best_gamma=%.4f' % (
                acc_zs, acc_seen, acc_novel, H, AUSUC, best_gamma))

            if acc_zs > best_performance[0]:
                best_performance[0] = acc_zs

            if H > best_performance[3]:
                best_epoch = epoch + 1
                best_performance[1:4] = [acc_seen, acc_novel, H]
                data = {}
                data["model"] = model.state_dict()
                torch.save(data, model_file_path)
                print('save best model: ' + model_file_path)

            if AUSUC > best_performance[4]:
                best_performance[4] = AUSUC
                model_file_path_AUSUC = model_file_path.split('.pth')[0] + '_AUSUC' + '.pth'
                torch.save(data, model_file_path_AUSUC)
                print('save best AUSUC model: ' + model_file_path_AUSUC)
            print("best: ep: %d" % best_epoch)
            print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f, AUSUC=%.4f' % tuple(best_performance))

    if is_main_process():
        print("best: ep: %d" % best_epoch)
        print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f, AUSUC=%.4f' % tuple(best_performance))
