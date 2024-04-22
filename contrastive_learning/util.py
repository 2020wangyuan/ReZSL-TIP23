import torch
import random


class Sampler:
    """
    用于采样出正负类别的 类
    成员变量里的target_att是被选作目标属性的属性
    """

    def __init__(self, atts_binary, train_att_binary, neg_class_number=1):
        self.atts_binary = atts_binary.to('cuda')
        self.classed_sim, self.class_dis = compute_classes_similarity(atts_binary)
        self.classed_sim = self.classed_sim.to('cuda').squeeze(0)
        self.class_dis = self.class_dis.to('cuda').squeeze(0)
        self.target_att_prob = compute_target_att_prob(train_att_binary).to('cuda')
        self.positive_and_negative_class_Wrt_atts = get_positive_and_negative_class_Wrt_atts(atts_binary).to('cuda')
        self.neg_class_number = neg_class_number
        self.target_att = []

    def sample_target_att(self, image_att_binary):
        """
        通过概率采样出作为目标属性的属性
        """
        prob = self.target_att_prob * image_att_binary
        prob_sum = torch.sum(prob, dim=0)
        prob /= prob_sum
        index = torch.multinomial(prob, 1, replacement=True)
        return index

    def sample_positive_and_negative_class(self, target_class ):
        """
        通过概率采样出正负类别,目标类别不能成为正负类别,不能成为负类别
        返回多个负类别的下标，和正类别的下标
        """
        image_att_binary = self.atts_binary[target_class]
        target_att = self.sample_target_att(image_att_binary)
        while target_att == 145:
            target_att = self.sample_target_att(image_att_binary)
        self.target_att.append(target_att)

        sim = self.classed_sim[target_class]

        negative_prob = self.classed_sim[target_class] * (1 - self.positive_and_negative_class_Wrt_atts[target_att])

        negative_prob /= torch.sum(negative_prob)
        positive_prob = (self.class_dis[target_class]) * self.positive_and_negative_class_Wrt_atts[target_att]
        positive_prob /= torch.sum(positive_prob)

        neg_classes = torch.multinomial(negative_prob, self.neg_class_number, replacement=True)
        pos_class = torch.multinomial(positive_prob, 1, replacement=True)

        return neg_classes, pos_class

    def sample(self, image_att_binary, target_class, label_queue, label_ptr):
        """
        输入不是一个batch
        返回字典中被选作样本的下标
        """
        neg_classes, pos_class = self.sample_positive_and_negative_class(target_class)
        neg_samples = torch.nonzero(torch.eq(neg_classes, label_queue))[:, 1]
        # neg_samples = [i for i in range(0, label_ptr) if label_queue[i] in neg_classes]
        pos_sample_candidates = torch.nonzero(torch.eq(pos_class, label_queue))[:, 1]
        if len(pos_sample_candidates) != 0:
            random_indices = torch.randperm(pos_sample_candidates.size(0))[:1]
            pos_sample = pos_sample_candidates[random_indices]
        else:
            pos_sample = []
        return neg_samples, pos_sample

    def samples(self, targets_class, label_queue, label_ptr):
        """
        输入为一个batch的图像二进制属性
        返回字典中被选作样本的下标
        """
        images_att_binary = self.atts_binary[targets_class]
        self.target_att = []
        neg_samples_list = []
        pos_sample_list = []
        for i in range(0, len(images_att_binary)):
            neg_samples, pos_sample = self.sample(images_att_binary[i], targets_class[i], label_queue, label_ptr)
            neg_samples_list.append(neg_samples)
            pos_sample_list.append(pos_sample)
        return neg_samples_list, pos_sample_list


# 根据二进制属性，计算每个类别之间的相似性
def compute_classes_similarity(att_binary, measure="Manhattan"):
    """
    根据二进制属性，计算每个类别之间的相似性
    相似性确保在0到1之间，1为一模一样
    """
    if measure == "Manhattan":
        dis = torch.cdist(att_binary.unsqueeze(0), att_binary.unsqueeze(0), p=1)
        size = dis.shape[-1]
        dis += torch.diag_embed(torch.ones(size))
        sim = 1 / dis

        return sim, dis


def compute_target_att_prob(train_att_binary):
    """
    根据训练集图片二进制属性，计算每个属性出现的频率，从而计算出属性被采样为目标属性的概率。
    """
    fre = torch.sum(torch.tensor(train_att_binary), dim=0)

    w = 1 / fre
    for idx, val in enumerate(w):
        if torch.isinf(val):
            # 处理 inf 值的情况，例如替换为一个有限值或者进行其他特定的操作
            w[idx] = 0.0  # 将 inf 替换为 0.0

    w_all = torch.sum(w)
    prob = w / w_all

    return prob


def get_positive_and_negative_class_Wrt_atts(atts_binary):
    """
    获取可能成为目标类别的正负类别的集合。
    将类别二进制属性转置即可获取到集合，0代表可能成为负样本，1代表可能成为正样本。
    """
    return torch.transpose(atts_binary, 0, 1)


def pad_tensor_list_to_uniform_length(tensor_list):
    """
    将张量列表中的张量长度调整为统一长度。

    Args:
        tensor_list (list): 包含张量的列表。

    Returns:
        list: 调整后的张量列表，所有张量的长度相同。
    """
    # 找到列表中最长的张量的长度
    max_length = max(len(tensor) for tensor in tensor_list)

    # 将列表中的所有张量长度调整为最长的张量长度
    for i, tensor in enumerate(tensor_list):
        padding_length = max_length - len(tensor)
        # 使用 torch.pad() 函数进行填充
        tensor_list[i] = torch.nn.functional.pad(tensor, (0, padding_length), mode='constant', value=0)

    return torch.stack(tensor_list, dim=0)


if __name__ == "__main__":
    import torch

    # 定义一个二维张量
    tensor = torch.tensor([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])

    # 沿着行的方向对张量的每一行进行求和
    sum_of_rows = torch.sum(tensor, dim=0)

    print(sum_of_rows)
