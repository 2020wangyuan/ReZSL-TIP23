import torch


class sampler():
    """
    用于采样出正负类别的 类
    """

    def __init__(self, atts_binary, train_att_binary):
        self.classed_sim = compute_classes_similarity(atts_binary)
        self.target_att_prob = compute_target_att_prob(train_att_binary)
        self.positive_and_negative_class_Wrt_atts = get_positive_and_negative_class_Wrt_atts(atts_binary)

    def sample_target_att(self, image_att_binary):
        """
        通过概率采样出作为目标属性的属性
        """
        prob = self.target_att_prob * image_att_binary
        prob_sum = torch.sum(prob, dim=0)
        prob /= prob_sum
        index = torch.multinomial(prob, 1, replacement=True)
        return index

    def sample_positive_and_negative_class(self, target_att, target_class):
        """
        通过概率采样出正负类别,目标类别不能成为正负类别,不能成为负样本
        """
        negative_prob = self.classed_sim[target_class] * self.positive_and_negative_class_Wrt_atts[target_att]
        positive_prob = self.classed_sim[target_class] * (1 - self.positive_and_negative_class_Wrt_atts[target_att])

        return 1


# 根据二进制属性，计算每个类别之间的相似性
def compute_classes_similarity(att_binary, measure="Manhattan"):
    """
    根据二进制属性，计算每个类别之间的相似性
    """
    if measure == "Manhattan":
        dis = torch.cdist(att_binary.unsqueeze(0), att_binary.unsqueeze(0), p=1)
        size = dis.shape[-1]
        dis += torch.diag_embed(torch.ones(size))
        sim = 1 / dis

        return sim


def compute_target_att_prob(train_att_binary):
    """
    根据训练集图片二进制属性，计算每个属性出现的频率，从而计算出属性被采样为目标属性的概率。
    """
    fre = torch.sum(train_att_binary, dim=0) + 0.001
    w = 1 / fre
    w_all = torch.sum(w, dim=1)
    prob = w / w_all
    return prob


def get_positive_and_negative_class_Wrt_atts(atts_binary):
    """
    获取可能成为目标类别的正负类别的集合。
    将类别二进制属性转置即可获取到集合，1代表可能成为负样本，0代表可能成为正样本。
    """
    return torch.transpose(atts_binary, 0, 1)


if __name__ == "__main__":
    import torch

    # 定义一个二维张量
    tensor = torch.tensor([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])

    # 沿着行的方向对张量的每一行进行求和
    sum_of_rows = torch.sum(tensor, dim=0)

    print(sum_of_rows)
