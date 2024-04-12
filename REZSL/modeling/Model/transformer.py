import torch.nn as nn
import torch
import timm
from transformers import AutoImageProcessor, ViTModel


class ViT(nn.Module):
    def __init__(self, model_name="vit_large_patch16_224_in21k", pretrained=True):
        super(ViT, self).__init__()
        # self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.vit = timm.create_model(model_name, pretrained=False)
        self.vit.load_state_dict(torch.load("/home/wangyuan/project/ReZSL/pretrained_model/VIT/base/pytorch_model.bin"))

        # Others variants of ViT can be used as well
        '''
        1 --- 'vit_small_patch16_224'
        2 --- 'vit_base_patch16_224'
        3 --- 'vit_large_patch16_224',
        4 --- 'vit_large_patch32_224'
        5 --- 'vit_deit_base_patch16_224'
        6 --- 'deit_base_distilled_patch16_224',
        7 --- 'vit_base_patch16_384'
        '''
        # model_vit = timm.list_models('*vit*')
        # print(model_vit)
        # Change the head depending of the dataset used
        self.vit.head = nn.Identity()

    def forward(self, x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)

        return x[:, 0], x[:, 1:]


# 使用transformers的ViT，可以获取到隐藏层的feature
class ViT1(nn.Module):
    def __init__(self, model_name='google/vit-large-patch16-224-in21k', pretrained=True):
        super(ViT1, self).__init__()
        if model_name == 'google/vit-large-patch16-224-in21k':
            self.vit = ViTModel.from_pretrained("/home/wangyuan/project/ReZSL/pretrained_model/VIT/large")
            # self.vit = torch.load('/home/wangyuan/project/ReZSL/pretrained_model/VIT/large/pytorch_model.bin')




    def forward(self, x):
        outputs = self.vit(x,output_hidden_states = True)
        x = outputs.last_hidden_state

        return x[:, 0], x[:, 1:],outputs.hidden_states


if __name__ == '__main__':
    # r18_features = resnet18_features(pretrained=True)
    # print(r18_features)
    #
    # r34_features = resnet34_features(pretrained=True)
    # print(r34_features)
    #
    # r50_features = resnet50_features(pretrained=True)
    # print(r50_features)

    vit_features = ViT(model_name='vit_base_patch16_224', pretrained=True)
    print(vit_features)
