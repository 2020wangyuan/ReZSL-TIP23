from REZSL.modeling.Model.mainModel import AttentionNet1
import torch.nn as nn
import torch
from torchvision.io import read_image
from REZSL.data.transforms.data_transform import batch_random_mask
from REZSL.modeling.Model.pipeline import build_zsl_pipeline

model = nn.Module()
model.load_state_dict(torch.load('/home/wangyuan/project/ReZSL/tools/checkpoints/cub_16w_2s/AttentionNet(vit224,hid:0)_SGD(lr=5e-4)_NCE_Ablation_Global+Attentive+ReMSE_seed=214.pth'))
model.eval()

image = read_image("/home/wangyuan/project/data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg")
selected_layer = 2
batch_img, mask_one_hot = batch_random_mask(image, mask_prob=0.4)
new_height = int(224 / 2 ** int(selected_layer / 5))
new_width = int(224 / 2 ** int(selected_layer / 5))
resized_image = torch.nn.functional.interpolate(batch_img, size=(new_height, new_width), mode='bilinear',
                                                align_corners=False)

v2s,rec_x,loss = model(x=batch_img, target_img=resized_image,
                                                             support_att=none, masked_one_hot=mask_one_hot,
                                                             selected_layer=selected_layer, )
