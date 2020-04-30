'''
author:zhaoyli
date:2020/4/30
Purpose of the program:修改resnet 50 第五层5_x（对应layer 4） 第一个block 第二个卷积步长为1，这样可以保持第四层（对应layer 3）
的输出图片size不变，更好地提取特征，目前很多论文都是这样做的。

经过验证 跟https://blog.csdn.net/qq_37405118/article/details/105847809 中的 图片一样。
'''

import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck

class RESNET(nn.Module):
    def __init__(self):
        super(RESNET, self).__init__()


        resnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,

        )
        layer_5 = nn.Sequential(
            Bottleneck(1024, 512,#关键所在，步长为1
                       downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))

        self.p1 = nn.Sequential(copy.deepcopy(layer_5))

    def forward(self, x):
        x = self.backbone(x)
        x = self.p1(x)

        return x

net = RESNET()
input = torch.randn([1,3,480,160])
print(net)
print(net(input).size())