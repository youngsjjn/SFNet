import torch
from torch import nn
import torch.nn.functional as F

import model.resnet as models

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class CFABlock(nn.Module):
    def __init__(self, output_dim, BatchNorm):
        super(CFABlock, self).__init__()
        hidden_dim = output_dim
        input_dim = output_dim

        self.F = nn.Sequential(nn.Conv2d(input_dim+hidden_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                 BatchNorm(output_dim),
                                 nn.ReLU(inplace=True),
                               nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False),
                               BatchNorm(output_dim),
                               nn.Tanh()
                               )

        self.H = nn.Sequential(nn.Conv2d(input_dim+hidden_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                 BatchNorm(output_dim),
                                 nn.ReLU(inplace=True),
                               nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1, bias=False),
                               BatchNorm(output_dim),
                               nn.Sigmoid()
                               )

        # self._init_weight()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input, hidden):
        ih_cat = torch.cat([input, hidden], 1)

        z = self.F(ih_cat)
        g = self.H(ih_cat)

        x_cur = input * z

        next_hidden = hidden + g * x_cur

        return next_hidden

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SFNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.2, classes=2, zoom_factor=8, use_aspp=True, output_stride=16,
                 criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, pretrained=True):
        super(SFNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_aspp = use_aspp
        self.criterion = criterion
        self.os = output_stride
        models.BatchNorm = BatchNorm

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.os == 16:
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.os == 8:
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        if use_aspp:
            self.aspp = ASPP('resnet', 16, BatchNorm)

        self.gate3 = CFABlock(256, BatchNorm)
        self.gate2 = CFABlock(256, BatchNorm)
        self.gate1 = CFABlock(256, BatchNorm)

        self.conv_f3 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1, bias=False),
                                     BatchNorm(256),
                                     nn.ReLU(inplace=True))
        self.conv_h3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                     BatchNorm(256),
                                     nn.ReLU(inplace=True))

        self.conv_f2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False),
                                     BatchNorm(256),
                                     nn.ReLU(inplace=True))
        self.conv_h2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                     BatchNorm(256),
                                     nn.ReLU(inplace=True))



        self.cls = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(256, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        x_size = x.size()

        x = self.layer0(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f3_aux = f3
        f4 = self.layer4(f3)
        # Atrous Spatial Pyramid Pooling (ASPP)
        if self.use_aspp:
            spp4 = self.aspp(f4)

        # CFA module
        # CFA block 1
        if self.os == 16:
            f3 = self.conv_f3(f3)
            h3 = self.gate3(f3, spp4)
        elif self.os == 8:
            f3 = self.conv_f3(f3)
            spp4_up = nn.functional.interpolate(
                spp4, size=(f3.size(2), f3.size(3)), mode='bilinear', align_corners=False)
            h3 = self.gate3(f3, spp4_up)

        # CFA block 2
        h3 = nn.functional.interpolate(
            h3, size=(f2.size(2), f2.size(3)), mode='bilinear', align_corners=False)
        h3 = self.conv_h3(h3)
        f2 = self.conv_f2(f2)
        h2 = self.gate2(f2, h3)

        # CFA block 3
        h2 = nn.functional.interpolate(
            h2, size=(f1.size(2), f1.size(3)), mode='bilinear', align_corners=False)
        h2 = self.conv_h2(h2)
        f1 = self.conv_f1(f1)
        h1 = self.gate1(f1, h2)


        # Concatenation of the ASPP output and the CFA output
        spp4_last = nn.functional.interpolate(
            spp4, size=(f1.size(2), f1.size(3)), mode='bilinear', align_corners=False)
        last_feat = torch.cat([h1, spp4_last], 1)

        # Classifcation
        x = self.cls(last_feat)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=x_size[2:], mode='bilinear', align_corners=True)

        if self.training:
            # Compute main loss and aux loss in training phase
            # Return category index at each pixel and the losses
            aux = self.aux(f3_aux)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=x_size[2:], mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            # Return classfication results(prob on categories) at each pixel in test phase
            return x