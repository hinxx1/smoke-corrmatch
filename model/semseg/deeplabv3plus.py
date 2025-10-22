import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()
        self.is_corr = True
        #选择backbone
        if 'resnet' in cfg['backbone']:
            self.backbone = \
                resnet.__dict__[cfg['backbone']](cfg['pretrain'], multi_grid=cfg['multi_grid'],
                                                 replace_stride_with_dilation=cfg['replace_stride_with_dilation'])
        else:
            assert cfg['backbone'] == 'xception'
            self.backbone = xception(True)

        low_channels = 256 #低层特征通道数（c1）
        high_channels = 2048 #高层特征通道数（c4）

        self.head = ASPPModule(high_channels, cfg['dilations'])#ASPP模块,扩张卷积金字塔，多尺度特征融合

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))#降维 256->48
        #high_channels//8 来源于 ASPP 输出（=2048/8=256）。浅层 48 + 高层 256 拼接后经过两次 3×3 卷积 -> 256 维。此操作把高、低层信息有效融合
        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))
        #1×1 卷积把 256 维变为类别数 logits。
        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)


        if self.is_corr:
            self.corr = Corr(nclass=cfg['nclass'])
            self.proj = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=False),
                nn.Dropout2d(0.1),
            )

    def forward(self, x, need_fp=False, use_corr=False):
        dict_return = {}
        h, w = x.shape[-2:]

        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]#c1:低层特征，c4:高层特征

        if need_fp:#特征扰动
            feats_decode = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))), torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = self.classifier(feats_decode)#获得类别 logits，并用 F.interpolate 上采样到原图大小
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)#若 need_fp 为真，outs 的通道数翻倍，chunk(2) 把它切成普通输出 out 和扰动输出 out_fp
            if use_corr:
                proj_feats = self.proj(c4)
                corr_out_dict = self.corr(proj_feats, out)
                dict_return['corr_map'] = corr_out_dict['corr_map']#corr_map：二值相关性掩码
                corr_out = corr_out_dict['out']#相关性增强后的预测
                corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)
                dict_return['corr_out'] = corr_out
            dict_return['out'] = out
            dict_return['out_fp'] = out_fp

            return dict_return

        feats_decode = self._decode(c1, c4)
        out = self.classifier(feats_decode)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        if use_corr:
            proj_feats = self.proj(c4)
            corr_out_dict = self.corr(proj_feats, out)
            dict_return['corr_map'] = corr_out_dict['corr_map']
            corr_out = corr_out_dict['out']
            corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)
            dict_return['corr_out'] = corr_out
        dict_return['out'] = out
        return dict_return

    def _decode(self, c1, c4):
        c4 = self.head(c4)#ASPP模块，多尺度特征融合
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)#上采样到c1大小

        c1 = self.reduce(c1)#降维 256->48

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        return feature


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)


class Corr(nn.Module):
    def __init__(self, nclass=21):
        super(Corr, self).__init__()
        self.nclass = nclass #把输入通道 256 压到 nclass 个通道
        self.conv1 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, feature_in, out):#feature_in即深层语义特征的 256 维版本。out-主干网络已经算出的类别 logits
        dict_return = {}
        h_in, w_in = math.ceil(feature_in.shape[2] / (1)), math.ceil(feature_in.shape[3] / (1))#feature_in 的空间尺寸
        h_out, w_out = out.shape[2], out.shape[3]#原始 logits 的尺寸
        out = F.interpolate(out.detach(), (h_in, w_in), mode='bilinear', align_corners=True)#out 上采样到 feature_in 大小
        feature = F.interpolate(feature_in, (h_in, w_in), mode='bilinear', align_corners=True)#feature_in 上采样到 out 大小
        f1 = rearrange(self.conv1(feature), 'n c h w -> n c (h w)')#rearrange 把空间维度展平成一维序列
        f2 = rearrange(self.conv2(feature), 'n c h w -> n c (h w)')
        out_temp = rearrange(out, 'n c h w -> n c (h w)')
        corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1]).float())#像素两两之间的点积相似度，再按 Transformer 习惯除以 √C 做尺度归一化
        corr_map = F.softmax(corr_map, dim=-1)#对行做 softmax —— 对于“第 i 个像素”，所有“j” 像素相似度和为 1
        corr_map_sample = self.sample(corr_map.detach(), h_in, w_in)#随机采样 128 个像素，得到 corr_map_sample
        dict_return['corr_map'] = self.normalize_corr_map(corr_map_sample, h_in, w_in, h_out, w_out)#把抽样结果转成 0/1 二值掩码并恢复到 (Ho,Wo) 分辨率。
        dict_return['out'] = rearrange(torch.matmul(out_temp, corr_map), 'n c (h w) -> n c h w', h=h_in, w=w_in)#用相关性矩阵 传播信息,每个像素的预测等于其相似像素预测的加权平均。
        return dict_return

    def sample(self, corr_map, h_in, w_in):
        index = torch.randint(0, h_in * w_in - 1, [128])
        corr_map_sample = corr_map[:, index.long(), :]
        return corr_map_sample

    def normalize_corr_map(self, corr_map, h_in, w_in, h_out, w_out):
        n, m, hw = corr_map.shape
        corr_map = rearrange(corr_map, 'n m (h w) -> (n m) 1 h w', h=h_in, w=w_in)
        corr_map = F.interpolate(corr_map, (h_out, w_out), mode='bilinear', align_corners=True)

        corr_map = rearrange(corr_map, '(n m) 1 h w -> (n m) (h w)', n=n, m=m)
        range_ = torch.max(corr_map, dim=1, keepdim=True)[0] - torch.min(corr_map, dim=1, keepdim=True)[0]
        temp_map = ((- torch.min(corr_map, dim=1, keepdim=True)[0]) + corr_map) / range_
        corr_map = (temp_map > 0.5)
        norm_corr_map = rearrange(corr_map, '(n m) (h w) -> n m h w', n=n, m=m, h=h_out, w=w_out)
        return norm_corr_map


