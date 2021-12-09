import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from train_log.refine import *
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),        
        nn.PReLU(out_planes)
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat) + feat
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale*2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask
        
class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7, c=192)
        self.block1 = IFBlock(8+4, c=128)
        self.block2 = IFBlock(8+4, c=96)
        self.block3 = IFBlock(8+4, c=64)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward( self, x, timestep=0.5, scale_list=[8, 4, 2, 1], training=False, fastmode=True, ensemble=False):
        if training == False:
            channel = x.shape[1] // 2
            img0 = x[:, :channel]
            img1 = x[:, channel:]
        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        # else:
        #     timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        loss_cons = 0
        block = [self.block0, self.block1, self.block2, self.block3]
        for i in range(4):
            if flow is None:
                flow, mask = block[i](torch.cat((img0[:, :3], img1[:, :3], timestep), 1), None, scale=scale_list[i])
                if ensemble:
                    f1, m1 = block[i](torch.cat((img1[:, :3], img0[:, :3], 1-timestep), 1), None, scale=scale_list[i])
                    flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    mask = (mask + (-m1)) / 2
            else:
                f0, m0 = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], timestep, mask), 1), flow, scale=scale_list[i])
                if ensemble:
            	    f1, m1 = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], 1-timestep, -mask), 1), torch.cat((flow[:, 2:4], flow[:, :2]), 1), scale=scale_list[i])
            	    f0 = (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
            	    m0 = (m0 + (-m1)) / 2
                flow = flow + f0
                mask = mask + m0
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        mask_list[3] = torch.sigmoid(mask_list[3])
        
        # output_img0 = merged[3][0][0].cpu().numpy().transpose(1,2,0) * 255
        # output_img1 = merged[3][1][0].cpu().numpy().transpose(1,2,0) * 255
        # cv2.imwrite("img0.jpg",output_img0)
        # cv2.imwrite("img1.jpg",output_img1)
        # cv2.imwrite("mask.jpg",mask_list[3][0].cpu().numpy().transpose(1,2,0) * 255)
        merged[3] = merged[3][0] * mask_list[3] + merged[3][1] * (1 - mask_list[3])
        if not fastmode:
            c0 = self.contextnet(img0, flow[:, :2])
            c1 = self.contextnet(img1, flow[:, 2:4])
            tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
            res = tmp[:, :3] * 2 - 1
            merged[3] = torch.clamp(merged[3] + res, 0, 1)
        return flow_list, mask_list[3], merged
