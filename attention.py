import functools
import sys 
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, *, stride=1,
                 padding=0, dilation=1, bias=True):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数，即卷积核的个数
        :param kernel_size: 卷积核的尺寸,如果传整数则是正方形边长，tuple则是实际尺寸
        :param stride: 步长， default 1
        :param padding: 填充个数， default 0
        :param bias: 是否使用偏置， default True
        """
        super().__init__()

        # 镜像填充 + 卷积 + BatchNorm
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(padding=padding),

            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, dilation=dilation, bias=bias),

            nn.BatchNorm2d(out_channels, eps=0.001)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

## Channel Attention (CA) Layer
class ChannelAttention(nn.Module):
    def __init__(self, inplans, reduction=16):
        super(ChannelAttention, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(inplans, inplans // reduction, 1, padding=0, bias=True),
                nn.BatchNorm2d(inplans // reduction),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplans // reduction, inplans, 1, padding=0, bias=True),
                nn.BatchNorm2d(inplans),
                nn.ReLU(inplace=True)
        )

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.conv_du(avg_out)
        max_out = self.conv_du(max_out)
        out = avg_out + max_out
        channel_attention = out.sigmoid()

        return x * channel_attention

class SpacialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpacialAttention,self).__init__()

        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2

        self.conv = nn.Sequential(
            BasicConv2d(in_channels=2, out_channels=1,
                        kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # return_shape = input_shape

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)

        spacial_attention = self.conv(out)

        return x * spacial_attention  # broadcasting

class CBAM(nn.Module):
    """
    串联的注意力机制
    """

    def __init__(self, inplans, reduction=16, kernel_size=3):
        super(CBAM,self).__init__()
    
        self.channel_attention = ChannelAttention(inplans=inplans, reduction=reduction)
        self.spacial_attention = SpacialAttention(kernel_size=kernel_size)
       
        self.batch_norm = nn.BatchNorm2d(inplans)

    def forward(self, x):
        # 返回经过注意力机制处理过的feature map，shape没有变化

        residual = x

        x = self.channel_attention(x)
        x = self.spacial_attention(x)

        x = x + residual

        x = self.batch_norm(x)
        x = x.relu()

        return x


class BAM(nn.Module):
    """
    并联的注意力机制
    """

    def __init__(self, inplans, reduction=16):
        super(BAM,self).__init__()

        mid_channels = inplans // reduction
        if mid_channels == 0:
            mid_channels = 1

        # 通道注意力
        self.channel_attention = nn.Sequential(
            # 通道平均池化
            nn.AdaptiveAvgPool2d(1),

            # MLP
            nn.Conv2d(inplans, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.RReLU(),
            nn.Conv2d(mid_channels, inplans, 1, bias=False),
            nn.BatchNorm2d(inplans)
        )

        # 空间注意力
        self.spacial_attention = nn.Sequential(
            BasicConv2d(in_channels=inplans,
                        out_channels=mid_channels, kernel_size=1),

            BasicConv2d(in_channels=mid_channels, out_channels=mid_channels,
                        kernel_size=3, padding=2, dilation=2),
            BasicConv2d(in_channels=mid_channels, out_channels=mid_channels,
                        kernel_size=3, padding=2, dilation=2),

            BasicConv2d(in_channels=mid_channels, out_channels=1, kernel_size=1),

            nn.BatchNorm2d(1)
        )

        self.batch_norm = nn.BatchNorm2d(inplans)

    def forward(self, x):
        channel_attention = self.channel_attention(x)
        spacial_attention = self.spacial_attention(x)

        bam_attention = channel_attention + spacial_attention  # broadcasting
        bam_attention = bam_attention.sigmoid()

        residual = x
        x = x * bam_attention

        x = x + residual

        x = self.batch_norm(x)
        x = x.relu()

        return x


class ACM(nn.Module):
    def __init__(self,inplans,s):
        super(ACM,self).__init__()
        # channel_1
        self.conv_1 = nn.Conv2d(inplans,256,kernel_size=1,bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool_1 = nn.AdaptiveAvgPool2d((1,1))
        self.conv_2 = nn.Conv2d(256,s*s,kernel_size=1,bias=True)
        self.sigmoid = nn.Sigmoid()
        #channel_2
        self.avg_pool_2 = nn.AdaptiveAvgPool2d(s)
        self.conv_3 = nn.Conv2d(inplans,256,kernel_size=1,bias=True)

    def forward(self,x):
        B, C, H, W = x.size()
        x_in = x
        #channel_1
        fea_1  = self.relu(self.conv_1(x))   #[B,256,H,W]
        
        fea_2 = self.avg_pool_1(fea_1)       #[B,256,1,1]
       
        fea = fea_1 + fea_2                  #[B,256,H,W]
        
        fea = self.relu(self.conv_2(fea))    #[B,s*s,H,W]
        b,c,h,w = fea.size()
        
        fea = fea.view(-1,H*W)               #[B,s*s,HW]
        
        #channel_2
        x_1 = self.avg_pool_2(x)             #[B,C,s,s]
        
        x_1 = self.relu(self.conv_3(x_1))    #[B,256,s,s]
        x_1 = self.sigmoid(x_1)
        x_1 = x_1.view(256,-1)               #[B,256,s*s]      
       
        M = torch.matmul(x_1,fea)            #[B,256,1024]

        M = M.view(256,H,W)                  #[B,256, H, W]
        
        M = M + fea_1                        #[B,256, H, W]
                        
        return M

def build_acm(inplans,s):
    return ACM(inplans,s)

def CA(inplans, reduction):
    return ChannelAttention(inplans, reduction)

def SA(kernel_size):
    return SpacialAttention(kernel_size)

def CBAM(inplans, reduction, kernel_size):
    return CBAM(inplans, reduction, kernel_size)

def BAM(inplans, kernel_size):
    return BAM(inplans, kernel_size)

if __name__ == "__main__":
    model = BAM(inplans=256,kernel_size=3)
    input = torch.rand(1, 256, 28, 28)
    output = model(input)
    print(output.shape)
    

