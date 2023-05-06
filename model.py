import torch
import torch.nn as nn
import torch.nn.functional as F

# 输入图像大小 320*240


class DsConv(nn.Module):
    def __init__(self, ch_in, ch_out, size, stride, pad, bn, ac):
        super(DsConv, self).__init__()
        pad_size = (size - 1) // 2 if pad else 0
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=size,
                                    stride=stride,  padding=pad_size, groups=ch_in, bias=False)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1), bias=False)
        self.bn_sw = bn
        self.bn = nn.BatchNorm2d(ch_out)
        self.ac_sw = ac
        self.ac = nn.LeakyReLU()

    def forward(self, x):
        x = self.depth_conv(x)  # 先进行逐通道卷积
        x = self.point_conv(x)  # 再进行逐点卷积
        if self.bn_sw:
            x = self.bn(x)
        if self.ac_sw:
            x = self.ac(x)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    @staticmethod
    def channel_shuffle(x):
        batchsize, num_channels, height, width = x.data.size()  # 获得输入数据大小
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)  #
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class Net(nn.Module):
    def __init__(self, out_ch):
        super(Net, self).__init__()
        self.l1_conv = DsConv(3, 48, 5, 3, pad=True, bn=True, ac=True)
        self.stage2 = nn.Sequential(
            ShuffleV2Block(48, 96, 24, ksize=3, stride=2),
            ShuffleV2Block(96 // 2, 96, 24, ksize=3, stride=1),
            ShuffleV2Block(96 // 2, 96, 24, ksize=3, stride=1),
            ShuffleV2Block(96 // 2, 96, 24, ksize=3, stride=1),
            ShuffleV2Block(96 // 2, 96, 24, ksize=3, stride=1),
            ShuffleV2Block(96 // 2, 96, 24, ksize=3, stride=1),
            ShuffleV2Block(96 // 2, 96, 24, ksize=3, stride=1),
            ShuffleV2Block(96 // 2, 96, 24, ksize=3, stride=1),
            ShuffleV2Block(96 // 2, 96, 24, ksize=3, stride=1)
        )  # /2

        self.stage3 = nn.Sequential(
            # ShuffleV2Block(96, 96, 48, ksize=3, stride=2),
            ShuffleV2Block(96 // 2, 96, 48, ksize=3, stride=1),
            ShuffleV2Block(96 // 2, 96, 48, ksize=3, stride=1),
            ShuffleV2Block(96 // 2, 96, 48, ksize=3, stride=1),
            ShuffleV2Block(96 // 2, 96, 48, ksize=3, stride=1),
            ShuffleV2Block(96 // 2, 96, 48, ksize=3, stride=1),
            ShuffleV2Block(96 // 2, 96, 48, ksize=3, stride=1),
            ShuffleV2Block(96 // 2, 96, 48, ksize=3, stride=1),
            ShuffleV2Block(96 // 2, 96, 48, ksize=3, stride=1)
        )

        self.stage4 = nn.Sequential(
            ShuffleV2Block(96, 192, 24, ksize=3, stride=2),
            ShuffleV2Block(192 // 2, 192, 96, ksize=3, stride=1),
            ShuffleV2Block(192 // 2, 192, 96, ksize=3, stride=1),
            ShuffleV2Block(192 // 2, 192, 96, ksize=3, stride=1),
            ShuffleV2Block(192 // 2, 192, 96, ksize=3, stride=1),
            ShuffleV2Block(192 // 2, 192, 96, ksize=3, stride=1),
            ShuffleV2Block(192 // 2, 192, 96, ksize=3, stride=1)
        )

        self.upSample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upSample2 = nn.Upsample(scale_factor=1, mode='nearest')

        # self.upSample4 = nn.Upsample(scale_factor=4, mode='nearest')
        # self.upSample2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.out_conv1 = DsConv(384, 128, 3, 1, pad=True, bn=True, ac=True)
        # self.out_conv1 = DsConv(334, 128, 3, 1, pad=True, bn=True, ac=True)
        self.out_conv2 = DsConv(128, 64, 3, 1, pad=True, bn=True, ac=True)
        self.out_conv3 = DsConv(64, 36, 3, 1, pad=True, bn=True, ac=True)
        self.out_conv4 = DsConv(36, out_ch, 3, 1, pad=True, bn=False, ac=False)

    def forward(self, x):
        x = self.l1_conv(x)
        # print(x.size())
        s2 = self.stage2(x)
        # print(s2.size())
        s3 = self.stage3(s2)
        # print(s3.size())
        s4 = self.stage4(s3)
        # print(s4.size())
        f4 = self.upSample4(s4)
        # print(f4.size())
        f3 = self.upSample2(s3)
        # print(f3.size())
        fpn = torch.cat((s2, f3, f4), 1)
        # print(fpn.size())
        fpn = self.out_conv1(fpn)
        # print(fpn.size())
        fpn = self.out_conv2(fpn)
        # print(fpn.size())
        fpn = self.out_conv3(fpn)
        fpn = self.out_conv4(fpn)
        return fpn


def main():
    classes = 12
    x = torch.randn(2, 3, 600, 600)
    print(x.size())
    net = Net(classes * 3)
    net.eval()
    x = net(x)


if __name__ == '__main__':
    main()
