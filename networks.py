import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class Generator(nn.Module):
    def __init__(self, opt, d=8):
        super(Generator, self).__init__()
        self.opt = opt
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)

        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d * 8)

        self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8_dep = nn.ConvTranspose2d(d * 2, 1, 4, 2, 1)

        self.deconv1_seg = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.deconv1_bn_seg = nn.BatchNorm2d(d * 8)
        self.deconv2_seg = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv2_bn_seg = nn.BatchNorm2d(d * 8)
        self.deconv3_seg = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv3_bn_seg = nn.BatchNorm2d(d * 8)
        self.deconv4_seg = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv4_bn_seg = nn.BatchNorm2d(d * 8)
        self.deconv5_seg = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        self.deconv5_bn_seg = nn.BatchNorm2d(d * 4)
        self.deconv6_seg = nn.ConvTranspose2d(d * 4 * 2, d * 4, 4, 2, 1)
        self.deconv6_bn_seg = nn.BatchNorm2d(d * 4)
        self.deconv7_seg = nn.ConvTranspose2d(d * 6, d * 4, 4, 2, 1)
        self.deconv7_bn_seg = nn.BatchNorm2d(d * 4)
        self.deconv8_seg = nn.ConvTranspose2d(d * 5, d * 4, 4, 2, 1)

        self.decoder_list_dep = []
        self.decoder_list_seg = []
        # self.conv3x3_1_dep = Conv3x3(d * 8, 1)
        # self.conv3x3_2_dep = Conv3x3(d * 8, 1)
        # self.conv3x3_3_dep = Conv3x3(d * 8, 1)
        # self.conv3x3_4_dep = Conv3x3(d * 8, 1)
        self.conv3x3_5_dep = Conv3x3(d * 4, 1)
        self.conv3x3_6_dep = Conv3x3(d * 2, 1)
        self.conv3x3_7_dep = Conv3x3(d * 1, 1)
        self.conv3x3_8_dep = Conv3x3(1, 1)

        self.conv_score = Score(d * 4)

    def weight_init(self, mean, std):
        for m in self._modules:
            self.normal_init(self._modules[m], mean, std)

    def normal_init(self, m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()

    def forward(self, input):
        input = input
        mid_feature = []
        e1 = self.conv1(input)
        mid_feature.append(e1)

        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        mid_feature.append(e2)

        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        mid_feature.append(e3)

        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        mid_feature.append(e4)

        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        mid_feature.append(e5)

        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        mid_feature.append(e6)

        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        mid_feature.append(e7)

        e8 = self.conv8(F.leaky_relu(e7, 0.2))
        # output mid_feature
        mid_feature.append(e8)

        d1_dep = self.deconv1_bn(self.deconv1(F.relu(e8)))
        d1_dep = torch.cat([d1_dep, e7], 1)

        d2_dep = self.deconv2_bn(self.deconv2(F.relu(d1_dep)))
        d2_dep = torch.cat([d2_dep, e6], 1)

        d3_dep = self.deconv3_bn(self.deconv3(F.relu(d2_dep)))
        d3_dep = torch.cat([d3_dep, e5], 1)

        d4_dep = self.deconv4_bn(self.deconv4(F.relu(d3_dep)))
        d4_dep = torch.cat([d4_dep, e4], 1)

        d5_dep = self.deconv5_bn(self.deconv5(F.relu(d4_dep)))
        d5_dep_conv = self.conv3x3_5_dep(d5_dep)
        self.decoder_list_dep.append(d5_dep_conv)
        d5_dep = torch.cat([d5_dep, e3], 1)

        d6_dep = self.deconv6_bn(self.deconv6(F.relu(d5_dep)))
        d6_dep_conv = self.conv3x3_6_dep(d6_dep)
        self.decoder_list_dep.append(d6_dep_conv)
        d6_dep = torch.cat([d6_dep, e2], 1)

        d7_dep = self.deconv7_bn(self.deconv7(F.relu(d6_dep)))
        d7_dep_conv = self.conv3x3_7_dep(d7_dep)
        self.decoder_list_dep.append(d7_dep_conv)
        d7_dep = torch.cat([d7_dep, e1], 1)

        d8_dep = self.deconv8_dep(F.relu(d7_dep))
        d8_dep_conv = self.conv3x3_8_dep(d8_dep)
        self.decoder_list_dep.append(d8_dep_conv)

        # output o_dep
        o_dep = self.decoder_list_dep
        self.decoder_list_dep = []

        d1_seg = self.deconv1_bn_seg(self.deconv1_seg(F.relu(e8)))

        d1_seg = torch.cat([d1_seg, e7], 1)
        d2_seg = self.deconv2_bn_seg(self.deconv2_seg(F.relu(d1_seg)))

        d2_seg = torch.cat([d2_seg, e6], 1)
        d3_seg = self.deconv3_bn_seg(self.deconv3_seg(F.relu(d2_seg)))

        d3_seg = torch.cat([d3_seg, e5], 1)
        d4_seg = self.deconv4_bn_seg(self.deconv4_seg(F.relu(d3_seg)))
        d4_seg = torch.cat([d4_seg, e4], 1)
        d5_seg = self.deconv5_bn_seg(self.deconv5_seg(F.relu(d4_seg)))
        # d5_seg_conv = self.conv_score(d5_seg)
        # self.decoder_list_seg.append(d5_seg_conv)
        d5_seg = torch.cat([d5_seg, e3], 1)
        d6_seg = self.deconv6_bn_seg(self.deconv6_seg(F.relu(d5_seg)))
        # d6_seg_conv = self.conv_score(d6_seg)
        # self.decoder_list_seg.append(d6_seg_conv)
        d6_seg = torch.cat([d6_seg, e2], 1)
        d7_seg = self.deconv7_bn_seg(self.deconv7_seg(F.relu(d6_seg)))
        # d7_seg_conv = self.conv_score(d7_seg)
        # self.decoder_list_seg.append(d7_seg_conv)
        d7_seg = torch.cat([d7_seg, e1], 1)
        d8_seg = self.deconv8_seg(F.relu(d7_seg))
        d8_seg_conv = self.conv_score(d8_seg)
        self.decoder_list_seg.append(d8_seg_conv)

        # output o_seg
        o_seg = self.decoder_list_seg
        self.decoder_list_seg = []

        # output last_seg_with_label
        last_seg_with_label = torch.cat((torch.max(d8_seg_conv, 1, keepdim=True)[1].float(), torch.zeros(
            [d8_seg_conv.size(0), 2, d8_seg_conv.size(2), d8_seg_conv.size(3)]).cuda()), 1)

        return o_dep, o_seg, mid_feature, last_seg_with_label


class FeatureDiscriminator(nn.Module):
    def __init__(self, output_nc=64, n_layers=2, use_bn=True):
        super(FeatureDiscriminator, self).__init__()
        nonlinearity = nn.PReLU()

        self.dim_list = (1004800, output_nc)

        self.use_bn = use_bn
        self.bn = nn.BatchNorm1d(1004800)

        first_dim = self.dim_list[1]

        model = [
            nn.Linear(self.dim_list[0], first_dim),
            nonlinearity,
        ]

        for j in range(1, n_layers):
            model += [
                nn.Linear(first_dim if j == 1 else self.dim_list[1], self.dim_list[1]),
                nonlinearity
            ]

        model += [nn.Linear(self.dim_list[1], 1)]

        self.model = (nn.Sequential(*model))

    def forward(self, input):
        input_0 = input[0]
        input_resized = input_0.view(input_0.shape[0], -1)

        for i in list(range(1, 8)):
            input_i = input[i]
            input_resized = torch.cat((input_resized, input_i.view(input_i.shape[0], -1)), dim=1)
        if self.use_bn:
            input_resized = self.bn(input_resized)
        output_i = self.model(input_resized)
        return output_i

class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True,  stride = 1, bias = True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias = bias)

    def forward(self, input):
        output = self.conv(input)
        return output

class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        return output

class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = input
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        output = self.conv(output)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=64,group=4):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = nn.GroupNorm(group, input_dim)
        self.bn2 = nn.GroupNorm(group, input_dim)
        # self.bn1 = nn.BatchNorm2d(input_dim)
        # self.bn2 = nn.BatchNorm2d(input_dim)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size=1, he_init=False)
        self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size=kernel_size, bias=False)
        self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size=kernel_size)

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output

class GoodDiscriminator(nn.Module):
    def __init__(self, dim=8):
        super(GoodDiscriminator, self).__init__()

        self.dim = dim
        self.rb1 = ResidualBlock(self.dim, 2*self.dim, 3, resample = 'down')
        self.rb2 = ResidualBlock(4*self.dim, 4*self.dim, 3, resample = 'down')
        self.rb3 = ResidualBlock(8*self.dim, 8*self.dim, 3, resample = 'down')
        self.rb4 = ResidualBlock(16*self.dim, 16*self.dim, 3, resample = 'down')
        self.rb5 = ResidualBlock(24*self.dim, 24*self.dim, 3, resample = 'down')
        self.rb6 = ResidualBlock(32*self.dim, 32*self.dim, 3, resample = 'down')
        self.rb7 = ResidualBlock(40*self.dim, 40*self.dim, 3, resample = 'down')
        self.rb8 = nn.Sequential(nn.ReLU(), MyConvo2d(48*self.dim, 48*self.dim, kernel_size=3, bias=True))

        self.model = [self.rb1,self.rb2,self.rb3,self.rb4,self.rb5,self.rb6,self.rb7]
        self.ln1 = nn.Linear(48 * self.dim * 4 * 1, 1)

    def forward(self, input):
        output = input[0]

        for i in range(7):
            output = self.model[i](output)
            output = torch.cat((output, input[i+1]), dim=1)
        output = self.rb8(output)
        output = output.view(-1, 48 * self.dim * 4 * 1)
        # todo 线性器后面要不要加非线性函数固定到0,1
        output = self.ln1(output)
        output = output.view(-1)
        return output

def init_weights(net, init_type='normal', gain=0.02):
    # initialize weights for discriminator
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight'):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.uniform_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    net = net.cuda()
    net.apply(init_func)


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        out = F.leaky_relu(out, 0.2)
        return out


class Score(nn.Module):
    # Layer to generate score
    def __init__(self, in_channels, num_class=15):
        super(Score, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(in_channels), 3)
        self.conv_final = nn.Conv2d(int(in_channels), int(num_class), 1)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        out = F.relu(out)
        out = self.conv_final(out)
        return out















