import torch
import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, groups=group, bias=bias)


# conv_block(nn.Module) for U-net convolution block
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, drop_out=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        self.dropout = drop_out

    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = nn.Dropout2d(0.5)(x)
        return x


# # UpCat(nn.Module) for U-net UP convolution
class UpCat(nn.Module):
    def __init__(self, in_feat, out_feat, is_deconv=True):
        super(UpCat, self).__init__()

        if is_deconv:
            self.up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        outputs = self.up(down_outputs)
        offset = inputs.size()[3] - outputs.size()[3]
        if offset == 1:
            addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2]), out=None).unsqueeze(
                3).cuda()
            outputs = torch.cat([outputs, addition], dim=3)
        elif offset > 1:
            addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2], offset), out=None).cuda()
            outputs = torch.cat([outputs, addition], dim=3)
        out = torch.cat([inputs, outputs], dim=1)

        return out


# # UpCatconv(nn.Module) for up convolution
class UpCatconv(nn.Module):
    def __init__(self, in_feat, out_feat, is_deconv=True, drop_out=False):
        super(UpCatconv, self).__init__()

        if is_deconv:
            self.conv = conv_block(in_feat, out_feat, drop_out=drop_out)
            self.up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
        else:
            self.conv = conv_block(in_feat + out_feat, out_feat, drop_out=drop_out)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv
        outputs = self.up(down_outputs)
        offset = inputs.size()[3] - outputs.size()[3]
        if offset == 1:
            addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2]), out=None).unsqueeze(
                3).cuda()
            outputs = torch.cat([outputs, addition], dim=3)
        elif offset > 1:
            addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2], offset), out=None).cuda()
            outputs = torch.cat([outputs, addition], dim=3)
        out = self.conv(torch.cat([inputs, outputs], dim=1))

        return out


# # UnetGridGatingSignal3(nn.Module)
class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1, 1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, (1, 1), (0, 0)),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, (1, 1), (0, 0)),
                                       nn.ReLU(inplace=True),
                                       )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(size=scale_factor, mode='bilinear'), )

    def forward(self, input):
        return self.dsv(input)


class downscale_CSARM(nn.Module) :
  def __init__(self,in_channel,out_channel) :
    super(downscale_CSARM,self).__init__()
    self.lrelu = nn.LeakyReLU(0.1)
    self.maxpool = nn.MaxPool2d(2,stride=2)
    self.conv1 = nn.Conv2d(in_channel,out_channel//2,kernel_size=3,padding=1)
    self.bn1 = nn.BatchNorm2d(out_channel//2)
    self.conv2 = nn.Conv2d(out_channel//2,out_channel,kernel_size=3,padding=1)
    self.bn2 = nn.BatchNorm2d(out_channel)
    #if in_channel==64 :
    #  self.conv3 = nn.Conv2d(out_channel//2,out_channel,kernel_size=3,padding=0,stride=2)
    #else :
    self.conv3 = nn.Conv2d(out_channel//2,out_channel,kernel_size=3,padding=1,stride=2)  
    self.bn3 = nn.BatchNorm2d(out_channel)

  def forward(self,input_1,input_2) :
    
    out_1 = self.maxpool(input_1)
    out_1 = self.conv1(out_1)
    out_1 = self.bn1(out_1)
    out_1 = self.lrelu(out_1)
    out_r = out_1
    out_1 = self.conv2(out_1)
    out_1 = self.bn2(out_1)

    '''addition = torch.rand((input_2.size()[0], input_2.size()[1], input_2.size()[2]), out=None).unsqueeze(
                3).cuda()
    input_2 = torch.cat([input_2, addition], dim=3)

    if input_2.size()[2]==56 :
      addition = torch.rand((input_2.size()[0], input_2.size()[1], input_2.size()[3]), out=None).unsqueeze(
                  2).cuda()
      input_2 = torch.cat([input_2, addition], dim=2)'''
    out_2 = self.conv3(input_2)
    out_2 = self.bn3(out_2)
    if out_2.size()[3]==38 :
      out = out_1+out_2[:,:,:,0:-1]
    else :  
      out = out_1+out_2  
    out = self.lrelu(out)
    return out, out_r


class Upscale_CSARM(nn.Module) :
  def __init__(self,in_channel,out_channel) :
    super(Upscale_CSARM,self).__init__()
    self.in_channel = in_channel
    self.out_channel = out_channel
    self.deconv = nn.ConvTranspose2d(in_channel,out_channel,kernel_size=2,stride=2)


  def forward(self, inputs) :
    outputs = self.deconv(inputs)

    '''if self.in_channel == 512 :
      addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2]), out=None).unsqueeze(
                  3).cuda()
      outputs = torch.cat([outputs, addition], dim=3)'''
    if self.in_channel == 256 :  
      addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2]), out=None).unsqueeze(
                  3).cuda()
      outputs = torch.cat([outputs, addition], dim=3)

      '''addition = torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[3]), out=None).unsqueeze(
                  2).cuda()
      outputs = torch.cat([outputs, addition], dim=2)'''      

    return outputs




