import torch
import torch.nn as nn

from Models.layers.modules import conv_block, UpCat, UpCatconv, UnetDsv3, UnetGridGatingSignal3, downscale_CSARM, Upscale_CSARM
from Models.layers.grid_attention_layer import GridAttentionBlock2D, MultiAttentionBlock
from Models.layers.channel_attention_layer import SE_Conv_Block, CSARM_block
from Models.layers.scale_attention_layer import scale_atten_convblock
from Models.layers.nonlocal_layer import NONLocalBlock2D
"""

class CSARM_CNN(nn.Module) :
  def __init__(self, args, in_ch=3, n_classes=2) : 
    super(CSARM_CNN, self).__init__()
    self.in_channels = in_ch
    self.n_classes = n_classes    
    self.lrelu = nn.LeakyReLU(0.1)   

    self.conv1 = nn.Conv2d(self.in_channels,32,kernel_size=3, padding=1)
    self.bns = nn.BatchNorm2d(32)
    self.CSARM1 = CSARM_block(32)
    self.down1 = downscale_CSARM(3,64)
    
    self.CSARM2 = CSARM_block(64)
    self.down2 = downscale_CSARM(32,128)

    self.CSARM3 = CSARM_block(128)
    self.down3 = downscale_CSARM(64,256)

    self.CSARM4 = CSARM_block(256)
    self.conv2 = nn.Conv2d(256,512,kernel_size=3,padding=1,stride=2)
    self.bn1 =  nn.BatchNorm2d(512)

    self.CSARM5 = CSARM_block(512)

    self.up1 = Upscale_CSARM(512,256)
    self.CSARM6 = CSARM_block(256)

    self.up2 = Upscale_CSARM(256,128)
    self.CSARM7 = CSARM_block(128)

    self.up3 = Upscale_CSARM(128,64)
    self.CSARM8 = CSARM_block(64)

    self.up4 = Upscale_CSARM(64,32)
    self.CSARM9 = CSARM_block(32)
    self.classifier1 = nn.Conv2d(256,2,kernel_size=3,padding=1)
    self.classifier2 = nn.Conv2d(128,2,kernel_size=3,padding=1)
    self.classifier3 = nn.Conv2d(64,2,kernel_size=3,padding=1)
    self.classifier4 = nn.Conv2d(32,2,kernel_size=3,padding=1)
    self.softmax = nn.Softmax(dim=1)
    self.interpolate = nn.Upsample(size=(224,300),mode='bilinear')


  def forward(self,inputs) :
    out = self.conv1(inputs)
    out = self.bns(out)
    out = self.lrelu(out)
    out1 = self.CSARM1(out)

    
    out2,out_r1 = self.down1(inputs,out1)
    out2 = self.CSARM2(out2)

    out3,out_r2 = self.down2(out_r1,out2)
    out3 = self.CSARM3(out3)

    out4,out_r3 = self.down3(out_r2,out3)
    out4 = self.CSARM4(out4)

    out5 = self.conv2(out4)
    out5 = self.bn1(out5)
    out5 = self.lrelu(out5)
    out5 = self.CSARM5(out5)
    
    
    out5 = self.up1(out5)
    out5 = out5[:,:,:,0:-1]+out4
    out6 = self.CSARM6(out5)

    out7 = self.up2(out6)
    out7 = out7+out3
    out7 = self.CSARM7(out7)

    out8 = self.up3(out7)
    out8 = out8+out2
    out8 = self.CSARM8(out8)

    out9 = self.up4(out8)
    out9 = out9+out1
    out9 = self.CSARM9(out9)

    out9 = self.classifier4(out9)
    out9 = self.softmax(out9)

    out8 = self.interpolate(out8)
    out8 = self.classifier3(out8)
    out8 = self.softmax(out8)

    out7 = self.interpolate(out7)
    out7 = self.classifier2(out7)
    out7 = self.softmax(out7)

    out6 = self.interpolate(out6)
    out6 = self.classifier1(out6)
    out6 = self.softmax(out6)

    return [out6,out7,out8,out9]