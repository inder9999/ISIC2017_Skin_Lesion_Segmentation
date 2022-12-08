import torch
import torch.nn as nn

from Models.layers.modules import conv_block, UpCat, UpCatconv, UnetDsv3, UnetGridGatingSignal3, downscale_CSARM, Upscale_CSARM
from Models.layers.grid_attention_layer import GridAttentionBlock2D, MultiAttentionBlock
from Models.layers.channel_attention_layer import SE_Conv_Block, CSARM_block
from Models.layers.scale_attention_layer import scale_atten_convblock
from Models.layers.nonlocal_layer import NONLocalBlock2D

class Comprehensive_Atten_Unet(nn.Module):
    def __init__(self, args, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super(Comprehensive_Atten_Unet, self).__init__()
        self.args = args
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.out_size = args.out_size

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)


        # attention blocks
        # self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[1],
        #                                             inter_channels=filters[0])
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[4], inter_channels=filters[4] // 4)

        # upsampling
        self.up_concat4 = UpCat(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UpCat(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UpCat(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UpCat(filters[1], filters[0], self.is_deconv)
        self.up4 = SE_Conv_Block(filters[4], filters[3], drop_out=True)
        self.up3 = SE_Conv_Block(filters[3], filters[2])
        self.up2 = SE_Conv_Block(filters[2], filters[1])
        self.up1 = SE_Conv_Block(filters[1], filters[0])

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=4, scale_factor=self.out_size)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=4, scale_factor=self.out_size)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=4, scale_factor=self.out_size)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=4, kernel_size=1)

        self.scale_att = scale_atten_convblock(in_size=16, out_size=4)
        # final conv (without any concat)
        self.final = nn.Sequential(nn.Conv2d(4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        up4 = self.up_concat4(conv4, center)
        g_conv4 = self.nonlocal4_2(up4)

        up4, att_weight4 = self.up4(g_conv4)
        g_conv3, att3 = self.attentionblock3(conv3, up4)

        # atten3_map = att3.cpu().detach().numpy().astype(np.float)
        # atten3_map = ndimage.interpolation.zoom(atten3_map, [1.0, 1.0, 224 / atten3_map.shape[2],
        #                                                      300 / atten3_map.shape[3]], order=0)

        up3 = self.up_concat3(g_conv3, up4)
        up3, att_weight3 = self.up3(up3)
        g_conv2, att2 = self.attentionblock2(conv2, up3)

        # atten2_map = att2.cpu().detach().numpy().astype(np.float)
        # atten2_map = ndimage.interpolation.zoom(atten2_map, [1.0, 1.0, 224 / atten2_map.shape[2],
        #                                                      300 / atten2_map.shape[3]], order=0)

        up2 = self.up_concat2(g_conv2, up3)
        up2, att_weight2 = self.up2(up2)
        # g_conv1, att1 = self.attentionblock1(conv1, up2)

        # atten1_map = att1.cpu().detach().numpy().astype(np.float)
        # atten1_map = ndimage.interpolation.zoom(atten1_map, [1.0, 1.0, 224 / atten1_map.shape[2],
        #                                                      300 / atten1_map.shape[3]], order=0)
        up1 = self.up_concat1(conv1, up2)
        up1, att_weight1 = self.up1(up1)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        out = self.scale_att(dsv_cat)

        out = self.final(out)

        return out

class UpCatconv_unet(nn.Module):
    def __init__(self, in_feat, out_feat, is_deconv=True, drop_out=False):
        super(UpCatconv_unet, self).__init__()

        if is_deconv:
            #self.conv = conv_block(in_feat, out_feat, drop_out=drop_out)
            self.up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
        else:
            #self.conv = conv_block(in_feat + out_feat, out_feat, drop_out=drop_out)
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

        return torch.cat([inputs, outputs], dim=1)

class Unet(nn.Module):
    def __init__(self, args, in_ch=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super(Unet, self).__init__()
        self.args = args
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.out_size = args.out_size

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)
        self.deconv1 = UpCatconv_unet(filters[4], filters[3])

        self.conv5 = conv_block(filters[4], filters[3], drop_out=True)
        self.deconv2 = UpCatconv_unet(filters[3], filters[2])

        self.conv6 = conv_block(filters[3], filters[2])
        self.deconv3 = UpCatconv_unet(filters[2], filters[1])

        self.conv7 = conv_block(filters[2], filters[1])
        self.deconv4 = UpCatconv_unet(filters[1], filters[0])

        self.conv8 = conv_block(filters[1], filters[0])
        self.final = nn.Sequential(nn.Conv2d(filters[0], n_classes, kernel_size=1), nn.Softmax2d())



    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        deconv1 = self.deconv1(conv4,center)

        conv5 = self.conv5( deconv1)
        deconv2 = self.deconv2(conv3,conv5)

        conv6 = self.conv6( deconv2)
        deconv3 = self.deconv3(conv2,conv6)

        conv7 = self.conv7(deconv3)
        deconv4 = self.deconv4(conv1,conv7)

        conv8 = self.conv8( deconv4)
        final = self.final(conv8)

        return final




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



