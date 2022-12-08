import torch
import torch.nn as nn

class FCN16s(nn.Module) :
  def __init__(self,in_ch=3,n_classes=2) :
    super(FCN16s,self).__init__()
    self.in_channels = in_ch
    self.n_classes = n_classes
    
    self.conv1 = nn.Sequential(
      nn.Conv2d(3,64,3,padding=100),
      nn.ReLU(inplace=True),
      nn.Conv2d(64,64,3,padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2,stride=2,ceil_mode=True)
      )

    self.conv2 = nn.Sequential(
      nn.Conv2d(64,128,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(128,128,3,padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2,stride=2,ceil_mode=True)
      )

    self.conv3 = nn.Sequential(
      nn.Conv2d(128,256,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256,256,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256,256,3,padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2,stride=2,ceil_mode=True)
      )

    self.conv4 = nn.Sequential(
      nn.Conv2d(256,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2,stride=2,ceil_mode=True)
      )

    self.conv5 = nn.Sequential(
      nn.Conv2d(512,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2,stride=2,ceil_mode=True)
      )   

    self.conv6 = nn.Sequential(
      nn.Conv2d(512,4096,7),
      nn.ReLU(inplace=True),
      nn.Dropout2d()
    )          

    self.conv7 = nn.Sequential(
      nn.Conv2d(4096,4096,1),
      nn.ReLU(inplace=True),
      nn.Dropout2d()
    )

    self.score_pool5 = nn.Sequential(
      nn.Conv2d(4096,n_classes,1)
      )

    self.score_pool4 = nn.Sequential(
      nn.Conv2d(512,n_classes,1)
      )  

    self.upscale2x = nn.Sequential(
      nn.ConvTranspose2d(n_classes,n_classes,4,stride=2,bias=False)
      )  

    self.upscale16x = nn.Sequential(
      nn.ConvTranspose2d(n_classes,n_classes,32,stride=16,bias=False)
      )    

  def forward(self,x) :
    output = self.conv1(x)
    output = self.conv2(output)
    output = self.conv3(output)
    output = self.conv4(output)

    pool4output = self.score_pool4(output)
    output = self.conv5(output)
    output = self.conv6(output)
    output = self.conv7(output)
    pool5output = self.score_pool5(output)
    pool5output = self.upscale2x(pool5output)
    fused_output = pool5output + pool4output[:, :, 5 : 5 + pool5output.size()[2], 5 : 5 + pool5output.size()[3]]

    fused_output = self.upscale16x(fused_output)
    fused_output = fused_output[:, :, 27 : 27 + x.size()[2], 27 : 27 + x.size()[3]] #crop the output to be the same size as input
    return fused_output


class FCN32s(nn.Module) :
  def __init__(self,in_ch=3,n_classes=2) :
    super(FCN32s,self).__init__()
    self.in_channels=in_ch
    self.n_classes=n_classes

    self.conv1 = nn.Sequential(
      nn.Conv2d(3,64,3,padding=100),
      nn.ReLU(inplace=True),
      nn.Conv2d(64,64,3,padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2,stride=2,ceil_mode=True)
      )

    self.conv2 = nn.Sequential(
      nn.Conv2d(64,128,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(128,128,3,padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2,stride=2,ceil_mode=True)
      )

    self.conv3 = nn.Sequential(
      nn.Conv2d(128,256,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256,256,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256,256,3,padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2,stride=2,ceil_mode=True)
      )

    self.conv4 = nn.Sequential(
      nn.Conv2d(256,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2,stride=2,ceil_mode=True)
      )

    self.conv5 = nn.Sequential(
      nn.Conv2d(512,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2,stride=2,ceil_mode=True)
      )   

    self.conv6 = nn.Sequential(
      nn.Conv2d(512,4096,7),
      nn.ReLU(inplace=True),
      nn.Dropout2d()
      )          

    self.conv7 = nn.Sequential(
      nn.Conv2d(4096,4096,1),
      nn.ReLU(inplace=True),
      nn.Dropout2d()
      )

    self.score = nn.Conv2d(4096,n_classes,1)
    self.upsample32x = nn.ConvTranspose2d(n_classes,n_classes,64,stride=32,bias=False)

  def forward(self,x) :
    output=self.conv1(x)
    output=self.conv2(output)
    output=self.conv3(output)
    output=self.conv4(output)
    output=self.conv5(output)
    output=self.conv6(output)
    output=self.conv7(output)
    output=self.score(output)
    output=self.upsample32x(output)
    output=output[:,:,19:19+x.size()[2],19:19+x.size()[3]]

    return output


class ResnetBlock(nn.Module):
    def __init__(self, indim,outdim, use_dropout,require_downsampling):
        super(ResnetBlock, self).__init__()
        self.indim=indim
        self.outdim=outdim
        self.conv_block = self.build_conv_block(indim,outdim,use_dropout,require_downsampling)
        if indim!=outdim and require_downsampling==True :
          self.resconv = nn.Sequential(nn.Conv2d(indim, outdim, kernel_size=3,stride=2),
                       nn.BatchNorm2d(outdim),
                       nn.ReLU(True))
          
        if indim!=outdim and require_downsampling==False  :
          self.resconv = nn.Sequential(nn.Conv2d(indim, outdim, kernel_size=1,stride=1),
                       nn.BatchNorm2d(outdim),
                       nn.ReLU(True))  

    def build_conv_block(self, indim, outdim, use_dropout,require_downsampling):
        conv_block = []
        if require_downsampling==True :
          conv_block += [nn.Conv2d(indim, outdim, kernel_size=3,stride=2),
                       nn.BatchNorm2d(outdim),
                       nn.ReLU(True)]
        else :
          conv_block += [nn.Conv2d(indim, outdim, kernel_size=1,stride=1),
                       nn.BatchNorm2d(outdim),
                       nn.ReLU(True)]


        conv_block += [nn.Conv2d(outdim, outdim, kernel_size=3, padding=1),
                       nn.BatchNorm2d(outdim),
                       nn.ReLU(True)]   

        conv_block += [nn.Conv2d(outdim, outdim, kernel_size=1),
                       nn.BatchNorm2d(outdim),
                       nn.ReLU(True)]                           

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        if self.indim!=self.outdim  :
          x = self.resconv(x)
        out = x + out
        return out


class FCRN(nn.Module) :
  def __init__(self,in_ch,n_classes,n_blocks) :
    super(FCRN,self).__init__()
    self.in_channels=in_ch
    self.n_classes=n_classes
    self.n_blocks=n_blocks
    ndf=64

    self.conv1 = nn.Sequential(nn.Conv2d(self.in_channels, ndf, kernel_size=7,stride=2),
                       nn.BatchNorm2d(ndf),
                       nn.ReLU(True))                           

    self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2)

    resblocks1 = [ResnetBlock(ndf,256,use_dropout=False, require_downsampling=False)]
    ndf=256
    for i in range(n_blocks//3-1) :
      resblocks1 += [ResnetBlock(ndf,ndf,use_dropout=False, require_downsampling=False)]
    self.resblocks1 = nn.Sequential(*resblocks1)  


    resblocks2 = [ResnetBlock(256,512,use_dropout=False, require_downsampling=True)]
    ndf=512
    for i in range(n_blocks//3-1) :
      resblocks2 += [ResnetBlock(ndf,ndf,use_dropout=False, require_downsampling=False)]
    self.resblocks2 = nn.Sequential(*resblocks2)

    
    resblocks3 = [ResnetBlock(ndf,2*ndf,use_dropout=False, require_downsampling=True)]
    ndf=2*ndf
    for i in range(n_blocks//3-1) :
      resblocks3 += [ResnetBlock(ndf,ndf,use_dropout=False, require_downsampling=False)]
    self.resblocks3 = nn.Sequential(*resblocks3)
    
    self.reduction = nn.Sequential(nn.Conv2d(ndf, n_classes, kernel_size=1),
                       nn.BatchNorm2d(n_classes),
                       nn.ReLU(True))

    self.upscale1 = nn.Sequential(nn.ConvTranspose2d(n_classes, n_classes, kernel_size=3,stride=2),
                       nn.BatchNorm2d(n_classes),
                       nn.ReLU(True))                  

    self.upscale2 = nn.Sequential(nn.ConvTranspose2d(n_classes, n_classes, kernel_size=3,stride=2),
                       nn.BatchNorm2d(n_classes),
                       nn.ReLU(True)) 

    self.upscale3 = nn.Sequential(nn.ConvTranspose2d(n_classes, n_classes, kernel_size=3,stride=4),
                       nn.BatchNorm2d(n_classes),
                       nn.ReLU(True))
    
    self.resize2 = nn.Sequential(nn.Conv2d(512, 2, kernel_size=1,stride=1),
                       nn.BatchNorm2d(2),
                       nn.ReLU(True))
    
    self.resize1 = nn.Sequential(nn.Conv2d(256, 2, kernel_size=1,stride=1),
                       nn.BatchNorm2d(2),
                       nn.ReLU(True))

  def forward(self,x) :
    out = self.conv1(x)
    out = self.maxpool(out)
    out_1 = self.resblocks1(out)
    out_2 = self.resblocks2(out_1)
    out = self.resblocks3(out_2)
    out = self.reduction(out)
    out = self.upscale1(out)
    out = nn.ReflectionPad2d((0, 1, 0, 0))(out)
    out = out+ self.resize2(out_2)
    out = self.upscale2(out)
    out = out + self.resize1(out_1)
    out = nn.ReflectionPad2d((1,2,1,2))(out)
    out = self.upscale3(out)
    out = out[:,:,1:-2,1:-2]

    return out

