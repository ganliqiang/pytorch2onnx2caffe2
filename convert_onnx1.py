#-*-coding:utf-8-*-
import collections
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import torch.onnx.symbolic
import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3mb4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        sz=out.size()
        #out=out.view(sz[0],sz[1],-1)        
        out = self.bn1(out)
        #out=out.view(sz[0],sz[1])
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out=out.view(sz[0],sz[1],-1)        
        #out = self.bn2(out)
        #out=out.view(sz[0],sz[1])

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=7, scale=1):
        cnDivider = 4
        baseChannel = 8   #64
        self.inplanes = 8
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 8, layers[0])
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Top layer
        self.toplayer = nn.Conv2d(64, 8, kernel_size=1, stride=1, padding=0)  # Reduce channels   2048
        self.toplayer_bn = nn.BatchNorm2d(8)
        self.toplayer_relu = nn.ReLU(inplace=True)

        # Smooth layers
        self.smooth1 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.smooth1_bn = nn.BatchNorm2d(8)
        self.smooth1_relu = nn.ReLU(inplace=True)

        self.smooth2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.smooth2_bn = nn.BatchNorm2d(8)
        self.smooth2_relu = nn.ReLU(inplace=True)

        self.smooth3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.smooth3_bn = nn.BatchNorm2d(8)
        self.smooth3_relu = nn.ReLU(inplace=True)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.latlayer1_bn = nn.BatchNorm2d(8)
        self.latlayer1_relu = nn.ReLU(inplace=True)

        self.latlayer2 = nn.Conv2d(16,  8, kernel_size=1, stride=1, padding=0)
        self.latlayer2_bn = nn.BatchNorm2d(8)
        self.latlayer2_relu = nn.ReLU(inplace=True)

        self.latlayer3 = nn.Conv2d(8,  8, kernel_size=1, stride=1, padding=0)
        self.latlayer3_bn = nn.BatchNorm2d(8)
        self.latlayer3_relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(8, 3, kernel_size=1, stride=1, padding=0)

        self.scale = scale
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        #return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')
        return F.interpolate(x, size=[ H // scale, W // scale], mode='bilinear',align_corners=False)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        #return F.upsample(x, size=(H, W), mode='bilinear') + y
        return F.interpolate(x, size=[H, W], mode='bilinear',align_corners=False) + y

    def forward(self, x):
        h = x  #(4,3,640,640)
        h = self.conv1(h)#(4,64,320,320)  #1/2
        h = self.bn1(h)
        h = self.relu1(h)

        h = self.maxpool(h)  #1/4

        h = self.layer1(h)#(4,64,160,160)
        c2 = h #(4,256,160,160)
        h = self.layer2(h)#1/8
        c3 = h #(4,512,80,80)
        h = self.layer3(h)#1/16
        c4 = h#(4,1024,40,40)
        h = self.layer4(h)#1/32
        c5 = h#(4,2048,20,20)

        # Top-down
        p5 = self.toplayer(c5)#(4,64,20,20)
        p5 = self.toplayer_relu(self.toplayer_bn(p5))
        n=4
        c4 = self.latlayer1(c4)#(4,64,40,40)
        c4 = self.latlayer1_relu(self.latlayer1_bn(c4))
        p5=F.interpolate(p5, size=(40*n, 40*n), mode='bilinear',align_corners=False)
        p4=p5+c4
        #p4 = self._upsample_add(p5, c4)
        p4 = self.smooth1(p4)#(4,64,40,40)
        p4 = self.smooth1_relu(self.smooth1_bn(p4))

        c3 = self.latlayer2(c3)#(4,64,80,80)
        c3 = self.latlayer2_relu(self.latlayer2_bn(c3))
        p4=F.interpolate(p4, size=(80*n, 80*n), mode='bilinear',align_corners=False)
        p3=p4+c3
        #p3 = self._upsample_add(p4, c3)
        p3 = self.smooth2(p3)
        p3 = self.smooth2_relu(self.smooth2_bn(p3))        

        c2 = self.latlayer3(c2)
        c2 = self.latlayer3_relu(self.latlayer3_bn(c2))
        p3=F.interpolate(p3, size=(160*n, 160*n), mode='bilinear',align_corners=False)
        p2=p3+c2
        #p2 = self._upsample_add(p3, c2)
        p2 = self.smooth3(p2)#(4,64,160,160)
        p2 = self.smooth3_relu(self.smooth3_bn(p2))

        #p3 = self._upsample(p3, p2)
        #p4 = self._upsample(p4, p2)
        #p5 = self._upsample(p5, p2)
        p3=F.interpolate(p3, size=(160*n, 160*n), mode='bilinear',align_corners=False)
        p4=F.interpolate(p4, size=( 160*n, 160*n), mode='bilinear',align_corners=False)
        p5=F.interpolate(p5, size=( 160*n, 160*n), mode='bilinear',align_corners=False)
        out = torch.cat((p2, p3, p4, p5), 1)
        out = self.conv2(out)
        out = self.relu2(self.bn2(out))
        out = self.conv3(out)
        #out = self._upsample(out, x, scale=self.scale)
        out=F.interpolate(out, size=(640*n/2, 640*n/2), mode='bilinear',align_corners=False)

        return out


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    #print model
    return model


torch_model = resnet18(pretrained=False, num_classes=3)

#print torch_model
checkpoint = torch.load("/home/user/psenet/PSENet/onnx/psenet/checkpoint.pth.tar")
print checkpoint['lr']
            
#torch_model.load_state_dict(checkpoint['state_dict'])
d = collections.OrderedDict()
for key, value in checkpoint['state_dict'].items():
    tmp = key[7:]
    d[tmp] = value
torch_model.load_state_dict(d)

torch_model.eval()

torch_model.train(False)


x = torch.randn(1, 3, 2560 ,2560)
input_names=["data"]

# 导出模型
torch.onnx.export(torch_model,x,"psenet.onnx",input_names=input_names)

import onnx
import caffe2.python.onnx.backend as backend
import numpy as np

# Load the ONNX ModelProto object. model is a standard Python protobuf object
model = onnx.load("/home/user/psenet/PSENet/onnx/psenet/psenet.onnx")

# prepare the caffe2 backend for executing the model this converts the ONNX model into a
# Caffe2 NetDef that can execute it. Other ONNX backends, like one for CNTK will be
# availiable soon.
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))


rep=backend.prepare(model,device="CUDA:0")
outputs=rep.run(np.random.randn(1,3,640,640).astype(np.float32))
print outputs[0].shape



