### """ Program got from github: https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py  """



import torch
import torch.nn as  nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 1 #4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


        
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        ### modified
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)  #self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        ### modified
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=1, padding=1)  #self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=1) #stride=2
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=1) #self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=1)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):

        ### SGN code for input:
        bs, step, dim = x.size()
        num_joints = dim //3
        x = x.view((bs, step, num_joints, 3))
        x = x.permute(0, 3, 2, 1).contiguous()  #[bs, 3, 25, 20]


        ### ResNet code:
        print("*******1) x", x.size())
        x = self.relu(self.batch_norm1(self.conv1(x)))
        print("*******2) x", x.size())
        x = self.max_pool(x)
        print("*******3) x", x.size())
        x = self.layer1(x)
        print("******* layer1) x", x.size())
        x = self.layer2(x)
        print("******* layer2) x", x.size())
        x = self.layer3(x)
        print("******* layer3) x", x.size())
        x = self.layer4(x)
        print("******* layer4) x", x.size())
        
        x = self.avgpool(x)
        print("******* avgpool) x", x.size())
        
        x = x.reshape(x.shape[0], -1)
        print("******* x.shape) x", x.size())
        x = self.fc(x)
        print("******* fc) x", x.size())
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

        

def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)


### added functions:
def ResNet10(num_classes, channels=3):
    return ResNet(Bottleneck, [1,1,1,1], num_classes, channels)
def ResNet18(num_classes, channels=3):
    return ResNet(Bottleneck, [2,2,2,2], num_classes, channels)

##############################################

if __name__ == '__main__':
	import torch
	
	
	model  = ResNet10(60).cuda()

#	x = torch.randn(4, 3,25, 20).cuda()
	x = torch.randn(4, 20, 75).cuda()
	
	y = model(x)
	print(x.size(), y.size())
	
#	print('*****************************\n\n', model)
	print("test")
