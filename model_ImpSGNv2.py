# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from torch import nn
import torch
import math
from ms_tcn_TP1 import MultiScale_TemporalConv
from EfficientGCN_modules.graphs import Graph
from EfficientGCN_modules.layers import Basic_Layer, Spatial_Graph_Layer
from EfficientGCN_modules.activations import Swish
from EfficientGCN_modules.attentions import ST_Joint_Att
import numpy as np
### InfoGCN
from InfoGCN_modules.modules import EncodingBlock

class ImpSGNv2(nn.Module):
    def __init__(self, num_classes, dataset, seg, args, bias = True):
        super(ImpSGNv2, self).__init__()

        self.dim1 = 256
        self.dataset = dataset
        self.seg = seg
        num_joint = 25
        bs = args.batch_size
        ### added for multi_inputs_function()
        self.connect_joint = np.array([2,2,21,3,21,5,6,7,21,9,10,11,1,13,14,15,1,17,18,19,2,23,8,25,12]) - 1
        self.velocity_vect = [1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47]
        self.velocity_mx_prime = torch.Tensor(self.velocity_vect).repeat(bs, 3, seg, 1).permute(0, 1, 3, 2).cuda()



        if args.train:
            self.spa = self.one_hot(bs, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(bs, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()
        else:
            self.spa = self.one_hot(32 * 5, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(32 * 5, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()
            self.velocity_mx_prime = torch.Tensor(self.velocity_vect).repeat(32 * 5, 3, seg, 1).permute(0, 1, 3, 2).cuda()
            
	### changed for state04
        self.tem_embed = embed(self.seg, self.dim1, norm=False, bias=bias)  #self.tem_embed = embed(self.seg, 64*4, norm=False, bias=bias)
        self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)
        ## commented and modified to velocity and bone
#        self.joint_embed = embed(3, 64, norm=True, bias=bias)
        self.joint_embed = embed(dim=6, dim1=64, norm=True, bias=bias)        
        self.velocity_embed = embed(6, 64, norm=True, bias=bias)
        self.bone_embed = embed(6, 64, norm=True, bias=bias) 
        self.velocity_prime_embed = embed(6, 64, norm=True, bias=bias)
        
#        self.dif_embed = embed(3, 64, norm=True, bias=bias)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)  #self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)
#        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
#        self.gcn1 = gcn_spa(self.dim1 // 2, self.dim1 // 2, bias=bias)
#        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
#        self.gcn3 = gcn_spa(self.dim1, self.dim1, bias=bias)
        self.fc = nn.Linear(self.dim1 * 2, num_classes)
        # self.stab_modules = nn.ModuleList((
        #     Spatial_Temporal_Att_Block(128, 256, bias=bias),
        #     Spatial_Temporal_Att_Block(256, 256, bias=bias),
        #     Spatial_Temporal_Att_Block(256, 256*2, bias=bias),
        # ))
        self.st_infogcn = ST_InfoGCN(self.dim1//2, self.dim1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


    def forward(self, input):
        
        # Dynamic Representation
        bs, step, dim = input.size()
        num_joints = dim //3
        input = input.view((bs, step, num_joints, 3))
        input = input.permute(0, 3, 2, 1).contiguous()

        pos , velocity, bone, velocity_prime = self.multi_input_ImpSGNv2_2(input, self.connect_joint, self.velocity_mx_prime)
        pos = self.joint_embed(pos)			#(bs, 64, v, t)
        velocity = self.velocity_embed(velocity)	#(bs, 64, v, t)
        bone = self.bone_embed(bone)			#(bs, 64, v, t)
        velocity_prime = self.velocity_prime_embed(velocity_prime)	#(bs, 64, v, t)	


        tem1 = self.tem_embed(self.tem)
        spa1 = self.spa_embed(self.spa)

        dy = pos + velocity + bone + velocity_prime

        # Joint-level Module
        input= torch.cat([dy, spa1], 1)
        # ### add STB_modules:
        # for stab in self.stab_modules:
        #     input = stab(input)

        ### InfoGCN Spatio-Temporal Blocks
        input = input.permute(0, 1, 3, 2) #(n c v t) -> (n c t v)
        input = self.st_infogcn(input)
        input = input.permute(0, 1, 3, 2) #(n c t v) -> (n c v t)

        # Frame-level Module
        input = input + tem1
        input = self.cnn(input)
        # Classification
        output = self.maxpool(input)
        output = torch.flatten(output, 1)
        output = self.fc(output)

        return output

    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot

    ##### added
    def multi_input_ImpSGNv2_2(self, data, conn, velocity_mx_prime):
        ### this function "multi_input_ImpSGNv2_2" contains "velocity_mx_prime" compared with "multi_input_SImpGN"
        N, C, V, T = data.size()
        if torch.cuda.is_available():	# 'GPU'
            joint = torch.zeros((N, C*2, V, T)).cuda()
            velocity = torch.zeros((N, C*2, V, T)).cuda()
            bone = torch.zeros((N, C*2, V, T)).cuda()
            ### added : Bakir idea
            velocity_prime = torch.zeros((N, C*2, V, T)).cuda()
        else:				# 'CPU'
            joint = torch.zeros((N, C*2, V, T))
            velocity = torch.zeros((N, C*2, V, T))
            bone = torch.zeros((N, C*2, V, T))
            velocity_prime = torch.zeros((N, C*2, V, T))
            
        
        joint[:, :C,:,:] = data
        for i in range(V):
            joint[:, C:,i,:] = data[:, :,i, :] - data[:, :,1, :]
        for i in range(T-2):
            velocity[:, :C,:, i] = data[:, :,:, i+1] - data[:, :,:, i]
            velocity[:, C:,:, i] = data[:, :,:, i+2] - data[:, :,:, i]
        velocity_prime[:, :C, :, :] = torch.mul(velocity[:, :C, :, :], velocity_mx_prime)
        velocity_prime[:, C:, :, :] = torch.mul(velocity[:, C:, :, :], velocity_mx_prime)
        
        for i in range(len(conn)):
            bone[:, :C,i, :] = data[:, :,i, :] - data[:, :,conn[i], :]
        bone_length = 0
        for i in range(C):
            bone_length += bone[:, i,:,:] ** 2
        bone_length = torch.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[:, C+i,:,:] = torch.acos(bone[:, i,:,:] / bone_length)#torch.arccos(bone[:, i,:,:] / bone_length)
        return joint, velocity, bone, velocity_prime
##############################################
"""class bn_act(nn.Module):
    def __init__(self, dim1=3, dim2=3)
        super(bn_act, self).__init__()
        self.bn = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.bn(x)        
"""     
class norm_act(nn.Module):
    def __init__(self, dim= 256):
        super(norm_act, self).__init__()

        self.bn = nn.BatchNorm1d(dim* 25) #dim * 25: he did it because he will use x.view(bs, -1, step) {-1= (c=dim) * (25-joint) }
        self.relu = nn.ReLU()

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.contiguous()  #since we use graph.A, there are 2 types, we use contiguous() to harmonize types.
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, c, num_joints, step).contiguous()
        x = self.relu(x)
        return x   


class SpatialTemporalBlock(nn.Module):
    def __init__(self, in_channels=256//2, out_channels=256, bias=True):
        super(SpatialTemporalBlock, self).__init__()
        self.dim1 = 256
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.compute_g1 = compute_g_spa(self.in_channels, self.dim1, bias=bias)
        self.gcn1 = gcn_spa(self.in_channels, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.out_channels, bias=bias)
	### added: 
        self.bn_relu = norm_act(self.out_channels)
        self.output_feature = 42*6
        self.mstcn = MultiScale_TemporalConv(self.out_channels, self.output_feature) #42*6=252   #the output_features must be divided by 6 {6: 4 Branches + 2}
        self.upsampling = nn.Conv2d(in_channels=self.output_feature, out_channels=self.out_channels, kernel_size=1, bias=bias)
        self.residual = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, bias=bias)

        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)

    def forward(self, input):
	### input: bs,c:128=dim1/2, v, t
        res = self.residual(input)	#c: 128 -> 256

        g = self.compute_g1(input)	#bs, c:256=dim1, v, t
        input = self.gcn1(input, g)
        input = self.gcn2(input, g)
        input = self.gcn3(input, g)	#bs, c:256
	
	### added:
        input = self.bn_relu(input)
        input = input + res
        ### add ms_tcn:
        input = self.mstcn(input)  #output: (bs, 42*6, v, t)
        input = self.upsampling(input) # (bs, 42*6+4=256, v, t)
        input = input + res
        return input	#bs, out_channels=256, v, t


class Spatial_A_Temporal_att_Block(nn.Module):
    def __init__(self, in_channels=256 // 2, out_channels=256, bias=True):
        super(Spatial_A_Temporal_att_Block, self).__init__()
        self.dim1 = 256
        self.in_channels = in_channels
        self.out_channels = out_channels

        graph = Graph('ntu')  # , max_hop=1)
        kwargs = {
            'A': torch.tensor(graph.A).type(torch.FloatTensor),
            'edge': False,
            'bias': bias,
            'act': Swish(),
            'amp_ratio': 2,
        }
        self.SGL = Spatial_Graph_Layer(self.in_channels, self.out_channels, 1, **kwargs)

        ### added:
        self.bn_relu = norm_act(self.out_channels)
        self.output_feature = 42 * 6
        self.mstcn = MultiScale_TemporalConv(self.out_channels,
                                             self.output_feature)  # 42*6=252   #the output_features must be divided by 6 {6: 4 Branches + 2}

        self.attention = ST_Joint_Att(self.output_feature,  reduct_ratio=2, **kwargs)

        self.upsampling = nn.Conv2d(in_channels=self.output_feature, out_channels=self.out_channels, kernel_size=1,
                                    bias=bias)
        self.residual = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1,
                                  bias=bias)
    def forward(self, input):
        ### input: bs,c:128=dim1/2, v, t
        res = self.residual(input)  # c: 128 -> 256

        input = input.permute(0, 1, 3, 2) #(bs, c, t, v)
        input = self.SGL(input) #(bs, 256, t, v)
        input = input.permute(0, 1, 3, 2)   #(bs, c, v, t)
        input = self.bn_relu(input)
        input = input + res

        input = self.mstcn(input)  # output: (bs, 42*6, v, t)
        input = input.permute(0, 1, 3, 2)
        input = self.attention(input)
        input = input.permute(0, 1, 3, 2)
        input = self.upsampling(input)  # (bs, 42*6+4=256, v, t)

        input = input + res
        return input  # bs, out_channels=256, v, t


class Spatial_Temporal_Att_Block(nn.Module):
    def __init__(self, in_channels=256 // 2, out_channels=256, bias=True):
        super(Spatial_Temporal_Att_Block, self).__init__()
        self.dim1 = 256
        self.in_channels = in_channels
        self.out_channels = out_channels

        graph = Graph('ntu')  # , max_hop=1)
        kwargs = {
            'A': torch.tensor(graph.A).type(torch.FloatTensor),
            'edge': False,
            'bias': bias,
            'act': Swish(),
            'amp_ratio': 2,
        }

        self.compute_g1 = compute_g_spa(self.in_channels, self.dim1, bias=bias)
        self.gcn1 = gcn_spa(self.in_channels, self.dim1 // 2, bias=bias)
        self.gcn2 = gcn_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn3 = gcn_spa(self.dim1, self.out_channels, bias=bias)
        ### added:
        self.bn_relu = norm_act(self.out_channels)
        self.output_feature = 42 * 6
        self.mstcn = MultiScale_TemporalConv(self.out_channels,
                                             self.output_feature)  # 42*6=252   #the output_features must be divided by 6 {6: 4 Branches + 2}
        self.upsampling = nn.Conv2d(in_channels=self.output_feature, out_channels=self.out_channels, kernel_size=1, bias=bias)
        self.attention = ST_Joint_Att(self.out_channels, reduct_ratio=2, **kwargs)
        self.residual = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, bias=bias)


        nn.init.constant_(self.gcn1.w.cnn.weight, 0)
        nn.init.constant_(self.gcn2.w.cnn.weight, 0)
        nn.init.constant_(self.gcn3.w.cnn.weight, 0)

    def forward(self, input):
        ### input: bs,c:128=dim1/2, v, t
        res = self.residual(input)  # c: 128 -> 256

        g = self.compute_g1(input)  # bs, c:256=dim1, v, t
        input = self.gcn1(input, g)
        input = self.gcn2(input, g)
        input = self.gcn3(input, g)  # bs, c:256

        ### added:
        input = self.bn_relu(input)
        input = input + res
        ### add ms_tcn:
        input = self.mstcn(input)  # output: (bs, 42*6, v, t)
        input = self.upsampling(input)  # (bs, 42*6+4=256, v, t)
        res2 = input
        input = self.attention(input)
        input = input + res2
        input = input + res
        return input  # bs, out_channels=256, v, t



class ST_InfoGCN(nn.Module):
    def __init__(self, in_channels=64, out_channels=64*4, num_point=25, num_head=3):
        super(ST_InfoGCN, self).__init__()
        A = np.stack([np.eye(num_point)] * num_head, axis=0)
        
        base_channel = 64
        
        self.l1 = EncodingBlock(in_channels, base_channel,A)
        self.l2 = EncodingBlock(base_channel, base_channel,A)
        self.l3 = EncodingBlock(base_channel, base_channel,A)
        self.l4 = EncodingBlock(base_channel, base_channel*2, A) #, stride=2)
        self.l5 = EncodingBlock(base_channel*2, base_channel*2, A)
        self.l6 = EncodingBlock(base_channel*2, base_channel*2, A)
        self.l7 = EncodingBlock(base_channel*2, base_channel*4, A) #, stride=2)
        self.l8 = EncodingBlock(base_channel*4, base_channel*4, A)
        self.l9 = EncodingBlock(base_channel*4, out_channels, A)

    def forward(self, x):
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        return x
    
##############################################

class norm_data(nn.Module):
    def __init__(self, dim= 64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim* 25)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x

class embed(nn.Module):
    def __init__(self, dim = 3, dim1 = 128, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x

class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

class local(nn.Module):
    def __init__(self, dim1 = 3, dim2 = 3, bias = False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias = False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)


    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x

class compute_g_spa(nn.Module):
    def __init__(self, dim1 = 64 *3, dim2 = 64*3, bias = False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):

        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g



#################################################
if __name__ == '__main__':
    ### Testing ST_InfoGCN():
    # n, c, t, v = 4, 64, 20, 25
    # st_infogcn = ST_InfoGCN(c, c * 4)
    # x = torch.randn(n, c, t, v)
    # y = st_infogcn(x)
    # print('****', x.size(), y.size())
    # assert False, 'aaaaa'

    # criterion = nn.MSELoss()    #nn.CrossEntropyLoss()
    # input = torch.randn(2, 3, 3, 5, requires_grad=True)
    # model_output = torch.randn(2, 3, 3, 5, requires_grad=True)
    # # target = torch.randint(0, 10, (2, )) #target = torch.empty(3, dtype=torch.long).random_(5)
    # loss = criterion(input, model_output)
    # # print(criterion)
    # print(input.size(),model_output.size(), loss)
    #
    # assert False, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    # norm = norm_act(256)
    # x = torch.randn(2, 256, 25, 20)
    # y = norm(x)
    # print('************ norm: ', x.size(), y.size())
    # assert False, 'bbbbbbbbbbbbbb'
    #######
    # bs, c, v, t = 2, 64, 25, 20
    # stab = Spatial_A_Temporal_att_Block(c, c*4, bias=False)
    # # stb = SpatialTemporalBlock(128, 256, bias=True)
    # x = torch.randn(bs, c, v, t)
    # y = stab(x)
    # print(x.size(), y.size())
    # print(stab)
    # assert False, 'aaaaaaaaaaaaaaaaaaaa'
    ########

    import argparse

    parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
    #	fit.add_fit_args(parser)
    parser.set_defaults(
        network='ImpSGNv2',
        dataset = 'NTU',
        case = 0,
        batch_size=2, #64,
        max_epochs=5,
        monitor='val_acc',
        lr=0.001,
        weight_decay=0.0001,
        lr_factor=0.1,
        workers=4, #16,
        print_freq = 20,
        train = 1,
        seg = 20,
        )
    args = parser.parse_args()


    args.num_classes = 60
    model = ImpSGNv2(args.num_classes, args.dataset, args.seg, args)


    #	x = torch.randn(2, 3, 10, 15)
    x = torch.randn(2, 20, 75)
    #	x = x.cuda()
    #	model = model.cuda()
    #
    #	cnn = cnn1x1(3, 3, bias=True).cuda()
    #	y = cnn(x)

    x = x.cuda()
    model = model.cuda()

    y = model(x)
    print('*** x, y: ', x.size(), y.size())

    print('hello world !!!')
    #	print(model)

"""	###
	bs, c, v, t = 2, 128, 25, 20
	stb = SpatialTemporalBlock(in_channels=c, out_channels=256, bias=True)
#	print(stb)
	x = torch.randn(bs, c, v, t)
	stb = stb.cuda()
	x = x.cuda()
	y = stb(x)
	print('****** ', x.size(), y.size())
	assert False, 'aaaaaaaaaaaaaaaaaa'
"""
