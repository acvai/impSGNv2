import math

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, einsum
from torch.autograd import Variable
# from torch import linalg as LA        #commented => bug

# from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from ms_tcn import MultiScale_TemporalConv as MS_TCN
from einops import rearrange, repeat

from utils import set_parameter_requires_grad, get_vector_property

# from model.modules import import_class, bn_init, EncodingBlock
from modules import import_class, bn_init, EncodingBlock


class InfoGCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, in_channels=3,
                 drop_out=0, num_head=3, noise_ratio=0.1, k=0, gain=1):
        super(InfoGCN, self).__init__()

        A = np.stack([np.eye(num_point)] * num_head, axis=0)

        base_channel = 64
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * base_channel * num_point)
        self.noise_ratio = noise_ratio
        self.z_prior = torch.empty(num_class, base_channel*4)
        self.A_vector = self.get_A(graph, k)
        self.gain = gain
        self.to_joint_embedding = nn.Linear(in_channels, base_channel)
        #### added:
        self.to_velocity_embedding = nn.Linear(in_channels, base_channel)
        self.to_bone_embedding = nn.Linear(in_channels, base_channel)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel))

        self.connect_joint = np.array([2, 2, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 23, 8, 25, 12]) - 1
        self.j1 = EncodingBlock(base_channel, 48, A)
        self.j2 = EncodingBlock(48, 16, A)
        self.v1 = EncodingBlock(base_channel, 48, A)
        self.v2 = EncodingBlock(48, 16, A)
        self.b1 = EncodingBlock(base_channel, 48, A)
        self.b2 = EncodingBlock(48, 16, A)
        self.m1 = EncodingBlock(48, 64, A, stride=2)
        self.m2 = EncodingBlock(64, 128, A, stride=2)
        self.m3 = EncodingBlock(128, 256, A, stride=2)

        # self.l1 = EncodingBlock(base_channel, base_channel,A)
        # self.l2 = EncodingBlock(base_channel, base_channel,A)
        # self.l3 = EncodingBlock(base_channel, base_channel,A)
        # self.l4 = EncodingBlock(base_channel, base_channel*2, A, stride=2)
        # self.l5 = EncodingBlock(base_channel*2, base_channel*2, A)
        # self.l6 = EncodingBlock(base_channel*2, base_channel*2, A)
        # self.l7 = EncodingBlock(base_channel*2, base_channel*4, A, stride=2)
        # self.l8 = EncodingBlock(base_channel*4, base_channel*4, A)
        # self.l9 = EncodingBlock(base_channel*4, base_channel*4, A)

        self.fc = nn.Linear(base_channel*4, base_channel*4)
        self.fc_mu = nn.Linear(base_channel*4, base_channel*4)
        self.fc_logvar = nn.Linear(base_channel*4, base_channel*4)
        self.decoder = nn.Linear(base_channel*4, num_class)
        nn.init.orthogonal_(self.z_prior, gain=gain)
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_mu.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.normal_(self.decoder.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def get_A(self, graph, k):
        Graph = import_class(graph)()
        A_outward = Graph.A_outward_binary
        I = np.eye(Graph.num_node)
        return  torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def latent_sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(self.noise_ratio).exp()
            # std = logvar.exp()
            std = torch.clamp(std, max=100)
            # std = std / (torch.norm(std, 2, dim=1, keepdim=True) + 1e-4)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std) + mu
        else:
            return mu

    def forward(self, x):
        pos, vel, bone = self.multi_input_jvb(x, self.connect_joint)                #[4, 6, 64, 25, 2], [4, 6, 64, 25, 2], [4, 6, 64, 25, 2]

        N, C, T, V, M = pos.size()  #x.size() # [4, 3, 64, 25, 2]
        pos = rearrange(pos, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
        # x = self.A_vector.to(x.device).expand(N*M*T, -1, -1) @ x                  #buggeed
        pos = self.A_vector.type(torch.cuda.FloatTensor).expand(N * M * T, -1, -1) @ pos

        pos = self.to_joint_embedding(pos)
        pos += self.pos_embedding[:, :self.num_point]
        pos = rearrange(pos, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()

        pos = self.data_bn(pos)
        pos = rearrange(pos, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()
        ##### added: EfficientGCN like network:
        pos = self.j1(pos)
        pos = self.j2(pos)
    ######### Velocity
        vel = rearrange(vel, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
        vel = self.A_vector.type(torch.cuda.FloatTensor).expand(N * M * T, -1, -1) @ vel

        vel = self.to_velocity_embedding(vel)
        vel += self.pos_embedding[:, :self.num_point]
        vel = rearrange(vel, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()

        vel = self.data_bn(vel)
        vel = rearrange(vel, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()
        vel = self.v1(vel)
        vel = self.v2(vel)
    ######### Bone
        bone = rearrange(bone, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
        bone = self.A_vector.type(torch.cuda.FloatTensor).expand(N * M * T, -1, -1) @ bone

        bone = self.to_bone_embedding(bone)
        bone += self.pos_embedding[:, :self.num_point]
        bone = rearrange(bone, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()

        bone = self.data_bn(bone)
        bone = rearrange(bone, 'n (m v c) t -> (n m) c t v', m=M, v=V).contiguous()
        bone = self.b1(bone)
        bone = self.b2(bone)
        xm = torch.cat([pos, vel, bone], dim=1)
        xm = self.m1(xm)
        xm = self.m2(xm)
        xm = self.m3(xm)
        # N*M,C,T,V
        c_new = xm.size(1)
        xm = xm.view(N, M, c_new, -1)
        xm = xm.mean(3).mean(1)
        xm = F.relu(self.fc(xm))
        xm = self.drop_out(xm)

        z_mu = self.fc_mu(xm)
        z_logvar = self.fc_logvar(xm)
        z = self.latent_sample(z_mu, z_logvar)

        y_hat = self.decoder(z)

        return y_hat, z


        # x = self.l1(x)
        # x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)
        # x = self.l5(x)
        # x = self.l6(x)
        # x = self.l7(x)
        # x = self.l8(x)
        # x = self.l9(x)

        # # N*M,C,T,V
        # c_new = x.size(1)
        # x = x.view(N, M, c_new, -1)
        # x = x.mean(3).mean(1)
        # x = F.relu(self.fc(x))
        # x = self.drop_out(x)
        #
        # z_mu = self.fc_mu(x)
        # z_logvar = self.fc_logvar(x)
        # z = self.latent_sample(z_mu, z_logvar)
        #
        # y_hat = self.decoder(z)
        #
        # return y_hat, z

    ##### added
    def multi_input_jvb(self, data, conn):
        N, C, T, V, M = data.size()
        if torch.cuda.is_available():  # 'GPU'
            joint = torch.zeros((N, C * 2, T, V, M)).cuda()
            velocity = torch.zeros((N, C * 2, T, V, M)).cuda()
            bone = torch.zeros((N, C * 2, T, V, M)).cuda()
        else:  # 'CPU'
            joint = torch.zeros((N, C * 2, T, V, M))
            velocity = torch.zeros((N, C * 2, T, V, M))
            bone = torch.zeros((N, C * 2, T, V, M))

        joint[:, :C, :, :, :] = data
        for i in range(V):
            joint[:, C:, :, i, :] = data[:, :, :, i, :] - data[:, :, : , 1, :]
        for i in range(T - 2):
            velocity[:, :C, i, :, :] = data[:, :, i + 1, :, :] - data[:, :, i, :, :]
            velocity[:, C:, i, :, :] = data[:, :, i + 2, :, :] - data[:, :, i, :, :]

        for i in range(len(conn)):
            bone[:, :C, :, i, :] = data[:, :, :, i, :] - data[:, :, :, conn[i], :]
        bone_length = 0
        for i in range(C):
            bone_length += bone[:, i, :, :, :] ** 2
        bone_length = torch.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[:, C + i, :, :, :] = torch.arccos(bone[:, i, :, :, :] / bone_length)
        return joint, velocity, bone


#############################################
if __name__ == '__main__':
    print("hello world!")

    from graph.ntu_rgb_d import Graph

    #	graph = Graph(labeling_mode='spatial')
    #	print(graph.get_adjacency_matrix()[2, :, :])
    #	assert False, 'aaaaaaaaaaaaaaa'

    graph = 'graph.ntu_rgb_d.Graph'

    n_heads, k, z_prior_gain, noise_ratio = 3, 0, 3, 0.5

    model = InfoGCN(
        num_class=60,
        num_point=25,
        num_person=2,
        graph=graph,
        in_channels=6,  #for multi_input_infogcn   #default:3,
        drop_out=0,
        num_head=n_heads,
        k=k,
        noise_ratio=noise_ratio,
        gain=z_prior_gain
    )

    # print(model)

    ### Input InfoGCN:
    N, C, T, V, M = 4, 3, 64, 25, 2         #4, 3, 20, 25, 2
    x = torch.randn(N, C, T, V, M)
    ### Input InfoGCN-SGN-framework
    # x = torch.randn(N, T, C * V)

    #	x = torch.randn(size=(N, C, T, V, M), out=None, dtype=torch.FloatTensor, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
    #

    model = model.cuda()
    x = x.cuda()
    y, z = model(x)
    print(x.size(), 'classes: ', y.size(), 'other: ', z.size())