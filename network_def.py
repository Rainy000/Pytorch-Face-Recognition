from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter, ModuleList
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class Flatten(Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MyAdaptiveAvePool2d(Module):
    def __init__(self, sz=None, Train=True):
        super().__init__()
        self.sz = sz
        self.Train = Train
        self.sz_list = range(0,100)

    def forward(self, x):
        if self.Train:
            return AdaptiveAvgPool2d(self.sz)(x)
        else:
            return nn.AvgPool2d(kernel_size=(self.sz_list.index(x.size(2)), self.sz_list.index(x.size(3))), ceil_mode=False)



def l2_norm(x, axis=1):
    norm = torch.norm(x, 2, axis, True)
    output = x / norm
    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.global_avg_pool = AdaptiveAvgPool2d(1)
        #self.global_avg_pool = MyAdaptiveAvePool2d(Train=False)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.global_avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BottleneckIR(Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckIR, self).__init__()
        self.identity = 0
        if in_channels == out_channels:
            if stride == 1:
                self.identity = 1
            else:
                self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channels, out_channels, (1, 1), stride, bias=False),
                                             BatchNorm2d(out_channels))
        self.res_layer = Sequential(BatchNorm2d(in_channels),
                                    Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
                                    BatchNorm2d(out_channels),
                                    PReLU(out_channels),
                                    Conv2d(out_channels, out_channels, (3, 3), stride, 1, bias=False),
                                    BatchNorm2d(out_channels))

    def forward(self, x):
        shortcut = x if self.identity == 1 else self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class BottleneckIRSE(Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckIRSE, self).__init__()
        self.identity = 0
        if in_channels == out_channels:
            if stride == 1:
                self.identity = 1
            else:
                self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channels, out_channels, (1, 1), stride, bias=False),
                                             BatchNorm2d(out_channels))
        self.res_layer = Sequential(BatchNorm2d(in_channels),
                                    Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
                                    BatchNorm2d(out_channels),
                                    PReLU(out_channels),
                                    Conv2d(out_channels, out_channels, (3, 3), stride, 1, bias=False),
                                    BatchNorm2d(out_channels),
                                    SEModule(out_channels, 16))

    def forward(self, x):
        shortcut = self.shortcut_layer(x) if self.identity != 1 else x
        res = self.res_layer(x)
        return res + shortcut


class BottleneckIRSE_V2(Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckIRSE_V2, self).__init__()
        self.identity = 0
        if in_channels == out_channels:
            if stride == 1:
                self.identity = 1
            else:
                #self.shortcut_layer = MaxPool2d(1, stride)
                self.shortcut_layer = AvgPool2d(3,stride,1)
        else:
            #self.shortcut_layer = Sequential(Conv2d(in_channels, out_channels, (1, 1), stride, bias=False),
            #                                 BatchNorm2d(out_channels))
            self.shortcut_layer = Sequential(Conv2d(in_channels, out_channels, (3, 3), stride, 1, bias=False),
                                             BatchNorm2d(out_channels))
        self.res_layer = Sequential(BatchNorm2d(in_channels),
                                    Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
                                    BatchNorm2d(out_channels),
                                    PReLU(out_channels),
                                    Conv2d(out_channels, out_channels, (3, 3), stride, 1, bias=False),
                                    BatchNorm2d(out_channels),
                                    SEModule(out_channels, 16))

    def forward(self, x):
        shortcut = self.shortcut_layer(x) if self.identity != 1 else x
        res = self.res_layer(x)
        return res + shortcut

class BasicResBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResBlock, self).__init__()
        self.identity = 0
        if in_channels == out_channels:
            if stride == 1:
                self.identity = 1
            else:
                self.shortcut_layer = MaxPool2d(2, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                             BatchNorm2d(out_channels))
        self.res_layer = Sequential(Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
                                    BatchNorm2d(out_channels),
                                    ReLU(inplace=True),
                                    Conv2d(out_channels, out_channels, (3, 3), stride, 1, bias=False),
                                    BatchNorm2d(out_channels))

    def forward(self, x):
        shortcut = x if self.identity == 1 else self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class MaskModule(Module):
    def __init__(self, down_sample_times, out_channels, r, net_mode='ir'):
        super(MaskModule, self).__init__()
        assert net_mode in ('ir', 'basic', 'irse', 'irse_v2')
        func = {'ir': BottleneckIR, 'irse': BottleneckIRSE, 'irse_v2': BottleneckIRSE_V2, 'basic': BasicResBlock}

        self.max_pool_layers = ModuleList()
        for i in range(down_sample_times):
            self.max_pool_layers.append(MaxPool2d(2, 2))

        self.prev_res_layers = ModuleList()
        for i in range(down_sample_times):
            tmp_prev_res_block_layers = []
            for j in range(r):
                tmp_prev_res_block_layers.append(func[net_mode](out_channels, out_channels, 1))
            self.prev_res_layers.append(Sequential(*tmp_prev_res_block_layers))

        self.mid_res_layers = None
        self.post_res_layers = None
        if down_sample_times > 1:
            self.mid_res_layers = ModuleList()
            for i in range(down_sample_times - 1):
                self.mid_res_layers.append(func[net_mode](out_channels, out_channels, 1))

            self.post_res_layers = ModuleList()
            for i in range(down_sample_times - 1):
                tmp_post_res_block_layers = []
                for j in range(r):
                    tmp_post_res_block_layers.append(func[net_mode](out_channels, out_channels, 1))
                self.post_res_layers.append(Sequential(*tmp_post_res_block_layers))

        self.r = r
        self.out_channels = out_channels
        self.down_sample_times = down_sample_times

    def mask_branch(self, x, cur_layers, down_sample_times):
        h = x.shape[2]
        w = x.shape[3]

        cur_layers.append(self.max_pool_layers[self.down_sample_times - down_sample_times](x))

        cur_layers.append(self.prev_res_layers[self.down_sample_times - down_sample_times](cur_layers[-1]))
        # down_sample_times -= 1
        if down_sample_times - 1 <= 0:

            cur_layers.append(F.interpolate(cur_layers[-1], (h, w), mode='bilinear', align_corners=True))
            return cur_layers[-1]
        else:
            cur_layers.append(self.mid_res_layers[self.down_sample_times - down_sample_times](cur_layers[-1]))

            shortcut_layer = cur_layers[-1]
            v = self.mask_branch(cur_layers[-1], cur_layers, down_sample_times - 1)
            cur_layers.append(shortcut_layer + v)

            cur_layers.append(self.post_res_layers[self.down_sample_times - down_sample_times](cur_layers[-1]))
            cur_layers.append(F.interpolate(cur_layers[-1], (h, w), mode='bilinear', align_corners=True))
            return cur_layers[-1]

    def forward(self, x):
        cur_layers = []
        return self.mask_branch(x, cur_layers, self.down_sample_times)


class AttentionModule(Module):
    def __init__(self, in_channels, out_channels, input_spatial_dim, p=1, t=2, r=1, net_mode='ir'):
        super(AttentionModule, self).__init__()
        self.func = {'ir': BottleneckIR, 'irse': BottleneckIRSE, 'irse_v2': BottleneckIRSE_V2, 'basic': BasicResBlock}

        # start branch
        self.start_branch = ModuleList()
        self.start_branch.append(self.func[net_mode](in_channels, out_channels, 1))
        for i in range(p - 1):
            self.start_branch.append(self.func[net_mode](out_channels, out_channels, 1))

        # trunk branch
        self.trunk_branch = ModuleList()
        for i in range(t):
            self.trunk_branch.append(self.func[net_mode](out_channels, out_channels, 1))

        # mask branch
        # 1st, determine how many down-sample operations should be executed.
        num_down_sample_times = 0
        resolution = input_spatial_dim
        while resolution > 4 and resolution not in (8, 7, 6, 5):
            num_down_sample_times += 1
            resolution = (resolution - 2) / 2 + 1
        self.num_down_sample_times = min(num_down_sample_times, 100)
        self.mask_branch = MaskModule(num_down_sample_times, out_channels, r, net_mode)

        self.mask_helper = Sequential(Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
                                      BatchNorm2d(out_channels),
                                      ReLU(inplace=True),
                                      Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
                                      BatchNorm2d(out_channels),
                                      Sigmoid())
        # output branch
        self.out_branch = ModuleList()
        for i in range(p):
            self.out_branch.append(self.func[net_mode](out_channels, out_channels, 1))
        self.p = p
        self.t = t
        self.r = r

    def forward(self, x):
        for i in range(self.p):
            x = self.start_branch[i](x)
        y = x
        for i in range(self.t):
            x = self.trunk_branch[i](x)

        trunk = x
        mask = self.mask_branch(y)
        mask = self.mask_helper(mask)
        out = trunk * (mask + 1)
        for i in range(self.p):
            out = self.out_branch[i](out)
        return out


class AttentionNet(Module):
    def __init__(self, in_channels=3, p=1, t=2, r=1, net_mode='ir', attention_stages=(1, 1, 1), img_dim=120):  # 362 383
        super(AttentionNet, self).__init__()
        final_res_block = 3
        func = {'ir': BottleneckIR, 'irse': BottleneckIRSE, 'irse_v2': BottleneckIRSE_V2, 'basic': BasicResBlock}
        self.input_layer = Sequential(Conv2d(in_channels, 64, 3, 1, 1),
                                      BatchNorm2d(64),
                                      ReLU(inplace=True),
                                      func[net_mode](64, 64, 2)) #Conv2d(in_channels, 64, 3, 1, 1, bias=False)  #PReLU(64)
        input_spatial_dim = (img_dim - 1) // 2 + 1
        modules = []

        # stage 1 (56*56)
        for i in range(attention_stages[0]):
            modules.append(AttentionModule(64, 64, input_spatial_dim, p, t, r, net_mode))

        modules.append(func[net_mode](64, 128, 2))
        input_spatial_dim = (input_spatial_dim - 1) // 2 + 1

        # stage 2 (28*28)
        for i in range(attention_stages[1]):
            modules.append(AttentionModule(128, 128, input_spatial_dim, p, t, r, net_mode))

        modules.append(func[net_mode](128, 256, 2))
        input_spatial_dim = (input_spatial_dim - 1) // 2 + 1

        # stage 3 (14*14)
        for i in range(attention_stages[2]):
            modules.append(AttentionModule(256, 256, input_spatial_dim, p, t, r, net_mode))

        modules.append(func[net_mode](256, 512, 2))
        input_spatial_dim = (input_spatial_dim - 1) // 2 + 1
        ## stage final (7*7)
        for i in range(final_res_block):
            modules.append(func[net_mode](512, 512, 1))

        self.body = Sequential(*modules)
        self.output_layer = Sequential(BatchNorm2d(512),
                                       Dropout(0.4),
                                       Flatten(),
                                       Linear(512 * input_spatial_dim * input_spatial_dim, 512, False),
                                       BatchNorm1d(512))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x


class Feat_Net(Module):
    def __init__(self, in_channels=3, img_dim=112, net_mode='ir', p=1, t=2, r=1, attention_stages=(1, 1, 1)):
        super(Feat_Net, self).__init__()
        self.feat_net = AttentionNet(in_channels=in_channels, p=p, t=t, r=r, net_mode=net_mode, attention_stages=attention_stages, img_dim=img_dim)

    def forward(self, inputs):
        feat = self.feat_net.forward(inputs)
        return feat

def feat_cosine(w, dim =1):
    w = nn.functional.normalize(w, p=2, dim=dim)
    ip = torch.mm(w, w.t()).clamp(-1, 1)
    return ip
               
               
class LMRegularProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, m=0.4, factor=5):
        super(LMRegularProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = scale
        self.m = m
        self.factor = factor
        self.weight = Parameter(torch.randn(out_features, in_features))
        stdv = 1./math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -stdv, stdv)
        
    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        lm_output = self.s*(cosine-one_hot*self.m)
        
        cos = feat_cosine(self.weight, dim=1)
        cos_cp = cos.detach()
        cos_cp.scatter_(1, torch.arange(self.out_features).view(-1,1).long().cuda(), -100)
        _, indices = torch.max(cos_cp, dim=0)
        mask = torch.zeros((self.out_features, self.out_features)).cuda()
        mask.scatter_(1, indices.view(-1, 1).long(), 1)
        regular_loss = torch.dot(cos.view(cos.numel()), mask.view(mask.numel()))/ self.out_features
        return lm_output, regular_loss*self.factor


def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    # ip = torch.mm(x1, x2)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1,w2).clamp(min=eps)

class MarginCosineProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """
    def __init__(self, in_features, out_features, scale=30.0, m=0.40):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = scale
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'
               

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

'''
class ArcMarginProduct(Module):
    """Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
    """
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, num_class=86876, s=64., m=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.num_class = num_class
        self.weight = Parameter(torch.Tensor(embedding_size, num_class))
        # initial kernel
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
        self.easy_margin = easy_margin

    def forward(self, embeddings, label):

        kernel_norm = l2_norm(self.weight, axis=0)
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        cos_theta = torch.mm(embeddings, kernel_norm)
        #         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            cos_theta_m = torch.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.mm)

        # label = label.view(-1, 1)  # size=(B,1)
        output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
        batch_size = label.size(0)
        output[torch.arange(0, batch_size), label] = cos_theta_m[torch.arange(0, batch_size), label]
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output
'''



class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'
