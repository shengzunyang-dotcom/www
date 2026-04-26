import numpy as np
import torch 

a = np.array([[1,2,3],[4,5,6]])
c = np.array([2,2,2])
d = np.array([2,2,2])
b = torch.tensor([[1,2,3],[4,5,6]])
print(torch.sum(b).item())

def ipot(self,C, beta=1, K=1, max_iter=50):
        n, m = C.shape  # 获取源分布和目标分布的大小
        # beta:0.01
        # 初始传输矩阵 T 和标准化因子 sigma
        T = torch.ones((n, m), dtype=torch.float32,device = DEVICE) / (n * m)
        sigma = torch.full((m,), 1.0 / m, dtype=torch.float32,device = DEVICE)  # 初始化sigma

        # 权重矩阵 A
        A = torch.exp(-C / beta)

        # 迭代计算最优传输矩阵
        for _ in range(max_iter):
            # 更新 Q
            Q = A * T
            # 内部迭代计算
            for _ in range(K):
                # 更新 delta 和 sigma
                delta = 1 / (n * torch.matmul(Q, sigma))
                sigma = 1 / (m * torch.matmul(Q.t(), delta))

            # 更新传输矩阵 T
            T = torch.diag(delta) @ Q @ torch.diag(sigma)
        return T, C


batch,length,dim = coarsed_t.size()
        a_t_g_matrix = self.get_cost_matrix(coarsed_a,coarsed_t)
        a_t_g_T, a_t_g_loss= None, 0.0
        for idx in range(len(a_t_g_matrix)):
            a_t_g_T_tmp,_ = self.ipot(a_t_g_matrix[idx])
            a_t_g_loss_tmp = torch.sum(a_t_g_T_tmp*a_t_g_matrix[idx]).item()
            if a_t_g_T is None:
                a_t_g_T = a_t_g_T_tmp
                a_t_g_loss = a_t_g_loss_tmp
            else:
                a_t_g_T = torch.cat([a_t_g_T, a_t_g_T_tmp], dim=0)
                a_t_g_loss = a_t_g_loss + a_t_g_loss_tmp

        v_t_g_matrix = self.get_cost_matrix(coarsed_v,coarsed_t)
        v_t_g_T ,v_t_g_loss= None,0.0
        for idx in range(len(v_t_g_matrix)):
            v_t_g_T_tmp,_ = self.ipot(v_t_g_matrix[idx])
            v_t_g_loss_tmp = torch.sum(v_t_g_T_tmp*v_t_g_matrix[idx]).item()
            if v_t_g_T is None:
                v_t_g_T = v_t_g_T_tmp
                v_t_g_loss = v_t_g_loss_tmp
            else:
                v_t_g_T = torch.cat([v_t_g_T, v_t_g_T_tmp], dim=0)
                v_t_g_loss = v_t_g_loss + v_t_g_loss_tmp

        a_v_g_matrix = self.get_cost_matrix(coarsed_a,coarsed_v)
        a_v_g_T ,a_v_g_loss= None,0.0
        for idx in range(len(a_v_g_matrix)):
            a_v_g_T_tmp,_ = self.ipot(a_v_g_matrix[idx])
            a_v_g_loss_tmp = torch.sum(a_v_g_T_tmp*a_v_g_matrix[idx]).item()
            if a_v_g_T is None:
                a_v_g_T = a_v_g_T_tmp
                a_v_g_loss = a_v_g_loss_tmp
            else:
                a_v_g_T = torch.cat([a_v_g_T, a_v_g_T_tmp], dim=0)
                a_v_g_loss = a_v_g_loss + a_v_g_loss_tmp

        g_loss = (a_t_g_loss + v_t_g_loss + a_v_g_loss)/batch

batch,length,dim = finegrained_t.size()
        a_t_l_matrix = self.get_cost_matrix(finegrained_a,finegrained_t)
        a_t_l_T, a_t_l_loss= None, 0.0
        for idx in range(len(a_t_l_matrix)):
            a_t_l_T_tmp,_ = self.ipot(a_t_l_matrix[idx])
            a_t_l_loss_tmp = torch.sum(a_t_l_T_tmp*a_t_l_matrix[idx]).item()
            if a_t_l_T is None:
                a_t_l_T = a_t_l_T_tmp
                a_t_l_loss = a_t_l_loss_tmp
            else:
                a_t_l_T = torch.cat([a_t_l_T, a_t_l_T_tmp], dim=0)
                a_t_l_loss = a_t_l_loss + a_t_l_loss_tmp

        v_t_l_matrix = self.get_cost_matrix(finegrained_v,finegrained_t)
        v_t_l_T ,v_t_l_loss= None,0.0
        for idx in range(len(v_t_l_matrix)):
            v_t_l_T_tmp,_ = self.ipot(v_t_l_matrix[idx])
            v_t_l_loss_tmp = torch.sum(v_t_l_T_tmp*v_t_l_matrix[idx]).item()
            if v_t_l_T is None:
                v_t_l_T = v_t_l_T_tmp
                v_t_l_loss = v_t_l_loss_tmp
            else:
                v_t_l_T = torch.cat([v_t_l_T, v_t_l_T_tmp], dim=0)
                v_t_l_loss = v_t_l_loss + v_t_l_loss_tmp

        a_v_l_matrix = self.get_cost_matrix(finegrained_a,finegrained_v)
        a_v_l_T ,a_v_l_loss= None,0.0
        for idx in range(len(a_v_l_matrix)):
            a_v_l_T_tmp,_ = self.ipot(a_v_l_matrix[idx])
            a_v_l_loss_tmp = torch.sum(a_v_l_T_tmp*a_v_l_matrix[idx]).item()
            if a_v_l_T is None:
                a_v_l_T = a_v_l_T_tmp
                a_v_l_loss = a_v_l_loss_tmp
            else:
                a_v_l_T = torch.cat([a_v_l_T, a_v_l_T_tmp], dim=0)
                a_v_l_loss = a_v_l_loss + a_v_l_loss_tmp


def cost_func(self,a, b, p=2, metric='cosine'):
        """ a, b in shape: (B, N, D) or (N, D)
        """ 
        assert type(a)==torch.Tensor and type(b)==torch.Tensor, 'inputs should be torch.Tensor'
        if metric=='euclidean' and p==1:
            return geomloss.utils.distances(a, b)
        elif metric=='euclidean' and p==2:
            return geomloss.utils.squared_distances(a, b)
        else:
            if a.dim() == 3:
                x_norm = a / a.norm(dim=2)[:, :, None]
                y_norm = b / b.norm(dim=2)[:, :, None]
                M = 1 - torch.bmm(x_norm, y_norm.transpose(-1, -2))
            elif a.dim() == 2:
                x_norm = a / a.norm(dim=1)[:, None]
                y_norm = b / b.norm(dim=1)[:, None]
                M = 1 - torch.mm(x_norm, y_norm.transpose(0, 1))
            M = pow(M, p)
            return M

def get_cost_matrix(self,A,B):
        norm_A = torch.norm(A, p=2, dim=2, keepdim=True)  # 形状为 (b, m, 1)
        norm_B = torch.norm(B, p=2, dim=2, keepdim=True)  # 形状为 (b, n, 1)

        # 计算余弦相似度
        cosine_sim = torch.matmul(A, B.transpose(1, 2))  # 形状为 (b, m, n)
        cosine_sim /= (norm_A * norm_B.transpose(1, 2))  # 归一化计算

        # 计算代价矩阵，代价 = 1 - 余弦相似度
        cost_matrix = 1 - cosine_sim
        return cost_matrix

def ot_loss(self,a,b,p=2,metric='cosine',entreg = .1):
        OTLoss = geomloss.SamplesLoss(
            loss='sinkhorn', p=p,
            blur=entreg**(1/p), backend='tensorized')
            # entreg**(1/p) cost=lambda a, b: self.cost_func(a, b, p=p, metric=metric),
        pw = OTLoss(a,b)
        return pw

'''
特征抽取阶段：
是否使用adampool方法 2种
otloss阶段：
三种loss的更换，以及使用T矩阵来增强音频和视频模态的方法 6种
融合阶段：
模态内先拼接再融合，先融合再拼接，2种
信息瓶颈阶段：
只对融合模块做信息瓶颈，和同时对单模态和融合做信息瓶颈 2种
'''

class fusion(nn.Module):
    def __init__(self, dim,beta):
        super().__init__()
        self.d_l = dim
        self.beta = beta
        # build encoder
        self.encoder = nn.Sequential(nn.Linear(self.d_l, 1024),
                                   #  nn.ReLU(inplace=True),
                                    # nn.Linear(1024, 1024),
                                     nn.ReLU(inplace=True) )  
        self.fc_mu  = nn.Linear(1024, self.d_l) 
        self.fc_std = nn.Linear(1024, self.d_l)
        # build decoder
        self.decoder = nn.Linear(self.d_l, 1)
      #  self.fusion1 = graph_fusion(self.d_l, self.d_l)
        self.fusion1 = concat(self.d_l, self.d_l)
       # self.fusion1 = tensor(self.d_l, self.d_l)
       # self.fusion1 = addition(self.d_l, self.d_l)
       # self.fusion1 = multiplication(self.d_l, self.d_l)
      #  self.fusion1 = low_rank(self.d_l, self.d_l)

    def encode(self, x):
        """
        x : [batch_size,784]
        """
        x = self.encoder(x)
        return self.fc_mu(x), F.softplus(self.fc_std(x)-5, beta=1)

    def decode(self, z):
        return self.decoder(z)
    
    def reparameterise(self, mu, std):
        """
        mu : [batch_size,z_dim]
        std : [batch_size,z_dim]        
        """        
        # get epsilon from standard normal
        eps = torch.randn_like(std)
        return mu + std*eps

    def loss_function(self, y_pred, y, mu, std):
        loss_fct = L1Loss()
        CE = loss_fct(y_pred.view(-1,), y.view(-1,))
        KL = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2*std.log() - 1)
        return (self.beta*KL + CE) 

    def forward(self,x,label_ids):
        outputf = self.fusion1(x)
        mu, std = self.encode(outputf)
        z = self.reparameterise(mu, std)
        output =  self.decode(z)
        loss = self.loss_function(output, label_ids, mu, std)
        return output, loss


    def test(self,x):
        outputf = self.fusion1(x)
        mu, std = self.encode(outputf)
        z = self.reparameterise(mu, std)
        output =  self.decode(z)
        return output


class concat(nn.Module):
    def __init__(self, in_size, output_dim,hidden = 50, dropout=0.5):
        super(concat, self).__init__()
        self.linear_1 = nn.Linear(in_size, output_dim)

    def forward(self, x):
        y_1 = torch.relu(self.linear_1(x))
        return y_1