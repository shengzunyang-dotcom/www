import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertPreTrainedModel
# from modules.transformer import TransformerEncoder
from transformers import AutoModel,AutoConfig,T5EncoderModel
import numpy as np
from global_configs_mosi import *
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
import ot, geomloss
class T5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.d_l*6 ,args.d_l*6)
        self.dropout = nn.Dropout(p=args.classifier_dropout)
        self.out_proj = nn.Linear(args.d_l*6, args.ds_label_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

'''KL divergence between two normal distributions'''
def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5

class GLoMo_BertModel(nn.Module):
    def __init__(self, args):
        super(GLoMo_BertModel,self).__init__()
        config = AutoConfig.from_pretrained(BERT_PRETRAIN_PATH)
        config.output_hidden_states = True  # 关键设置
        self.bert_encoder = AutoModel.from_pretrained(BERT_PRETRAIN_PATH,config=config)
        self.d_l = args.d_l
        self.gran_t = args.gran_t
        self.linear1 = nn.Linear(in_features=args.TEXT_DIM, out_features=self.d_l)
        self.proj_l = nn.Conv1d(args.TEXT_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgmaxpooling_t = nn.AdaptiveMaxPool1d(self.gran_t)

    def forward(self,input_ids,attention_mask=None):
        text = self.bert_encoder(input_ids = input_ids, attention_mask = attention_mask)
        last_sequence_output = text[0]# b*l*d
        global_features = self.linear1(last_sequence_output)
        outputs = last_sequence_output.transpose(1,2)
        outputs_t = self.proj_l(outputs).transpose(1,2)
        return global_features, outputs_t
    
class GLoMo_T5Model(nn.Module):
    def __init__(self, args):
        super(GLoMo_T5Model,self).__init__()
        self.bert_encoder = T5EncoderModel.from_pretrained(T5_PRETRAIN_PATH)
        self.freeze_params(self.bert_encoder)
        self.d_l = args.d_l
        self.gran_t = args.gran_t
        self.linear1 = nn.Linear(in_features=4096, out_features=self.d_l)
        self.proj_l = nn.Conv1d(4096, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)

    def freeze_params(self, model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False

    def forward(self,input_ids,attention_mask=None):
        text = self.bert_encoder(input_ids = input_ids, attention_mask = attention_mask)
        last_sequence_output = text[0]# b*l*d
        global_features = self.linear1(last_sequence_output)
        outputs = last_sequence_output.transpose(1,2)
        outputs_t = self.proj_l(outputs).transpose(1,2)
        return global_features, outputs_t
    

class Audio_Video_network(nn.Module): 
    def __init__(self, modality_dim, grans, args = None):
        super(Audio_Video_network, self).__init__()
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.d_l = args.d_l
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_dropout = args.attn_dropout
        self.modality_dim = modality_dim # for 
        self.grans = grans
        self.projs = nn.Conv1d(self.modality_dim, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        # self.avgmaxpoolings_c = nn.AdaptiveMaxPool1d(1)
        self.avgmaxpoolings_f = nn.AdaptiveMaxPool1d(self.grans)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_l, nhead=self.num_heads)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1) # num_layers 
        self.encoder = nn.Linear(in_features=self.modality_dim,out_features=self.d_l)   

    def forward(self,feas,mask):

        # the modality dimension can not be divided by num_heads, so first use Conv1d to change the dimensions
        # get local features
        fine_output = self.projs(feas.transpose(1,2)).transpose(1,2) 
        # fine_output = self.avgmaxpoolings_f(fine_output.transpose(1,2)).transpose(1,2)
        # get global features
        coarsed = self.encoder(feas)
        
        return coarsed, fine_output

class GatingFunction(nn.Module):
    def __init__(self, d_l):
        super().__init__()
        self.d_l = d_l
        self.mlp_mha = nn.Sequential(
            nn.Linear(self.d_l*2,self.d_l*2),
            nn.ReLU(),
            nn.Linear(self.d_l*2,1)
        ) 
        self.mlp_ffn = nn.Sequential(
            nn.Linear(self.d_l*2,self.d_l*2),
            nn.ReLU(),
            nn.Linear(self.d_l*2,1)
        ) 
        self.tanh = nn.Tanh()
    def forward(self,x_1,x_2,mask=None):
        # x_g,x_l: l * b * d
        x = torch.cat([x_1[0],x_2[0]],dim = 1)
        x_mha = self.mlp_mha(x).squeeze()
        x_ffn = self.mlp_ffn(x).squeeze()
        g_mha = self.tanh(x_mha)
        g_ffn = self.tanh(x_ffn)
        return g_mha,g_ffn


class GatedFusionBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_l = args.d_l
        self.num_heads = args.num_heads
        self.mha = nn.MultiheadAttention(embed_dim = self.d_l,num_heads = self.num_heads)
        self.mhsa = nn.MultiheadAttention(embed_dim = self.d_l,num_heads = self.num_heads)
        self.ffn1 = nn.Sequential(
            nn.Linear(self.d_l, self.d_l*2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.d_l*2, self.d_l)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(self.d_l, self.d_l*2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.d_l*2, self.d_l)
        )
        self.ln = nn.LayerNorm(self.d_l)
        self.gate = GatingFunction(self.d_l)

    def forward(self,x_1,x_2,mask1=None,mask2=None):
        x_1 = self.ln(x_1)
        x_2 = self.ln(x_2)
        g_mha,g_ffn = self.gate(x_1,x_2,mask1)
        if mask1 is None:
            x_mha,_ = self.mha(x_1,x_2,x_2)
        else:
            x_mha,_ = self.mha(x_1,x_2,x_2,mask1)
        z1 = torch.einsum('lbd,b->lbd', x_mha, g_mha) + x_1
        x_ffn = self.ffn1(self.ln(z1))
        z2 = torch.einsum('lbd,b->lbd', x_ffn, g_ffn) + z1
        z2 = self.ln(z2)
        if mask2 is None:
            x_mhsa,_ = self.mhsa(z2,z2,z2)
        else:
            x_mhsa,_ = self.mhsa(z2,z2,z2,mask2)
        x_mhsa += z2
        output = self.ffn2(self.ln(x_mhsa)) + x_mhsa
        return output

class GatedFusionModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = args.layers
        self.encoder = nn.ModuleList([])
        for layer in range(self.layers):
            new_layer = GatedFusionBlock(args)
            self.encoder.append(new_layer)
    def forward(self,x_1,x_2,mask1=None,mask2=None):
        for layer in self.encoder:
                x_1 = layer(x_1,x_2,mask1,mask2)
        return x_1

'''Prototypical Information Bottleneck'''
class PIB(nn.Module):
    def __init__(self,
                 x_dim,
                 z_dim = 256,
                 beta = 1e-2,
                 sample_num = 50,
                 topk = 256,
                 num_classes = 4):
        super(PIB, self).__init__()

        self.beta = beta
        self.sample_num = sample_num
        self.topk = topk
        self.num_classes = num_classes
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, z_dim*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim*2, z_dim*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim*2, z_dim),
        )

        # design proxies for histology images
        # self.proxies = nn.Parameter(torch.empty([num_classes, z_dim*2]))
        # torch.nn.init.xavier_uniform_(self.proxies, gain=1.0)
        self.decoder_logits = nn.Linear(z_dim, 1)

    def gaussian_noise(self, samples, K):
        
        return torch.normal(torch.zeros(*samples, K), torch.ones(*samples, K)).cuda()

    def encoder_result(self, x):

        encoder_output = self.encoder(x)

        return encoder_output

    def encoder_proxies(self,proxies):
        # mu_proxy = self.proxies[:, :self.z_dim]
        mu_proxy = proxies[:, :self.z_dim]
        sigma_proxy = torch.nn.functional.softplus(proxies[:, self.z_dim:]) # Make sigma always positive

        return mu_proxy, sigma_proxy
    
    def getproxyindex(self,label_ids):
        return  [torch.round(torch.clamp(ids+3,min = 0.0,max = 6.0)).long() for ids in label_ids]

    def forward(self,x,proxies,label_ids = None):
        #feature number
        feature_num = x.shape[1]

        # get z from encoder
        z = self.encoder_result(x)

        # get mu and sigma from proxies
        mu_proxy, sigma_proxy = self.encoder_proxies(proxies)

        # sample
        eps_proxy = self.gaussian_noise(samples=([self.num_classes, self.sample_num]), K=self.z_dim)
        z_proxy_sample = mu_proxy.unsqueeze(dim=1) + sigma_proxy.unsqueeze(dim=1) * eps_proxy
        z_proxy = torch.mean(z_proxy_sample, dim=1)

        # get attention maps
        z_norm = F.normalize(z,dim=2) # normalize z in the feature dimension
        z_proxy_norm = torch.unsqueeze(F.normalize(z_proxy),dim=0)

        att = torch.matmul(z_norm, torch.transpose(z_proxy_norm, 1, 2))

        if label_ids is None:
            '''validating and testing'''

            # get the proxy with the highest attention map
            att_unbind_proxy = torch.cat(torch.unbind(att, dim=1), dim=1)
            _, att_topk_proxy_idx = torch.topk(att_unbind_proxy, self.topk, dim=1)
            att_topk_proxy_idx = att_topk_proxy_idx % (self.num_classes)

            # get the positive proxy index
            positive_proxy_idx,_ = torch.mode(att_topk_proxy_idx, dim=1)
            positive_proxy_idx = positive_proxy_idx.unsqueeze(1).repeat(1, self.z_dim).unsqueeze(dim=1)#batch_size, index,z_dim

            proxy_loss = None

        else:
            '''training'''
            # get proxy_index for each sample
            proxy_indices = self.getproxyindex(label_ids)
            proxy_indices = torch.tensor(proxy_indices).long().cuda()
            mask = torch.zeros_like(att, dtype=torch.bool).cuda()
            mask[torch.arange(att.size(0)), :, proxy_indices] = True
            # get att_positive for each sample
            att_positive = torch.masked_select(att, mask).view(att.size(0), att.size(1), 1)
            # get att_negative for each sample
            att_negative = torch.masked_select(att, ~mask).view(att.size(0), att.size(1), -1)

            # calculate proxy loss
            att_topk_positive, att_topk_idx_positive = torch.topk(att_positive.squeeze(dim=2), self.topk, dim=1)
            att_topk_negative, _ = torch.topk(att_negative, self.topk, dim=1)
            att_positive_mean = torch.mean(att_topk_positive, dim=1)
            att_negative_mean = torch.mean(torch.mean(att_topk_negative, dim=1),dim=1)
            # proxy_loss = -(att_positive_mean-att_negative_mean).mean()
            proxy_loss = (att_negative_mean-att_positive_mean).mean()

            positive_proxy_idx = proxy_indices.unsqueeze(1).repeat(1, self.z_dim).unsqueeze(dim=1)

        # Gather mu_proxy and sigma_proxy for each sample
        mu_proxy_repeat = mu_proxy.repeat(x.shape[0], 1, 1)  # batch_size, num_classes*2, z_dim
        sigma_proxy_repeat = sigma_proxy.repeat(x.shape[0], 1, 1)
        mu_topk = torch.gather(mu_proxy_repeat, 1, positive_proxy_idx).squeeze(dim=1)
        sigma_topk = torch.gather(sigma_proxy_repeat, 1, positive_proxy_idx).squeeze(dim=1)

        att_unbind = torch.cat(torch.unbind(att, dim=2), dim=1)
        # get topk z features from attention maps
        att_topk, att_topk_idx = torch.topk(att_unbind, self.topk, dim=1)
        att_topk_idx = att_topk_idx % feature_num
        # get topk z features from z
        z_topk = torch.gather(z, 1, att_topk_idx.unsqueeze(dim=2).repeat(1, 1, self.z_dim))
        z_temp = z.transpose(0,1)
        z_temp  = z_temp[0].unsqueeze(1)
        z_topk = torch.cat([z_temp,z_topk],dim = 1)
        decoder_logits_proxy = torch.mean(self.decoder_logits(z_proxy_sample), dim=1)


        return decoder_logits_proxy,mu_proxy, sigma_proxy, z_topk, mu_topk, sigma_topk, proxy_loss


class GLOTModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.d_l = args.d_l
        self.lambd = args.lambd
        self.num_classes = args.num_classes
        self.topk = args.topk
        self.sample_num = args.sample_num
        self.alpha = args.alpha
        self.beta = args.beta
        self.audio_network = Audio_Video_network(args.ACOUSTIC_DIM, args.gran_a, args)
        self.video_network = Audio_Video_network(args.VISUAL_DIM, args.gran_v, args)
        if args.dsbert == 'bert':
            self.bert = GLoMo_BertModel(args)
        elif args.dsbert == 'T5':
            self.bert = GLoMo_T5Model(args)
        self.t_global_encoder = GatedFusionModule(args)
        self.t_local_encoder = GatedFusionModule(args)
        self.a_global_encoder = GatedFusionModule(args)
        self.a_local_encoder = GatedFusionModule(args)
        self.v_global_encoder = GatedFusionModule(args)
        self.v_local_encoder = GatedFusionModule(args)

        self.ot_loss = geomloss.SamplesLoss(
            loss='sinkhorn', p=args.p,
            blur=args.entreg**(1/args.p), backend='tensorized')
        self.bn = nn.BatchNorm1d(self.d_l*3, affine=False)
        self.PIB_g = PIB(self.d_l*3, self.d_l*3, num_classes=self.num_classes,
                                     topk=self.topk, sample_num=args.sample_num)
        self.PIB_l = PIB(self.d_l*3, self.d_l*3, num_classes=self.num_classes,
                                      topk=self.topk,sample_num=args.sample_num)
        self.classifier = T5ClassificationHead(args)

        self.abl_mode = args.abl_mode

        self.mseloss = MSELoss()
        self.proxies = nn.Parameter(torch.empty([self.num_classes, self.d_l*3*2]))
        torch.nn.init.xavier_uniform_(self.proxies, gain=1.0)
        self.mse = nn.MSELoss()
        self.w = nn.Parameter(torch.ones(4))

        self.alpha1 = args.alpha1
        self.alpha2 = args.alpha2
        
    def get_KL_loss(self, mu, std):
        '''
        :param mu: [batch_size, dimZ]
        :param std: [batch_size, dimZ]
        :return:
        '''
        # KL divergence between prior and posterior
        prior_z_distr = torch.zeros_like(mu), torch.ones_like(std)
        encoder_z_distr = mu, std

        I_zx_bound = torch.mean(KL_between_normals(encoder_z_distr, prior_z_distr))

        return torch.mean(I_zx_bound)
    
    def combine_features_with_mask(self,text_feats, audio_feats, video_feats, text_mask, audio_mask, video_mask):
        """
        拼接三个特征矩阵并保留有效向量，最后统一序列长度
        Args:
            text_feats:  文本特征 [batch_size, text_seq_len, feat_dim]
            audio_feats: 音频特征 [batch_size, audio_seq_len, feat_dim]
            video_feats: 视频特征 [batch_size, video_seq_len, feat_dim]
            text_mask:   文本mask [batch_size, text_seq_len] (True=无效, False=有效)
            audio_mask:  音频mask [batch_size, audio_seq_len]
            video_mask:  视频mask [batch_size, video_seq_len]
        Returns:
            combined_feats: 统一长度后的拼接特征 [batch_size, max_len, feat_dim]
            combined_masks: 统一长度后的掩码 [batch_size, max_len] (True=无效, False=有效)
        """
        batch_size = text_feats.size(0)
        feat_dim = text_feats.size(-1)
        combined_feats_list = []
        original_lengths = []  # 记录每个样本原始有效长度
        
        # 第一步：拼接有效特征
        for i in range(batch_size):
            # 提取当前样本的特征和mask
            text_i = text_feats[i]
            audio_i = audio_feats[i]
            video_i = video_feats[i]
            
            text_mask_i = text_mask[i]
            audio_mask_i = audio_mask[i]
            video_mask_i = video_mask[i]

            # 获取有效向量索引（反转mask）
            text_valid_idx = ~text_mask_i
            audio_valid_idx = ~audio_mask_i
            video_valid_idx = ~video_mask_i

            # 选择有效特征向量
            text_valid = text_i[text_valid_idx]
            audio_valid = audio_i[audio_valid_idx]
            video_valid = video_i[video_valid_idx]

            # 沿序列维度拼接有效特征
            combined_i = torch.cat((text_valid, audio_valid, video_valid), dim=0)
            combined_feats_list.append(combined_i)
            original_lengths.append(combined_i.size(0))
        
        # 第二步：确定最大长度
        # max_len = max(original_lengths)
        max_len = 100
        
        # 第三步：统一序列长度
        padded_feats_list = []
        padded_masks_list = []
        
        for i, feat in enumerate(combined_feats_list):
            curr_len = original_lengths[i]
            
            # 处理超长序列：截断
            if curr_len > max_len:
                feat = feat[:max_len]  # 截断超长部分
                mask = torch.zeros(max_len, dtype=torch.bool)  # 全部有效
            # 处理过短序列：补0
            else:
                # 在序列末尾补0
                pad_size = max_len - curr_len
                feat = F.pad(feat, (0, 0, 0, pad_size), value=0)  # (左, 右, 上, 下)
                
                # 创建掩码：原始部分有效，补0部分无效
                mask = torch.cat([
                    torch.zeros(curr_len, dtype=torch.bool),   # 原始有效部分
                    torch.ones(pad_size, dtype=torch.bool)    # 补0部分
                ])
            
            padded_feats_list.append(feat)
            padded_masks_list.append(mask)
        
        # 转换为张量 [batch_size, max_len, feat_dim]
        combined_feats = torch.stack(padded_feats_list).to(DEVICE)
        combined_masks = torch.stack(padded_masks_list).to(DEVICE)
        
        return combined_feats, combined_masks


    def forward(self, input_ids, attention_mask, text_mask, visual, visual_len, visual_mask, audio, audio_len, audio_mask, label_ids=None,epoch = None):
        # get global and local features
        coarsed_t, finegrained_t = self.bert(input_ids,attention_mask)
        coarsed_a, finegrained_a = self.audio_network(audio,audio_mask)
        coarsed_v, finegrained_v = self.video_network(visual,visual_mask)

        #global ot loss
        a_t_g_loss = self.ot_loss(coarsed_a.contiguous(),coarsed_t.contiguous())
        v_t_g_loss = self.ot_loss(coarsed_v.contiguous(),coarsed_t.contiguous())
        a_v_g_loss = self.ot_loss(coarsed_a.contiguous(),coarsed_v.contiguous())
        g_loss = 0.01*(a_t_g_loss.mean().item() + v_t_g_loss.mean().item() + a_v_g_loss.mean().item())

        #local ot loss
        a_t_l_loss = self.ot_loss(finegrained_a.contiguous(),finegrained_t.contiguous())
        v_t_l_loss = self.ot_loss(finegrained_v.contiguous(),finegrained_t.contiguous())
        a_v_l_loss = self.ot_loss(finegrained_a.contiguous(),finegrained_v.contiguous())
        l_loss = 0.01*(a_t_l_loss.mean().item() + v_t_l_loss.mean().item() + a_v_l_loss.mean().item())

        # gate fusion
        # global_feature_t,global_feature_mask = self.combine_features_with_mask(coarsed_t,coarsed_a,coarsed_v,text_mask,audio_mask,visual_mask)
        # local_feature_t,local_feature_mask = self.combine_features_with_mask(finegrained_t,finegrained_a,finegrained_v,text_mask,audio_mask,visual_mask)
        coarsed_t, finegrained_t,coarsed_a, finegrained_a,coarsed_v, finegrained_v = coarsed_t.transpose(0,1), \
            finegrained_t.transpose(0,1),coarsed_a.transpose(0,1), finegrained_a.transpose(0,1),coarsed_v.transpose(0,1), finegrained_v.transpose(0,1)
        global_feature_t = torch.cat([coarsed_t,coarsed_a,coarsed_v],dim = 0)
        local_feature_t = torch.cat([finegrained_t,finegrained_a,finegrained_v],dim = 0)
        # global_feature_t = global_feature_t.transpose(0,1)
        # local_feature_t = local_feature_t.transpose(0,1)
        # global_feature = global_feature.permute(1,0,2)
        # local_feature = local_feature.permute(1,0,2)
        t_global_feature = self.t_global_encoder(coarsed_t,global_feature_t,mask2 = text_mask)  # l*b*d
        t_local_feature = self.t_local_encoder(finegrained_t,local_feature_t,mask2 =text_mask)  # l*b*d
        a_global_feature = self.a_global_encoder(coarsed_a,global_feature_t,mask2 =audio_mask)  # l*b*d
        a_local_feature = self.a_local_encoder(finegrained_a,local_feature_t,mask2 =audio_mask)  # l*b*d
        v_global_feature = self.v_global_encoder(coarsed_v,global_feature_t,mask2 =visual_mask)  # l*b*d
        v_local_feature = self.v_local_encoder(finegrained_v,local_feature_t,mask2 =visual_mask)  # l*b*d
        global_feature = torch.cat([t_global_feature,a_global_feature,v_global_feature],dim = 2)
        local_feature = torch.cat([t_local_feature,a_local_feature,v_local_feature],dim = 2)
        
        # information bottleneck 
        decoder_logits_proxy_g, mu_proxy_g, sigma_proxy_g, h_g, mu_topk_g, sigma_topk_g,proxy_loss_g = self.PIB_g(global_feature.transpose(0,1),self.proxies,label_ids)
        decoder_logits_proxy_l, mu_proxy_l, sigma_proxy_l, h_l, mu_topk_l, sigma_topk_l,proxy_loss_l = self.PIB_l(local_feature.transpose(0,1),self.proxies,label_ids)
        
        
        IB_loss = self.alpha * self.mseloss(decoder_logits_proxy_g.view(-1), torch.arange(-3,4,dtype=torch.float32).to(DEVICE)) + self.alpha * self.mseloss(decoder_logits_proxy_l.view(-1), torch.arange(-3,4,dtype=torch.float32).to(DEVICE)) \
                              + self.beta * self.get_KL_loss(mu_proxy_g, sigma_proxy_g) + self.beta * self.get_KL_loss(mu_proxy_l, sigma_proxy_l)
        if label_ids is not None:
            proxy_loss = (proxy_loss_g + proxy_loss_l)

        h_g = torch.mean(h_g,dim = 1)
        h_l = torch.mean(h_l,dim = 1)
        h = torch.cat([h_g,h_l],dim = 1)
        logits = self.classifier(h)

        gl_at_loss = a_t_l_loss + a_t_g_loss
        gl_vt_loss = v_t_l_loss + v_t_g_loss
        gl_va_loss = a_v_l_loss + a_v_g_loss
        
        if self.abl_mode == "TV":
            gl_loss = gl_at_loss + gl_va_loss
        elif self.abl_mode == "TA":
            gl_loss = gl_vt_loss + gl_va_loss
        elif self.abl_mode == "VA":
            gl_loss = gl_at_loss + gl_vt_loss
        else:
            gl_loss = gl_at_loss + gl_vt_loss + gl_va_loss

        gl_loss = gl_loss.mean().item()

        if label_ids is not None:
            task_loss = self.mse(logits.squeeze(), label_ids.view(-1))
            # loss_all = self.w[0] * task_loss + self.w[1] * (l_loss + g_loss) + self.w[2] * proxy_loss + self.w[3] * IB_loss
            loss_all = self.alpha1 * task_loss + self.alpha2 * gl_loss + self.w[2] * proxy_loss + self.w[3] * IB_loss

        # bottleneck
        if label_ids is not None:
            return logits.squeeze(), loss_all, self.w, proxy_loss, IB_loss
        else:
            return logits.squeeze(), IB_loss



'''
优化方法
把cls token加入到选出的特征当中去
减少各种线性层的层数
'''



