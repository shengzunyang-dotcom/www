import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CrossModalTransformer(nn.Module):
    """跨模态Transformer: 使用Q从一个模态,K/V从另一个模态"""
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 构建多层Transformer
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
    def forward(self, query_modal, key_value_modal):
        """
        query_modal: (batch, seq_len_q, d_model) - 查询模态
        key_value_modal: (batch, seq_len_kv, d_model) - 键值模态
        """
        out = query_modal
        for layer in self.layers:
            out = layer(out, key_value_modal)
        return out


class MultimodalTransformer(nn.Module):
    """多模态Transformer模型"""
    def __init__(
        self,
        text_dim=768,
        visual_dim=2048,
        audio_dim=128,
        d_model=512,
        nhead=8,
        cross_modal_layers=2,
        fusion_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 各模态的投影层
        self.text_proj = nn.Linear(text_dim, d_model)
        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # 跨模态Transformer (两两交互)
        # Text-Visual
        self.text_visual_trans = CrossModalTransformer(
            d_model, nhead, cross_modal_layers, dim_feedforward, dropout
        )
        self.visual_text_trans = CrossModalTransformer(
            d_model, nhead, cross_modal_layers, dim_feedforward, dropout
        )
        
        # Text-Audio
        self.text_audio_trans = CrossModalTransformer(
            d_model, nhead, cross_modal_layers, dim_feedforward, dropout
        )
        self.audio_text_trans = CrossModalTransformer(
            d_model, nhead, cross_modal_layers, dim_feedforward, dropout
        )
        
        # Visual-Audio
        self.visual_audio_trans = CrossModalTransformer(
            d_model, nhead, cross_modal_layers, dim_feedforward, dropout
        )
        self.audio_visual_trans = CrossModalTransformer(
            d_model, nhead, cross_modal_layers, dim_feedforward, dropout
        )
        
        # 融合Transformer
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(
            fusion_layer,
            num_layers=fusion_layers
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, visual, audio):
        """
        text: (batch, seq_len_text, text_dim)
        visual: (batch, seq_len_visual, visual_dim)
        audio: (batch, seq_len_audio, audio_dim)
        """
        # 投影到统一维度
        text_emb = self.dropout(self.pos_encoder(self.text_proj(text)))
        visual_emb = self.dropout(self.pos_encoder(self.visual_proj(visual)))
        audio_emb = self.dropout(self.pos_encoder(self.audio_proj(audio)))
        
        # 两两跨模态交互
        # Text-Visual
        text_from_visual = self.text_visual_trans(text_emb, visual_emb)
        visual_from_text = self.visual_text_trans(visual_emb, text_emb)
        
        # Text-Audio
        text_from_audio = self.text_audio_trans(text_emb, audio_emb)
        audio_from_text = self.audio_text_trans(audio_emb, text_emb)
        
        # Visual-Audio
        visual_from_audio = self.visual_audio_trans(visual_emb, audio_emb)
        audio_from_visual = self.audio_visual_trans(audio_emb, visual_emb)
        
        # 融合各模态的增强表示
        text_enhanced = text_emb + text_from_visual + text_from_audio
        visual_enhanced = visual_emb + visual_from_text + visual_from_audio
        audio_enhanced = audio_emb + audio_from_text + audio_from_visual
        
        # 在序列维度拼接
        fused = torch.cat([text_enhanced, visual_enhanced, audio_enhanced], dim=1)
        
        # 最终融合Transformer
        output = self.fusion_transformer(fused)
        
        return output


def count_parameters(model):
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# 测试代码
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    
    # 模型配置
    config = {
        'text_dim': 768,
        'visual_dim': 768,
        'audio_dim': 768,
        'd_model': 768,
        'nhead': 8,
        'cross_modal_layers': 2,  # 跨模态Transformer层数
        'fusion_layers': 2,        # 融合Transformer层数
        'dim_feedforward': 768,
        'dropout': 0.1
    }
    
    # 创建模型
    model = MultimodalTransformer(**config)
    print("=" * 60)
    print("多模态Transformer模型")
    print("=" * 60)
    print(f"配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # 统计参数
    total_params, trainable_params = count_parameters(model)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"参数大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # 创建测试数据
    batch_size = 4
    text_seq_len = 20
    visual_seq_len = 10
    audio_seq_len = 50
    
    text = torch.randn(batch_size, text_seq_len, config['text_dim'])
    visual = torch.randn(batch_size, visual_seq_len, config['visual_dim'])
    audio = torch.randn(batch_size, audio_seq_len, config['audio_dim'])
    
    print("\n" + "=" * 60)
    print("测试模型前向传播")
    print("=" * 60)
    print(f"输入形状:")
    print(f"  Text:   {tuple(text.shape)}")
    print(f"  Visual: {tuple(visual.shape)}")
    print(f"  Audio:  {tuple(audio.shape)}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(text, visual, audio)
    
    print(f"\n输出形状: {tuple(output.shape)}")
    print(f"预期序列长度: {text_seq_len + visual_seq_len + audio_seq_len}")
    print(f"输出维度: {config['d_model']}")
    
    # 显示模型结构概览
    print("\n" + "=" * 60)
    print("模型结构概览")
    print("=" * 60)
    print(f"1. 模态投影层 (3个)")
    print(f"2. 跨模态Transformer (6个, 每个{config['cross_modal_layers']}层)")
    print(f"   - Text-Visual, Visual-Text")
    print(f"   - Text-Audio, Audio-Text")
    print(f"   - Visual-Audio, Audio-Visual")
    print(f"3. 融合Transformer ({config['fusion_layers']}层)")
    print("=" * 60)
    
    print("\n✓ 模型测试完成!")