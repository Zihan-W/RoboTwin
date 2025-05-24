from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super().__init__()
        # 第一个线性层及归一化
        self.linear1 = nn.Linear(in_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        
        # 第二个线性层及归一化
        self.linear2 = nn.Linear(out_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)
        
        # 正则化和激活
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

        # 快捷连接处理维度匹配
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features)
            )

    def forward(self, x):
        # 残差连接
        residual = self.shortcut(x)
        
        # 主路径处理
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.norm2(out)
        
        # 合并残差
        out += residual
        out = self.activation(out)
        out = self.dropout(out)
        return out
    
class BehaviorCloningPolicy(nn.Module):
    def __init__(self, 
            shape_meta: dict,
            obs_encoder: MultiImageObsEncoder,
            horizon,
            n_action_steps: int,
            n_obs_steps: int,
            hidden_dim: int = 512,
            num_layers: int = 4,
            dropout: float = 0.1,
            **kwargs):
        super().__init__()

        self.is_disturb = False

        # 解析输入输出维度
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        self.action_dim = action_shape[0]
        
        # 观测编码器
        self.obs_encoder = obs_encoder
        self.obs_feature_dim = obs_encoder.output_shape()[0]
        
        # 时间步参数
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        
        self.horizon = horizon
        # MLP网络结构 - 修改为处理单时间步特征
        # 输入维度调整为单个时间步的特征维度
        input_dim = self.obs_feature_dim * self.n_obs_steps
        output_dim = self.action_dim * self.horizon  # 输出维度扩大n_action_steps倍
    
        # # MLP Type 1
        # layers = []
        # current_dim = input_dim
        # for _ in range(num_layers - 1):
        #     layers.append(nn.Linear(current_dim, hidden_dim))
        #     layers.append(nn.ReLU())
        #     # layers.append(nn.Dropout(dropout))
        #     current_dim = hidden_dim
        # # 最终输出层调整为n_action_steps倍的维度
        # layers.append(nn.Linear(hidden_dim, output_dim))
        # self.mlp = nn.Sequential(*layers)
        
        # MLP Type 2
        layers = []
        drop_rates = [0.05, 0.1, 0.05]
        dims = [input_dim, 1024, 512, 256]  # 逐步降维
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),  # 添加BN层
                nn.ReLU(),
                nn.Dropout(drop_rates[i])
            ])
        
        # 最终输出层调整为n_action_steps倍的维度
        layers.append(nn.Linear(dims[-1], output_dim))
        self.mlp = nn.Sequential(*layers)

        # # Residual Network
        # layers = []
        # current_dim = input_dim
        # for _ in range(num_layers - 1):  # 最后一个层为输出层
        #     layers.append(ResidualBlock(current_dim, hidden_dim, dropout))
        #     current_dim = hidden_dim
        # layers.append(nn.Linear(current_dim, output_dim))
        # self.mlp = nn.Sequential(*layers)

        # # positional embedding for transformer
        # self.pos_embed = nn.Parameter(torch.randn(n_obs_steps, hidden_dim))
        # # transform obs feature dim to hidden dim
        # self.obs_proj = nn.Linear(self.obs_feature_dim, hidden_dim)
        # # transformer encoder
        # encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*2, dropout=dropout)
        # self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        # # 最终输出层，用于预测动作序列
        # self.output_layer = nn.Linear(hidden_dim, self.action_dim * self.n_action_steps)
        
        # 标准化器
        self.normalizer = LinearNormalizer()

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        '''
        Input shape: batch_size, horzion, channels, H, W
        '''
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(obs_dict.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs) # batch_size * n_obs_steps, obs_feature_dim
        # MLP
        nobs_features = nobs_features.view(B, self.n_obs_steps * self.obs_feature_dim)

        # 获取预测动作
        naction_pred = self.mlp(nobs_features)  # batch_size, n_action_steps, action_dim
        naction_pred = naction_pred.view(B, T, Da)   # batch_size, n_action_steps, action_dim

        # # Transformer encoding
        # nobs_features = nobs_features.view(B, self.n_obs_steps, self.obs_feature_dim)  # [B, To, Do]
        # x = self.obs_proj(nobs_features) + self.pos_embed[None, :, :]  # [B, T, D]
        # x = x.transpose(0, 1)  # [T, B, D]  <- transformer expects this shape
        # x = self.transformer(x)  # [T, B, D]

        # x = x[-1]  # 取最后一步的表示作为全局状态编码 [B, D]
        # naction_pred = self.output_layer(x)  # [B, n_action_steps * action_dim]
        # naction_pred = naction_pred.view(B, self.n_action_steps, self.action_dim)

        # 反标准化
        action_pred = self.normalizer['action'].unnormalize(naction_pred)   # # batch_size, n_action_steps, action_dim
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
    
        return {
            'action': action,
            'action_pred': action_pred   # 返回完整序列
        }

    def compute_loss(self, batch):
        # 标准化输入
        nobs = self.normalizer.normalize(batch['obs'])  # 正则化, batch_size, horzion, channels, H, W
        nactions = self.normalizer['action'].normalize(batch['action']) # 正则化, batch_size, horzion, action_dim

        if self.is_disturb:
            # add noise on observation
            nobs = {k: self.perturb_obs(v, k) for k, v in nobs.items()}
        
        # reshape
        B = nactions.shape[0]

        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))  # batch_size * n_obs_steps, channles, H, W
        nobs_features = self.obs_encoder(this_nobs) # batch_size * n_obs_steps, obs_feature_dim
        # MLP
        nobs_features = nobs_features.view(B, self.n_obs_steps * self.obs_feature_dim)  # [B, n_obs_steps * obs_feature_dim]
        
        # 获取预测动作
        pred_actions = self.mlp(nobs_features)  # batch_size * n_obs_steps，  n_action_steps, action_dim
        pred_actions = pred_actions.view(B, self.horizon, self.action_dim)
        
        # # Transformer 编码
        # nobs_features = nobs_features.view(B, self.n_obs_steps, self.obs_feature_dim)  # [B, To, Do]
        # x = self.obs_proj(nobs_features) + self.pos_embed[None, :, :]  # [B, T, D]
        # x = x.transpose(0, 1)  # [T, B, D]
        # x = self.transformer(x)  # [T, B, D]

        # x = x[-1]  # 使用最后时间步的全局状态表示 [B, D]
        # pred_actions = self.output_layer(x).view(B, self.n_action_steps, self.action_dim)  # [B, n_action_steps, action_dim]

        gt_actions = nactions
        if self.is_disturb:
            # add noise on gt_actions
            gt_actions = gt_actions + torch.randn_like(gt_actions) * 0.01
        
        loss = F.mse_loss(pred_actions, gt_actions)
        
        return loss


    def perturb_obs(self, x, key):
        if key in ['head_cam', 'right_cam']:
            return (x + torch.randn_like(x) * 0.01).clamp(0, 1)
        else:
            return (x + torch.randn_like(x) * 0.01).clamp(0, 0.5)
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())