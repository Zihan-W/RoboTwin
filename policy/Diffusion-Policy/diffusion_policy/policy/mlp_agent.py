from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class MLPAgent(nn.Module):
    def __init__(self, 
            shape_meta: dict,
            obs_encoder: MultiImageObsEncoder,
            horizon,
            n_action_steps: int,
            n_obs_steps: int,
            hidden_dim: int = 512,
            num_layers: int = 3,
            dropout: float = 0.1,
            **kwargs):
        super().__init__()

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
        
        # 采样深度
        self.horizon = horizon

        # MLP网络结构
        input_dim = self.obs_feature_dim
        output_dim = self.action_dim * self.n_action_steps  # 输出维度扩大n_action_steps倍
        
        layers = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # 最终输出层调整为n_action_steps倍的维度
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
        # 标准化器
        self.normalizer = LinearNormalizer()

    def forward(self, nobs_features) -> torch.Tensor:
        '''
        Input shape: nobs_features: batch_size * n_obs_steps, obs_feature_dim
        Output shape: batch_size, n_action_steps, action_dim
        '''
        # 预测动作概率
        action_logits = self.mlp(nobs_features)  # [B, action_dim * n_action_steps]
        
        # 重塑维度
        B_T, D = action_logits.shape
        B = B_T // self.n_obs_steps
        action_logits = action_logits.view(B, self.n_obs_steps, self.n_action_steps, self.action_dim)
        action_logits = action_logits[:, -1]  # [B, n_action_steps, action_dim]
        
        # 转换为概率分布
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        '''
        Input shape: batch_size, horzion, channels, H, W
        '''
        # 处理输入的obs
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        
        # 使用encoder编码得到nobs的特征
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs) # batch_size * n_obs_steps, obs_feature_dim
        
        # 获取动作概率分布
        action_probs = self.forward(nobs_features)
        action_dist = torch.distributions.Categorical(action_probs)
        
        # 采样动作
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        # 反标准化
        action_pred = self.normalizer['action'].unnormalize(action)
        
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        return {
            'action': action,
            'action_pred': action_pred,
            'log_prob': log_prob
        }

    def compute_loss(self, batch):
        '''
        计算策略梯度损失
        '''
        # 标准化输入
        nobs = self.normalizer.normalize(batch['obs'])  # 正则化, batch_size, horzion, channels, H, W
        nactions = self.normalizer['action'].normalize(batch['action']) # 正则化, batch_size, horzion, action_dim
        rewards = batch['rewards']  # [B, T]
        dones = batch['dones']  # [B, T]
        
        # reshape观测数据
        this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))  # batch_size * n_obs_steps, channles, H, W
        nobs_features = self.obs_encoder(this_nobs) # batch_size * n_obs_steps, obs_feature_dim
        
        # 获取动作概率分布
        action_probs = self.forward(nobs_features)  # [B, n_action_steps, action_dim]
        action_dist = torch.distributions.Categorical(action_probs)
        
        # 计算动作的对数概率
        log_probs = action_dist.log_prob(nactions[:, :self.n_action_steps])  # [B, n_action_steps]
        
        # 计算折扣回报
        returns = torch.zeros_like(rewards)
        running_return = 0
        gamma = 0.99  # 折扣因子
        
        for t in reversed(range(rewards.shape[1])):
            running_return = rewards[:, t] + gamma * running_return * (1 - dones[:, t])
            returns[:, t] = running_return
        
        # 计算优势
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 标准化
        
        # 计算策略梯度损失
        policy_loss = -(log_probs * returns.unsqueeze(-1)).mean()
        
        # 计算熵正则化
        entropy_loss = -action_dist.entropy().mean()
        
        # 总损失
        total_loss = policy_loss - 0.01 * entropy_loss
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())        