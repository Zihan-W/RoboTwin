from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiStepActor(nn.Module):
    def __init__(self, input_dim, action_dim, pred_steps, hidden_dim=512):
        super().__init__()
        self.pred_steps = pred_steps
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * pred_steps),
            nn.Tanh()
        )
    
    def forward(self, obs_feature):
        batch_size = obs_feature.size(0)
        actions = self.net(obs_feature)
        return actions.view(batch_size, self.pred_steps, self.action_dim)

class MultiStepCritic(nn.Module):
    def __init__(self, input_dim, action_dim, pred_steps, hidden_dim=512):
        super().__init__()
        self.pred_steps = pred_steps
        self.action_dim = action_dim
        
        self.q1_net = nn.Sequential(
            nn.Linear(input_dim + action_dim * pred_steps, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        
        self.q2_net = nn.Sequential(
            nn.Linear(input_dim + action_dim * pred_steps, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, obs_feature, actions):
        batch_size = actions.shape[0]
        actions_flat = actions.view(batch_size, -1)  # 展平动作序列
        combined = torch.cat([obs_feature, actions_flat], dim=1)
        q1 = self.q1_net(combined)
        q2 = self.q2_net(combined)
        return q1, q2


class MultiStepTD3Agent(nn.Module):
    def __init__(self, 
            shape_meta: dict,
            lr_critic: float = 1e-3,
            lr_actor: float = 1e-4,
            obs_encoder: MultiImageObsEncoder = None,
            horizon: int = 10,
            n_action_steps: int = 5,
            n_obs_steps: int = 2,
            hidden_dim: int = 512,
            gamma: float = 0.99,
            tau: float = 0.005,
            policy_noise: float = 0.2,
            noise_clip: float = 0.5,
            policy_update_freq: int = 2,
            **kwargs):
        super().__init__()
        
        # 参数配置
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_update_freq = policy_update_freq
        self.n_obs_steps = n_obs_steps
        
        # 输入输出维度
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        self.action_dim = action_shape[0]
        self.n_action_steps = n_action_steps
        
        # 观测编码器
        self.obs_encoder = obs_encoder
        if obs_encoder is not None:
            self.obs_feature_dim = obs_encoder.output_shape()[0]
        else:
            # 如果没有图像编码器，直接使用状态维度
            state_shape = shape_meta['obs']['state']['shape']
            self.obs_feature_dim = state_shape[0]
        
        # 网络结构
        self.actor = MultiStepActor(
            input_dim=self.obs_feature_dim,
            action_dim=self.action_dim,
            pred_steps=n_action_steps,
            hidden_dim=hidden_dim
        ).to(device)
        
        self.actor_target = MultiStepActor(
            input_dim=self.obs_feature_dim,
            action_dim=self.action_dim,
            pred_steps=n_action_steps,
            hidden_dim=hidden_dim
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = MultiStepCritic(
            input_dim=self.obs_feature_dim,
            action_dim=self.action_dim,
            pred_steps=n_action_steps,
            hidden_dim=hidden_dim
        ).to(device)
        
        self.critic_target = MultiStepCritic(
            input_dim=self.obs_feature_dim,
            action_dim=self.action_dim,
            pred_steps=n_action_steps,
            hidden_dim=hidden_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 标准化器
        self.normalizer = LinearNormalizer()
        self.step_count = 0


    def update(self, batch):
        # 数据标准化和预处理
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['tcp_action'].normalize(batch['tcp_action']).to(device)
        nrewards = batch['reward'].to(device)
        next_nobs = self.normalizer.normalize(batch['next_obs'])
        dones = batch['done'].to(device)

        # 编码当前和下一时刻的观测特征
        def process_obs(x):
            # 将多个观测时间步在特征维度拼接
            # 原始形状 [batch_size, horizon, ...]
            x = x[:,:self.n_obs_steps,...] 
            
            # 对于状态观测
            if x.ndim == 3:  # [B, T, D]
                x = x.reshape(x.shape[0], -1)  # [B, T*D]
            
            # 对于图像观测（示例）
            elif x.ndim == 5:  # [B, T, C, H, W]
                x = x.permute(0,2,1,3,4)  # [B, C, T, H, W]
                x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])  # [B, C*T, H, W]
            
            return x
        
        obs_features = self.obs_encoder(dict_apply(nobs, process_obs)).to(device)
        next_obs_features = self.obs_encoder(dict_apply(next_nobs, process_obs)).to(device)

        with torch.no_grad():  # 目标网络计算
            # 生成带噪声的目标动作
            target_actions = self.actor_target(next_obs_features)
            noise = torch.randn_like(target_actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            target_actions = (target_actions + noise).clamp(-1, 1)
            
            # 计算目标Q值
            target_q1, target_q2 = self.critic_target(next_obs_features, target_actions)
            target_q = torch.min(target_q1, target_q2)
            import ipdb; ipdb.set_trace()
            target_q = nrewards + (1 - dones) * self.gamma * target_q

        # 计算当前Critic的预测值
        current_q1, current_q2 = self.critic(obs_features, nactions[:, :self.n_action_steps])
        
        # Critic损失计算
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Critic网络更新
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor网络周期性更新
        actor_loss = None
        if self.step_count % self.policy_update_freq == 0:
            pred_actions = self.actor(obs_features)
            q1, _ = self.critic(obs_features, pred_actions)
            actor_loss = -q1.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 目标网络软更新
            with torch.no_grad():
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.lerp_(param.data, self.tau)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.lerp_(param.data, self.tau)

        self.step_count += 1
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss else 0.0
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())