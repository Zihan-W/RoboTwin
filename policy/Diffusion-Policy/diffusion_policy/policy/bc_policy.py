from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class BehaviorCloningPolicy(nn.Module):
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
        
        self.horizon = horizon
        # MLP网络结构 - 修改为处理单时间步特征
        # 输入维度调整为单个时间步的特征维度
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
        # 预测动作
        naction_pred = self.mlp(nobs_features)  # [B, action_dim * n_action_steps]
        
        # 重塑维度
        B_T, D = naction_pred.shape
        B = B_T // self.n_obs_steps
        naction_pred = naction_pred.view(B, self.n_obs_steps, self.n_action_steps, self.action_dim)
        
        # 合并观测步和动作步维度
        # 假设每个观测步预测对应的n_action_steps个动作
        # 最终取最后一个观测步的预测结果
        naction_pred = naction_pred[:, -1]  # [B, n_action_steps, action_dim]
        
        return naction_pred

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        '''
        Input shape: batch_size, horzion, channels, H, W
        '''

        def safe_normalize(normalizer, data):
            result = {}
            for key, value in data.items():
                if key in normalizer.params_dict:
                    result[key] = normalizer[key].normalize(value)
                else:
                    result[key] = value  # 原样返回
            return result

        # nobs = safe_normalize(self.normalizer, obs_dict)
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs) # batch_size * n_obs_steps, obs_feature_dim
        
        # 获取预测动作
        naction_pred = self.forward(nobs_features)  # batch_size, n_action_steps, action_dim
        
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
        
        # reshape
        this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))  # batch_size * n_obs_steps, channles, H, W
        nobs_features = self.obs_encoder(this_nobs) # batch_size * n_obs_steps, obs_feature_dim
        
        # 获取预测动作
        pred_actions = self.forward(nobs_features)  # 现在是torch.Size([64, 14])，应该为torch.Size([32, 4, 14])，即batch_size, n_action_steps, action_dim
        
        # 计算损失 (仅计算前n_action_steps步)
        gt_actions = nactions[:, :self.n_action_steps]
        loss = F.mse_loss(pred_actions, gt_actions)
        
        return loss

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())