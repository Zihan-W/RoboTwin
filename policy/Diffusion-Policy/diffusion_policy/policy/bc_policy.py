from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer

class BehaviorCloningPolicy(nn.Module):
    def __init__(self, 
            shape_meta: dict,
            obs_encoder: MultiImageObsEncoder,
            n_action_steps: int,
            n_obs_steps,
            horizon,
            hidden_dim: int = 32,
            **kwargs):
        super().__init__()

        # 解析输入输出维度
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        # 创建MLP模型 obs→action
        self.obs_encoder = obs_encoder
        self.action_predictor = nn.Sequential(
            nn.Linear(obs_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
          )  # 输出多步动作
        
        self.normalizer = LinearNormalizer()
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.horizon = horizon

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ 直接预测动作 """
        # 编码观测
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs) # nobs_features torch.Size([96, 526])
        nobs_features = nobs_features.reshape(B, T, -1)
        # 预测动作
        naction_pred = self.action_predictor(nobs_features)  # [B, action_dim]  naction_pred torch.Size([96, 14])
        action_pred = self.normalizer['action'].unnormalize(naction_pred)   # torch.Size([24, 8, 14])
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        result = {
            'action':action,
            'action_pred':action_pred
        }
        return result

    def compute_loss(self, batch):
        """ 简化的MSE损失 """
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, T, Do
        nobs_features = nobs_features.reshape(batch_size, horizon, -1)
       
        # 预测动作
        pred_actions = self.action_predictor(nobs_features)  # torch.Size([32, 4, 14])
        pred_actions = pred_actions.view(-1, self.n_action_steps, self.action_dim)  # [B, T, D_a] naction_pred torch.Size([32, 4, 14])

        gt_actions = batch['action'].view(pred_actions.shape)  # 对齐形状，batch_action [B, T, Da]
        
        # 计算MSE
        loss = F.mse_loss(pred_actions, gt_actions)
        return loss