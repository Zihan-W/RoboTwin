import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from typing import TypeVar

device = torch.device("cuda:0")


# class Actor(nn.Module):
# 	def __init__(self, state_dim, action_dim, action_seq_dim, max_action, n_action_steps, phi=0.05):
# 		super(Actor, self).__init__()
# 		self.n_action_steps = n_action_steps
# 		self.l1 = nn.Linear(state_dim + action_seq_dim, 400)
# 		self.bn1 = nn.BatchNorm1d(400)
# 		self.l2 = nn.Linear(400, 300)
# 		self.bn2 = nn.BatchNorm1d(300)
# 		self.l3 = nn.Linear(300, action_seq_dim)
		
# 		self.max_action = max_action
# 		self.action_dim = action_dim
# 		self.phi = phi


# 	def forward(self, state, action_seq):
# 		batch_size = action_seq.shape[0]	# action_seq: [batch_size, n_action_steps, action_dim]
# 		action_seq_flat = action_seq.view(batch_size, -1)  # [batch_size, n_action_steps*action_dim]
# 		x = torch.cat([state, action_seq_flat], 1)
# 		a = F.relu(self.bn1(self.l1(x)))
# 		a = F.relu(self.bn2(self.l2(a)))
# 		a = self.phi * self.max_action * torch.tanh(self.l3(a))

# 		a = a.view(batch_size, self.n_action_steps, -1)  # [batch_size, n_steps, action_dim]
# 		return (a + action_seq).clamp(-self.max_action[0:self.action_dim], self.max_action[0:self.action_dim])

class Actor(nn.Module):
	def __init__(self, state_seq_dim, action_dim, action_seq_dim, max_action, horizon, phi=0.05):
		super().__init__()
		self.base = nn.Sequential(
			nn.Linear(state_seq_dim + action_seq_dim, 1024),
			nn.LayerNorm(1024),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.LayerNorm(512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.LayerNorm(256),
			nn.ReLU()
		)
		self.res_block1 = nn.Sequential(
			nn.Linear(256, 256),
			nn.LayerNorm(256),
			nn.ReLU()
		)
		self.res_block2 = nn.Sequential(
			nn.Linear(256, 256),
			nn.LayerNorm(256),
			nn.ReLU()
		)
		self.output = nn.Linear(256, action_seq_dim)
		self.horizon = horizon
		self.max_action = max_action
		self.action_dim = action_dim
		self.phi = phi

	def forward(self, state, action_seq_flat):
		repeat_batch_size = action_seq_flat.shape[0]
		x = torch.cat([state, action_seq_flat], 1)

		# Base network
		x = self.base(x)

		# Residual blocks
		residual = x
		x = self.res_block1(x) + residual
		residual = x
		x = self.res_block2(x) + residual

		# Output layer
		a = self.phi * self.max_action * torch.tanh(self.output(x))
		a = a.view(repeat_batch_size, self.horizon, self.action_dim)
		action_seq = action_seq_flat.view(repeat_batch_size, self.horizon, self.action_dim)
		return (a + action_seq).clamp(-self.max_action[0:self.action_dim], self.max_action[0:self.action_dim])


class Critic(nn.Module):
	def __init__(self, state_dim, action_seq_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_seq_dim, 512)
		self.l2 = nn.Linear(512, 512)
		self.l3 = nn.Linear(512, 256)
		self.l4 = nn.Linear(256, 1)

		self.l5 = nn.Linear(state_dim + action_seq_dim, 512)
		self.l6 = nn.Linear(512, 512)
		self.l7 = nn.Linear(512, 256)
		self.l8 = nn.Linear(256, 1)

	def forward(self, state, action_seq):
		action_seq_flat = action_seq.view(action_seq.shape[0], -1)
		q1 = F.relu(self.l1(torch.cat([state, action_seq_flat], 1)))
		q1 = F.relu(self.l2(q1)) + q1
		q1 = self.l3(q1)
		q1 = self.l4(q1)

		q2 = F.relu(self.l5(torch.cat([state, action_seq_flat], 1)))
		q2 = F.relu(self.l6(q2)) + q2
		q2 = self.l7(q2)
		q2 = self.l8(q2)
		return q1, q2

	def q1(self, state, action_seq):
		action_seq_flat = action_seq.view(action_seq.shape[0], -1)
		q1 = F.relu(self.l1(torch.cat([state, action_seq_flat], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
	def __init__(self, state_dim, action_seq_dim, latent_dim, max_action, horizon, device):
		super(VAE, self).__init__()
		self.e1 = nn.Linear(state_dim + action_seq_dim, 750)
		self.e2 = nn.Linear(750, 750)

		self.mean = nn.Linear(750, latent_dim)
		self.log_std = nn.Linear(750, latent_dim)

		self.d1 = nn.Linear(state_dim + latent_dim, 750)
		self.d2 = nn.Linear(750, 750)
		self.d3 = nn.Linear(750, action_seq_dim)

		self.max_action = max_action
		self.latent_dim = latent_dim
		self.device = device
		self.horizon = horizon


	def forward(self, state, action_seq):
		z = F.relu(self.e1(torch.cat([state, action_seq], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)
		
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		# When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.06,0.06)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		a = self.max_action * torch.tanh(self.d3(a))
		return a

class BCQ(nn.Module):
	def __init__(self,
			  shape_meta: dict,
			  obs_encoder: MultiImageObsEncoder,
			  horizon,
			  n_obs_steps: int,
			  n_action_steps: int,
			  discount=0.99,
			  tau=0.01,
			  lmbda=0.75,
			  lr_critic=1e-3,
			  lr_actor=1e-3,
			  phi=0.05,
			  **kwargs):
		super(BCQ, self).__init__()
		self.device = torch.device("cuda:0")
		action_shape = shape_meta['action']['shape']
		self.action_dim = action_shape[0]
		latent_dim = self.action_dim * 2
		self.n_action_steps = n_action_steps
		self.horizon = horizon
		self.action_seq_dim = self.action_dim * self.horizon  # 多步动作序列维度
		max_action = torch.tensor([2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 0.045], device=self.device)
		self.max_action = max_action.repeat(2 * self.horizon)

		self.obs_encoder = obs_encoder
		self.obs_feature_dim = obs_encoder.output_shape()[0]
		self.n_obs_steps = n_obs_steps
		self.state_dim = self.obs_feature_dim	# TODO:可能需要加上robot_state这几个维度
		self.state_seq_dim = n_obs_steps * self.state_dim

		self.actor = Actor(self.state_seq_dim, self.action_dim, self.action_seq_dim, self.max_action, self.horizon, phi).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

		self.critic = Critic(self.state_seq_dim, self.action_seq_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-4)

		self.vae = VAE(self.state_seq_dim, self.action_seq_dim, latent_dim, self.max_action, self.horizon, device).to(device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

		self.discount = discount
		self.tau = tau
		self.lmbda = lmbda

		self.normalizer = LinearNormalizer()

	def select_action(self, state):		
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
			action_seq = self.actor(state, self.vae.decode(state))	# # [batch, n_steps, action_dim]
			q1 = self.critic.q1(state, action_seq)
			ind = q1.argmax(0)
		return action_seq[ind, 0, :]  # 返回序列动作
		#return action[ind].cpu().data.numpy().flatten()


	def update(self, batch):
		iterations = 1
		batch_size = batch['reward'].shape[0]
		batch = dict_apply(batch, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)
		for it in range(iterations):
			with torch.no_grad():
				# Gsussian Noise
				# noise_level = 0.1  # 根据动作范围调整（如动作范围±2.7，噪声设为0.1~0.3）
				# batch['action'] += noise_level * torch.randn_like(batch['action'])

				nobs = self.normalizer.normalize(batch['obs'])  # 正则化, batch_size, horzion, channels, H, W
				next_nobs = self.normalizer.normalize(batch['next_obs'])
				# obs reshape
				this_nobs = dict_apply(nobs, 
						lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))  # batch_size * n_obs_steps, channles, H, W
				next_this_nobs = dict_apply(next_nobs, 
						lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))  # batch_size * n_obs_steps, channles, H, W
				nobs_features = self.obs_encoder(this_nobs) # [batch_size * n_obs_steps, obs_feature_dim]
				next_nobs_features = self.obs_encoder(next_this_nobs) # [batch_size * n_obs_steps, obs_feature_dim]
				nobs_features = nobs_features.view(batch_size, self.n_obs_steps * self.obs_feature_dim)  # 转换为 [batch_size, n_obs_steps * obs_feature_dim]
				next_nobs_features = next_nobs_features.view(batch_size, self.n_obs_steps * self.obs_feature_dim)  # 转换为 [batch_size, n_obs_steps * obs_feature_dim]

				nactions = self.normalizer['action'].normalize(batch['action']) # 正则化, [batch_size, horzion, action_dim]
				nactions_flat = nactions[:, :self.horizon, :]  # [batch_size, n_steps, action_dim]
				nactions_flat = nactions.view(batch_size, -1)  # [batch_size, n_steps*action_dim]
				
				# process rewards and dones
				done = batch['done']  # [batch_size, n_action_steps]
				done_cum = torch.cumsum(done, dim=1)  # [batch_size, n_action_steps]
				valid_mask = (done_cum == 0).float()  # [batch_size, n_action_steps]
				
				gamma_powers = torch.tensor([self.discount**k for k in range(self.horizon)], 
							device=self.device)  # [n_action_steps,]
				# rewards = nactions = self.normalizer.normalize(batch['reward'])
				discounted_rewards = batch['reward'] * gamma_powers.unsqueeze(0) * valid_mask  # [batch_size, n_action_steps]
				nrewards = discounted_rewards.sum(dim=1)  # [batch_size,]

				any_done = (done_cum[:, -1] > 0).float().unsqueeze(1)  # [batch_size, 1]

			# Sample replay buffer / batch
			# TODO：可能需要重新评估用到的数据
			state = nobs_features.to(device)
			next_state = next_nobs_features.to(device)
			action = nactions_flat.to(device)
			reward = nrewards.to(device)
			not_done = (1.0 - any_done).to(device)

			# Variational Auto-Encoder Training
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 1.0 * KL_loss

			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()

			# FID record
			with torch.no_grad():
				# 生成动作（与重构动作分开）
				batch_size = state.shape[0]
				latent_dim = self.vae.latent_dim
				sampled_z = torch.randn(batch_size, latent_dim).to(self.device)  # 正确：显式生成噪声
				generated_actions = self.vae.decode(state, sampled_z)
				
				# 转换为CPU numpy数组
				real_actions = nactions.cpu().numpy()           # [B, T, D]
				recon_actions = recon.detach().cpu().numpy()     # [B, T, D]
				gen_actions = generated_actions.detach().cpu().numpy()  # [B, T, D]
				
				# 计算FID（重构动作 vs 真实动作）
				fid_recon = self.calculate_fid(real_actions, recon_actions)
				# 计算FID（生成动作 vs 真实动作）
				fid_gen = self.calculate_fid(real_actions, gen_actions)

			# Critic Training
			with torch.no_grad():
				# Duplicate next state 10 times
				next_state = torch.repeat_interleave(next_state, 10, 0)

				next_flat_nsampled_actions = self.vae.decode(next_state)	# shape of result: [batch_size*repeat_num, horizon * action_dim]
				next_nsampled_actions = self.actor_target(next_state, next_flat_nsampled_actions)	# shape of result: [batch_size*repeat_num, horizon, action_dim]
				# Compute value of perturbed actions sampled from the VAE
				target_Q1, target_Q2 = self.critic_target(next_state, next_nsampled_actions)

				# Soft Clipped Double Q-learning 
				target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
				# Take max over each action sampled from the VAE
				target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)
				
				# calculate n_step Q-value
				target_Q = reward.unsqueeze(1) + (gamma_powers[-1] * self.discount) * not_done * target_Q
			current_Q1, current_Q2 = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
			self.critic_optimizer.step()

			# Pertubation Model / Action Training
			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)

			# Update through DPG
			actor_loss = -self.critic.q1(state, perturbed_actions).mean()
		 	 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
			self.actor_optimizer.step()

			# Update Target Networks 
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			return critic_loss.item(), actor_loss.item(), recon_loss.item(), KL_loss.item(), fid_gen.item(), fid_recon.item()
		
	def set_normalizer(self, normalizer: LinearNormalizer):
		self.normalizer.load_state_dict(normalizer.state_dict())

	def evaluate(self, batch):
		batch_size = batch['reward'].shape[0]
		with torch.no_grad():
			# process obs and action			
			nobs = self.normalizer.normalize(batch['obs'])
			next_nobs = self.normalizer.normalize(batch['next_obs'])
			
			this_nobs = dict_apply(nobs,
					lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
			next_this_nobs = dict_apply(next_nobs,
					lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
			
			nobs_features = self.obs_encoder(this_nobs).view(batch_size, -1)
			next_nobs_features = self.obs_encoder(next_this_nobs).view(batch_size, -1)
			
			nactions = self.normalizer['action'].normalize(batch['action'])
			nactions = nactions[:, :self.horizon, :]
			nactions_flat = nactions.view(batch_size, -1)
			# process rewards and dones
			done = batch['done']  # [batch_size, n_action_steps]
			done_cum = torch.cumsum(done, dim=1)  # [batch_size, n_action_steps]
			valid_mask = (done_cum == 0).float()  # [batch_size, n_action_steps]
			
			gamma_powers = torch.tensor([self.discount**k for k in range(self.horizon)], 
						device=self.device)  # [n_action_steps,]
			discounted_rewards = batch['reward'] * gamma_powers.unsqueeze(0) * valid_mask  # [batch_size, n_action_steps]
			nrewards = discounted_rewards.sum(dim=1)  # [batch_size,]

			any_done = (done_cum[:, -1] > 0).float().unsqueeze(1)  # [batch_size, 1]

			# === VAE损失计算 ===
			state = nobs_features
			action = nactions_flat
			recon, mean, std = self.vae(state, action)
			recon_loss = F.mse_loss(recon, action)
			KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 1.0 * KL_loss

			# === Critic损失计算 ===
			next_state = next_nobs_features
			reward = nrewards
			not_done = 1.0 - any_done  # [batch_size, 1]
			
			# 复制next_state用于目标计算
			next_state_rep = torch.repeat_interleave(next_state, 10, 0)
			
			# 目标网络计算
			with torch.no_grad():
				target_actions = self.actor_target(next_state_rep, self.vae.decode(next_state_rep))
				target_Q1, target_Q2 = self.critic_target(next_state_rep, target_actions)
				target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
				target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)
				target_Q = reward.unsqueeze(1) + (gamma_powers[-1] * self.discount) * not_done * target_Q
			
			# 当前Q值计算
			current_Q1, current_Q2 = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			# === Actor损失计算 ===
			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)
			actor_loss = -self.critic.q1(state, perturbed_actions).mean()

			return critic_loss.item(), actor_loss.item(), recon_loss.item(), KL_loss.item()
	
	def predict_action(self, obs_dict: dict) -> np.ndarray:
		"""
		输入当前观测，预测未来n_action_steps步的动作序列
		
		Args:
			obs_dict (dict): 观测字典，包含图像和其他状态信息
				shape: 
					head_cam: [1, n_obs_steps, C, H, W] 
					agent_pos: [1, n_obs_steps, robot_state_dim]
		
		Returns:
			np.ndarray: 预测的动作序列 shape [n_action_steps, action_dim]
		"""
		with torch.no_grad():
			batch_size = obs_dict['head_cam'].shape[0]

			nobs = self.normalizer.normalize(obs_dict)
			value = next(iter(obs_dict.values()))
			B, To = value.shape[:2]
			T = self.horizon
			Da = self.action_dim
			Do = self.obs_feature_dim
			To = self.n_obs_steps

			this_nobs = dict_apply(nobs,
				lambda x: x[:,:self.n_obs_steps,...].reshape(-1, *x.shape[2:]))  # [n_obs_steps, C, H, W]
			nobs_features = self.obs_encoder(this_nobs)  # [n_obs_steps, obs_feature_dim]
			
			state = nobs_features.view(batch_size, -1).to(self.device)

			sampled_actions = self.vae.decode(state)
			perturbed_actions = self.actor(state, sampled_actions)

			action_seq = perturbed_actions.squeeze(0).cpu().numpy()  # [n_action_steps, action_dim]
			
			denorm_action = self.normalizer['action'].unnormalize(action_seq)
			start = To - 1
			end = start + self.n_action_steps
			result = {
				'action': denorm_action[start+1:end, :],
				'action_pred': denorm_action
			}
		return result
	
	def calculate_fid(self, real_actions: np.ndarray, 
					fake_actions: np.ndarray) -> float:
		"""
		计算动作序列的Fréchet距离（简化版，适用于一维动作）
		
		Args:
			real_actions: 真实动作序列 [N, T, D]
			fake_actions: 生成动作序列 [N, T, D]
			
		Returns:
			fid: Fréchet距离
		"""
		# 展平时间步和维度 [N*T*D, ]
		real_flat = real_actions.reshape(-1)
		fake_flat = fake_actions.reshape(-1)
		
		# 计算均值和标准差
		mu_real, sigma_real = real_flat.mean(), real_flat.std()
		mu_fake, sigma_fake = fake_flat.mean(), fake_flat.std()
		
		# 计算简化FID（适用于单变量高斯分布假设）
		fid = (mu_real - mu_fake)**2 + (sigma_real**2 + sigma_fake**2 - 2*sigma_real*sigma_fake)
		return fid