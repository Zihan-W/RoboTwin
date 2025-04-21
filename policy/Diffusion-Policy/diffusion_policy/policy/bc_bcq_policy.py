import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer

class ResidualActor(nn.Module):
	'''
	Perturbation model, perturbate residual actions given by VAE
	'''
	def __init__(self, state_dim, action_dim, action_seq_dim, max_action, n_action_steps, phi=0.05):
		super(ResidualActor, self).__init__()
		self.n_action_steps = n_action_steps
		self.l1 = nn.Linear(state_dim + action_seq_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_seq_dim)
		
		self.max_action = max_action
		self.action_dim = action_dim
		self.phi = phi

	def forward(self, state, residual_action_seq):
		batch_size = residual_action_seq.shape[0]	# action_seq: [batch_size, n_action_steps, action_dim]
		action_seq_flat = residual_action_seq.view(batch_size, -1)  # [batch_size, n_action_steps*action_dim]
		a = F.relu(self.l1(torch.cat([state, action_seq_flat], 1)))
		a = F.relu(self.l2(a))
		a = self.phi * self.max_action * torch.tanh(self.l3(a))	# perturbation

		residual = a.view(batch_size, self.n_action_steps, -1)  # [batch_size, n_steps, action_dim]
		return residual
		# return (a + action_seq).clamp(-self.max_action[0:self.action_dim], self.max_action[0:self.action_dim])


class Critic(nn.Module):
	'''
	Critic for final actions (base_actions + residual_actions)
	'''
	def __init__(self, state_dim, action_seq_dim):
		super(Critic, self).__init__()
		self.l1 = nn.Linear(state_dim + action_seq_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		self.l4 = nn.Linear(state_dim + action_seq_dim, 400)
		self.l5 = nn.Linear(400, 300)
		self.l6 = nn.Linear(300, 1)


	def forward(self, state, action_seq):
		action_seq_flat = action_seq.view(action_seq.shape[0], -1)
		q1 = F.relu(self.l1(torch.cat([state, action_seq_flat], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(torch.cat([state, action_seq_flat], 1)))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def q1(self, state, action_seq):
		action_seq_flat = action_seq.view(action_seq.shape[0], -1)
		q1 = F.relu(self.l1(torch.cat([state, action_seq_flat], 1)))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


# Vanilla Variational Auto-Encoder 
class ResidualVAE(nn.Module):
	def __init__(self, state_dim, action_dim, latent_dim, max_action, n_action_steps, device):
		super(ResidualVAE, self).__init__()
		action_seq_dim = action_dim * n_action_steps
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
		self.action_dim = action_dim
		self.n_action_steps = n_action_steps


	def forward(self, state, action_seq):
		'''
		Learn distribution of residual actions
		'''
		z = F.relu(self.e1(torch.cat([state, action_seq], 1)))
		z = F.relu(self.e2(z))

		mean = self.mean(z)
		# Clamped for numerical stability 
		log_std = self.log_std(z).clamp(-4, 15)
		std = torch.exp(log_std)
		z = mean + std * torch.randn_like(std)
		
		u = self.decode(state, z)

		return u, mean, std


	def decode(self, state, z=None):
		'''
		Output residual actions learned from dataset
		'''
		# When sampling from the VAE, the latent vector is clipped to [-max_action, max_action]
		if z is None:
			z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

		a = F.relu(self.d1(torch.cat([state, z], 1)))
		a = F.relu(self.d2(a))
		a = self.max_action * torch.tanh(self.d3(a))
		return a.view(-1, self.n_action_steps, self.action_dim)

class BC_BCQ(nn.Module):
	def __init__(self,
			  shape_meta: dict,
			  n_obs_steps: int,
			  n_action_steps: int,
			  task_name: str,
			  head_camera_type: str,
			  expert_data_num: int,
			  discount=0.99,
			  tau=0.005,
			  lmbda=0.75,
			  lr_critic=1e-3,
			  lr_actor=1e-3,
			  phi=0.05,
			  obs_encoder: MultiImageObsEncoder = None,
			  **kwargs):
		super(BC_BCQ, self).__init__()

		self.device = torch.device("cuda:0")

		action_shape = shape_meta['action']['shape']
		self.action_dim = action_shape[0]
		latent_dim = self.action_dim * 2
		self.n_action_steps = n_action_steps
		self.action_seq_dim = self.action_dim * n_action_steps  # 多步动作序列维度

		self.max_action = torch.full((self.action_seq_dim,), 2.7).to(self.device)  # 设置动作上限值为2.7

		self.obs_encoder = obs_encoder
		self.obs_feature_dim = obs_encoder.output_shape()[0]
		self.n_obs_steps = n_obs_steps
		self.state_dim = self.obs_feature_dim	# TODO:可能需要加上robot_state这几个维度
		self.state_seq_dim = n_obs_steps * self.state_dim
		
		# Initialize Q-networks and target_Q-networks
		self.critic = Critic(self.state_seq_dim, self.action_seq_dim).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

		# Initialize perturbation network and target_perturbation network
		self.actor = ResidualActor(self.state_seq_dim, self.action_dim, self.action_seq_dim, self.max_action, self.n_action_steps, phi).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

		# Initialize VAE network
		self.vae = ResidualVAE(self.state_seq_dim, self.action_dim, latent_dim, self.max_action, self.n_action_steps, self.device).to(self.device)
		self.vae_optimizer = torch.optim.Adam(self.vae.parameters()) 

		self.discount = discount
		self.tau = tau
		self.lmbda = lmbda

		self.normalizer = LinearNormalizer()

		from script.eval_policy import BC
		# 新增BC策略组件
		self.bc_policy = BC('put_apple_cabinet', head_camera_type, 600, expert_data_num, seed=0)
		# self.bc_policy.normalizer = self.normalizer
		# self.policy.update_obs(obs)	# 更新obs
		# actions = self.policy.get_action(obs)	# 获取BC模型动作

	def get_base_action(self, obs_dict):
		"""获取BC基础动作并进行归一化处理"""
		# 原始BC动作 [n_steps, action_dim]
		raw_action = self.bc_policy.get_action_as_base(obs_dict)  
        # 转换为tensor并归一化
		return self.normalizer['action'].normalize(raw_action)

	def select_action(self, state):	
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
			action_seq = self.actor(state, self.vae.decode(state))	# # [batch, n_steps, action_dim]
			q1 = self.critic.q1(state, action_seq)
			ind = q1.argmax(0)
		return action_seq[ind, 0, :]  # 返回序列动作
		#return action[ind].cpu().data.numpy().flatten()

	def update(self, batch):
		# process sampled batch
		iterations = 1
		batch_size = batch['reward'].shape[0]
		batch = dict_apply(batch, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x)	# transfer to a certain device
		for it in range(iterations):
			with torch.no_grad():
				nobs = self.normalizer.normalize(batch['obs'])  # normalize, shape of result: [batch_size, horzion, channels, H, W]
				next_nobs = self.normalizer.normalize(batch['next_obs'])	# normalize, shape of result: [batch_size, horzion, channels, H, W]
				# obs encoder and reshape
				this_nobs = dict_apply(nobs, 
						lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))  # shape of result: [batch_size * n_obs_steps, channles, H, W]
				next_this_nobs = dict_apply(next_nobs, 
						lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))  # shape of result: [batch_size * n_obs_steps, channles, H, W]
				nobs_features = self.obs_encoder(this_nobs) # shape of result: [batch_size * n_obs_steps, obs_feature_dim]
				next_nobs_features = self.obs_encoder(next_this_nobs) # shape of result: [batch_size * n_obs_steps, obs_feature_dim]
				nobs_features = nobs_features.view(batch_size, -1)  # shape of result: [batch_size, n_obs_steps * obs_feature_dim]
				next_nobs_features = next_nobs_features.view(batch_size, -1)  # shape of result: [batch_size, n_obs_steps * obs_feature_dim]

			# process action and reshape
			nactions = batch['action']
			with torch.no_grad():
				nbc_actions = self.get_base_action(batch['obs'])  # get BC action as base action，shape of result: [batch_size, n_steps, action_dim]
			ngt_residual_actions = (nactions - nbc_actions)	# get ground_truth residual action, shape of result: [batch_size, n_steps, action_dim]
			nactions_flat = self.normalizer['action'].normalize(nactions) # Normalize, shape of result: [batch_size, horzion, action_dim]
			nactions_flat = nactions_flat[:, :self.n_action_steps, :]  # shape of result: [batch_size, n_action_steps, action_dim], if horzion==n_action_steps
			nactions_flat = nactions.view(batch_size, -1)  # shape of result: [batch_size, n_steps*action_dim]
			ngt_residual_actions_flat = ngt_residual_actions.view(batch_size, -1)  # shape of result: [batch_size, n_steps*action_dim]
			
			# process reward and dones
			nrewards = batch['reward'].sum(dim=1)	# [batch_size, horizon] -> [batch_size, ]
			nnot_done = 1 - batch['done']	# [batch_size, horizon]

			# Sample replay buffer / batch
			# TODO：maybe modify data used
			state = nobs_features.to(self.device)
			next_state = next_nobs_features.to(self.device)
			action = nactions_flat.to(self.device)
			reward = nrewards.to(self.device)
			not_done = nnot_done.to(self.device)
			ngt_residual_actions = ngt_residual_actions.to(self.device)
			ngt_residual_actions_flat = ngt_residual_actions_flat.to(self.device)

			# Variational Auto-Encoder Training
			recon, mean, std = self.vae(state, ngt_residual_actions_flat)
			recon_loss = F.mse_loss(recon, ngt_residual_actions)
			KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 0.5 * KL_loss

			self.vae_optimizer.zero_grad()
			vae_loss.backward()
			self.vae_optimizer.step()

			# Critic Training
			with torch.no_grad():
				# Duplicate next state 10 (repeat_num) times
				repeat_num = 10
				next_state_repeat_torch = torch.repeat_interleave(next_state, repeat_num, 0)
				next_state_repeat_dict = self.repeat_dict_interleave(batch['next_obs'], repeat_num)
				
				# Get base action
				next_nbc_action = self.get_base_action(next_state_repeat_dict)
				# Get sampled actions from the VAE
				next_nsampled_actions = self.vae.decode(next_state_repeat_torch)
				# Get perturbed actions from actor
				next_nsampled_actions = self.actor_target(next_state_repeat_torch, next_nsampled_actions)
				# Combine base actions and perturbed actions
				next_nsampled_actions = next_nsampled_actions + next_nbc_action
				
				# Compute value of perturbed actions predicted via Base Policy
				target_Q1, target_Q2 = self.critic_target(next_state_repeat_torch, next_nsampled_actions)

				# Soft Clipped Double Q-learning 
				target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
				# Take max over each action sampled from the VAE
				target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)

				# Compute Q value using n_step rewards
				reward_expanded = reward.unsqueeze(1)  # [batch_size, 1]
				not_done_last = not_done[:, -1].unsqueeze(1)  # 取最后一步的终止标志 [batch_size, 1]
				target_Q = reward_expanded + (self.discount ** self.n_action_steps) * not_done_last * target_Q
			
			current_Q1, current_Q2 = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
			
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Pertubation Model / Action Training
			nsampled_actions = self.vae.decode(state)
			perturbed_actions = nbc_actions + self.actor(state, nsampled_actions)

			# Update through DPG
			actor_loss = -self.critic.q1(state, perturbed_actions).mean()
			
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update Target Networks 
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			return critic_loss.item(), actor_loss.item()
		
	def set_normalizer(self, normalizer: LinearNormalizer):
		self.normalizer.load_state_dict(normalizer.state_dict())

	def evaluate(self, batch):
		batch_size = batch['reward'].shape[0]
		with torch.no_grad():			
			# process and reshape obs
			nobs = self.normalizer.normalize(batch['obs'])	# normalize
			next_nobs = self.normalizer.normalize(batch['next_obs'])	# normalize
			this_nobs = dict_apply(nobs,
					lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
			next_this_nobs = dict_apply(next_nobs,
					lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
			nobs_features = self.obs_encoder(this_nobs).view(batch_size, -1)
			next_nobs_features = self.obs_encoder(next_this_nobs).view(batch_size, -1)
			
			# process and reshape actions
			nactions = batch['action']
			nbc_actions = self.get_base_action(batch['obs'])  # get BC action as base action
			ngt_residual_actions = (nactions - nbc_actions)	# get ground_truth residual action
			nactions_flat = self.normalizer['action'].normalize(nactions)
			nactions_flat = nactions[:, :self.n_action_steps, :]
			nactions_flat = nactions.view(batch_size, -1)	
			ngt_residual_actions_flat = ngt_residual_actions.view(batch_size, -1)

			# process rewards and dones
			nrewards = batch['reward'].sum(dim=1)
			nnot_done = 1 - batch['done']

			# data used
			state = nobs_features.to(self.device)
			next_state = next_nobs_features.to(self.device)
			action = nactions_flat.to(self.device)
			reward = nrewards.to(self.device)
			not_done = nnot_done.to(self.device)
			ngt_residual_actions = ngt_residual_actions.to(self.device)
			ngt_residual_actions_flat = ngt_residual_actions_flat.to(self.device)

			# Variational Auto-Encoder Evaluating		
			recon, mean, std = self.vae(state, ngt_residual_actions_flat)
			recon_loss = F.mse_loss(recon, ngt_residual_actions)
			KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
			vae_loss = recon_loss + 0.5 * KL_loss
			
			# critic evaluating
			repeat_num = 10
			with torch.no_grad():	# ensure that net won't update
				next_state_repeat_torch = torch.repeat_interleave(next_state, repeat_num, 0)
				next_state_repeat_dict = self.repeat_dict_interleave(batch['next_obs'], repeat_num)
				
				# Get base action
				next_nbc_action = self.get_base_action(next_state_repeat_dict)
				# Get sampled actions from the VAE
				next_nsampled_actions = self.vae.decode(next_state_repeat_torch)
				# Get perturbed actions from actor
				next_nsampled_actions = self.actor_target(next_state_repeat_torch, next_nsampled_actions)
				# Combine base actions and perturbed actions
				next_nsampled_actions = next_nsampled_actions + next_nbc_action

				# Compute value of perturbed actions predicted via Base Policy
				target_Q1, target_Q2 = self.critic_target(next_state_repeat_torch, next_nsampled_actions)
				
				# Soft Clipped Double Q-learning 
				target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
				# Take max over each action sampled from the VAE
				target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)
				# Compute Q value using n_step rewards
				reward_expanded = reward.unsqueeze(1)
				not_done_last = not_done[:, -1].unsqueeze(1)  # 取最后一步的终止标志 [batch_size, 1]
				target_Q = reward_expanded + (self.discount ** self.n_action_steps) * not_done_last * target_Q
			
			# Calculate critic loss
			current_Q1, current_Q2 = self.critic(state, action)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

			# Calculate actor loss
			nsampled_actions = self.vae.decode(state)
			perturbed_actions = nbc_actions + self.actor(state, nsampled_actions)
			actor_loss = -self.critic.q1(state, perturbed_actions).mean()

			return vae_loss, critic_loss.item(), actor_loss.item()
	
	def predict_action(self, obs_dict: dict) -> np.ndarray:
		"""
		输入当前观测,预测未来n_action_steps步的动作序列
		
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
			
			# process obs
			nobs = self.normalizer.normalize(obs_dict)
			this_nobs = dict_apply(nobs,
				lambda x: x[:,:self.n_obs_steps,...].reshape(-1, *x.shape[2:]))  # [n_obs_steps, C, H, W]
			nobs_features = self.obs_encoder(this_nobs)  # [n_obs_steps, obs_feature_dim]
			state = nobs_features.view(batch_size, -1).to(self.device)

			# Get base action from Base Policy
			base_action_seq = self.get_base_action(obs_dict)  # [n_action_steps, action_dim]
			
			# Get residual action from vae
			sampled_actions = self.vae.decode(state)
			residual_actions = self.actor(state, sampled_actions)
			
			# Combine the base actions and residual actions
			final_actions = sampled_actions + residual_actions
			# Output final_actions
			final_actions = final_actions.squeeze(0).cpu().numpy()  # [n_action_steps, action_dim]
			
			# 反归一化（如果训练时做了动作归一化）
			denorm_action = self.normalizer['action'].unnormalize(final_actions)
		return denorm_action
	
	def repeat_dict_interleave(self, data_dict, repeats, dim=0):
		"""对字典中的所有Tensor进行重复采样"""
		return {
			k: torch.repeat_interleave(v, repeats, dim=dim)
			for k,v in data_dict.items()
		}