"""
Copyright 2022 Sahand Rezaei-Shoshtari. All Rights Reserved.

Implementation of SAC-AE.
https://arxiv.org/abs/1910.01741

Code is based on:
https://github.com/denisyarats/pytorch_sac_ae
"""

import hydra
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn import PixelEncoder, PixelDecoder
from models.core import StochasticActor, Critic
from models.autoencoder import UniAE
import utils.utils as utils

from agents.drqv2 import RandomShiftsAug


class SACAEAgent:
    def __init__(self, obs_shape, action_shape, device, lr, beta, feature_dim,
                 hidden_dim, linear_approx, init_temperature, alpha_lr, alpha_beta,
                 actor_log_std_min, actor_log_std_max, actor_update_freq,
                 critic_target_tau, critic_target_update_freq, encoder_tau,
                 decoder_update_freq, decoder_latent_lambda, decoder_weight_lambda,
                 num_expl_steps, use_aug):
        self.device = device
        self.action_dim = action_shape[0]
        self.num_expl_steps = num_expl_steps
        self.critic_target_update_freq = critic_target_update_freq
        self.critic_target_tau = critic_target_tau
        self.encoder_tau = encoder_tau
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda
        self.actor_update_freq = actor_update_freq
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.actor_log_std_min = actor_log_std_min
        self.actor_log_std_max = actor_log_std_max
        self.lr = lr
        self.beta = beta
        self.alpha_lr = alpha_lr
        self.alpha_beta = alpha_beta
        self.init_temperature = init_temperature
        self.use_aug = use_aug

        # models
        #self.pixel_encoder = PixelEncoder(obs_shape, feature_dim).to(device)
        #self.pixel_decoder = PixelDecoder(obs_shape, feature_dim).to(device)
        self.unified_ae = UniAE(obs_shape, action_shape, feature_dim).to(device)

        self.actor = StochasticActor(feature_dim*2, action_shape[0], hidden_dim, linear_approx,
                                     actor_log_std_min, actor_log_std_max).to(device)

        self.critic = Critic(feature_dim*2, action_shape[0], hidden_dim, linear_approx).to(device)
        self.critic_target = Critic(feature_dim*2, action_shape[0], hidden_dim, linear_approx).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        #self.pixel_encoder_opt = torch.optim.Adam(self.pixel_encoder.parameters(), lr=lr)
        #self.pixel_decoder_opt = torch.optim.Adam(self.pixel_decoder.parameters(), lr=lr,
        #                                          weight_decay=decoder_weight_lambda)
        self.uae_opt = torch.optim.AdamW(self.unified_ae.parameters(), lr=lr,)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr,betas=(beta, 0.999))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(beta, 0.999))
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999))

        if self.use_aug:
            self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        #self.pixel_encoder.train(training)
        #self.pixel_decoder.train(training)
        self.unified_ae.train(training)
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        #obs = self.pixel_encoder(obs.unsqueeze(0))
        obs = self.unified_ae.obs_encode(obs.unsqueeze(0))

        if eval_mode:
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            action = mu.cpu().numpy()[0]
        else:
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            action = pi.cpu().numpy()[0]
            if step < self.num_expl_steps:
                action = np.random.uniform(-1.0, 1.0, size=self.action_dim)
        return action.astype(np.float32)

    def observe(self, obs, action):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        action = torch.as_tensor(action, device=self.device).float().unsqueeze(0)

        #obs = self.pixel_encoder(obs)
        obs = self.unified_ae.obs_encode(obs.unsqueeze(0))
        q, _ = self.critic(obs, action)

        return {
            'state': obs.cpu().numpy()[0],
            'value': q.cpu().numpy()[0]
        }

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)

            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic 
        self.critic_opt.zero_grad(set_to_none=True)
        self.uae_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        self.critic_opt.step()

        return metrics

    def update_actor_and_alpha(self, obs, step):
        metrics = dict()

        _, pi, log_pi, log_std = self.actor(obs)
        Q1, Q2 = self.critic(obs, pi)
        Q = torch.min(Q1, Q2)

        actor_loss = (self.alpha.detach() * log_pi - Q).mean()

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # optimize alpha
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.log_alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.log_alpha_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_target_ent'] = self.target_entropy.item()
        metrics['actor_ent'] = entropy.mean().item()
        metrics['alpha_loss'] = alpha_loss.item()
        metrics['alpha_value'] = self.alpha.item()

        return metrics

    # def update_encoder_and_decoder(self, obs, target_obs, step):
    #     metrics = dict()

    #     obs = obs.float()
    #     target_obs = target_obs.float()

    #     h = self.pixel_encoder(obs)
    #     if target_obs.dim() == 4:
    #         # preprocess images to be in [-0.5, 0.5] range
    #         target_obs = utils.preprocess_obs(target_obs)
    #     rec_obs = self.pixel_decoder(h)
    #     rec_loss = F.mse_loss(target_obs, rec_obs)

    #     # add L2 penalty on latent representation
    #     # see https://arxiv.org/pdf/1903.12436.pdf
    #     latent_loss = (0.5 * h.pow(2).sum(1)).mean()

    #     loss = rec_loss + self.decoder_latent_lambda * latent_loss
    #     self.pixel_encoder_opt.zero_grad(set_to_none=True)
    #     self.pixel_decoder_opt.zero_grad(set_to_none=True)
    #     loss.backward()
    #     self.pixel_encoder_opt.step()
    #     self.pixel_decoder_opt.step()

    #     metrics['ae_loss'] = loss.item()

    #     return metrics

    def update_encoder_and_decoder(self, obs, action, next_obs, step):
        metrics = dict()
        device = obs.device

        obs = obs.float()
        next_obs = next_obs.float()
        fake_obs = torch.zeros_like(obs).to(device)
        fake_action = torch.zeros_like(action).to(device)
        
        # generate 3 inputs
        obs_total = torch.cat([obs, obs, fake_obs], dim=0)
        action_total = torch.cat([action, fake_action, action], dim=0)
        next_total = torch.cat([fake_obs, next_obs, next_obs], dim=0)
        
        h, o, a, o_ = self.unified_ae(obs_total, action_total, next_total)

        #h = self.pixel_encoder(obs)
        #if target_obs.dim() == 4:
        if obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = utils.preprocess_obs(obs).detach()
            target_obs_ = utils.preprocess_obs(next_obs)
        #rec_obs = self.pixel_decoder(h)
        #target_obs = target_obs.tile((3, 1))#torch.cat([target_obs for _ in range(3)], dim=0)
        #target_obs_ = target_obs_.tile((3, 1)) #torch.cat([target_obs_ for _ in range(3)], dim=0)
        #target_actions = torch.cat([action.detach() for _ in range(3)], dim=0)

        #print(f"{o.shape=}, {target_obs.tile((3, 1)).shape=}, {a.shape=}")
        o_rec_loss = F.mse_loss(o, target_obs.tile((3, 1, 1, 1)))
        a_rec_loss = F.mse_loss(a, action.detach().tile((3, 1)))
        o_next_rec_loss = F.mse_loss(o_, target_obs_.tile((3, 1, 1, 1)))
        rec_loss = o_rec_loss + a_rec_loss + o_next_rec_loss

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss
        self.uae_opt.zero_grad()
        loss.backward()
        self.uae_opt.step()

        metrics['ae_loss'] = loss.item()

        return metrics

    # def update_encoder_and_decoder(self, obs, action, next_obs, step):
    #     metrics = dict()
    #     device = obs.device

    #     obs = obs.float()
    #     next_obs = next_obs.float()
    #     fake_obs = torch.zeros_like(obs).to(device)
    #     fake_action = torch.zeros_like(action).to(device)
        
    #     # generate 3 inputs
    #     h1, o1, a1, o1_ = self.unified_ae(obs, action, fake_obs)
    #     h2, o2, a2, o2_ = self.unified_ae(obs, fake_action, next_obs)
    #     h3, o3, a3, o3_ = self.unified_ae(fake_obs, action, next_obs)

    #     #h = self.pixel_encoder(obs)
    #     #if target_obs.dim() == 4:
    #     if obs.dim() == 4:
    #         # preprocess images to be in [-0.5, 0.5] range
    #         target_obs = utils.preprocess_obs(obs).detach()
    #         target_obs_ = utils.preprocess_obs(next_obs)
    #     #rec_obs = self.pixel_decoder(h)

    #     o_rec_loss = F.mse_loss(o1, target_obs) + F.mse_loss(o2, target_obs) + F.mse_loss(o3, target_obs)
    #     a_rec_loss = F.mse_loss(a1, action.detach()) + F.mse_loss(a2, action.detach()) + F.mse_loss(a3, action.detach())
    #     o_next_rec_loss = F.mse_loss(o1_, target_obs_) + F.mse_loss(o2_, target_obs_) + F.mse_loss(o3_, target_obs_)
    #     rec_loss = o_rec_loss + a_rec_loss + o_next_rec_loss

    #     # add L2 penalty on latent representation
    #     # see https://arxiv.org/pdf/1903.12436.pdf
    #     latent_loss = (0.5 * h1.pow(2).sum(1)).mean() + (0.5 * h2.pow(2).sum(1)).mean() + (0.5 * h3.pow(2).sum(1)).mean()

    #     loss = rec_loss + self.decoder_latent_lambda * latent_loss
    #     self.uae_opt.zero_grad()
    #     loss.backward()
    #     self.uae_opt.step()

    #     metrics['ae_loss'] = loss.item()

    #     return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, _ = utils.to_torch(
            batch, self.device)

        # image augmentation
        if self.use_aug:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())

        # encode
        # h = self.pixel_encoder(obs)
        # with torch.no_grad():
        #     next_h = self.pixel_encoder(next_obs)

        h = self.unified_ae.obs_encode(obs)
        with torch.no_grad():
            next_h = self.unified_ae.obs_encode(next_obs)

        metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(h, action, reward, discount, next_h, step))

        # update actor
        if step % self.actor_update_freq == 0:
            metrics.update(self.update_actor_and_alpha(h.detach(), step))

        # update critic target
        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)

        # update encoder and decoder
        if step % self.decoder_update_freq == 0:
            #metrics.update(self.update_encoder_and_decoder(obs, obs, step))
            metrics.update(self.update_encoder_and_decoder(obs, action, next_obs, step))

        return metrics

    def save(self, model_dir, step):
        model_save_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')
        model_save_dir.mkdir(exist_ok=True, parents=True)

        torch.save(self.actor.state_dict(), f'{model_save_dir}/actor.pt')
        torch.save(self.critic.state_dict(), f'{model_save_dir}/critic.pt')
        #torch.save(self.pixel_encoder.state_dict(), f'{model_save_dir}/pixel_encoder.pt')
        #torch.save(self.pixel_decoder.state_dict(), f'{model_save_dir}/pixel_decoder.pt')
        torch.save(self.unified_ae.state_dict(), f'{model_save_dir}/unified_ae.pt')

    def load(self, model_dir, step):
        print(f"Loading the model from {model_dir}, step: {step}")
        model_load_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')

        self.actor.load_state_dict(
            torch.load(f'{model_load_dir}/actor.pt', map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f'{model_load_dir}/critic.pt', map_location=self.device)
        )
        # self.pixel_encoder.load_state_dict(
        #     torch.load(f'{model_load_dir}/pixel_encoder.pt', map_location=self.device)
        # )
        # self.pixel_decoder.load_state_dict(
        #     torch.load(f'{model_load_dir}/pixel_decoder.pt', map_location=self.device)
        # )
        self.unified_ae.load_state_dict(
            torch.load(f'{model_load_dir}/unified_ae.pt', map_location=self.device)
        )