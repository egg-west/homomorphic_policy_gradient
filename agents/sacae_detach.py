"""
Copyright 2022 Sahand Rezaei-Shoshtari. All Rights Reserved.

Implementation of SAC-AE.
https://arxiv.org/abs/1910.01741

Code is based on:
https://github.com/denisyarats/pytorch_sac_ae
"""

import hydra
import wandb
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.cnn import PixelEncoder, PixelDecoder
from models.core import StochasticActor, Critic
import utils.utils as utils

from agents.drqv2 import RandomShiftsAug


class SACAEAgent:
    def __init__(self, obs_shape, action_shape, device, lr, beta, feature_dim,
                 hidden_dim, linear_approx, init_temperature, alpha_lr, alpha_beta,
                 actor_log_std_min, actor_log_std_max, actor_update_freq,
                 critic_target_tau, critic_target_update_freq, encoder_target_tau,
                 decoder_update_freq, decoder_latent_lambda, decoder_weight_lambda,
                 num_expl_steps, use_aug, decoder_update_epoch):
        self.device = device
        self.action_dim = action_shape[0]
        self.num_expl_steps = num_expl_steps
        self.critic_target_update_freq = critic_target_update_freq
        self.critic_target_tau = critic_target_tau
        self.encoder_target_tau = encoder_target_tau
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
        self.decoder_update_epoch = decoder_update_epoch
        self.obs_shape = obs_shape
        self.decoder_weight_lambda = decoder_weight_lambda

        # models
        self.pixel_encoder = PixelEncoder(obs_shape, feature_dim).to(device)
        self.pixel_encoder_target = PixelEncoder(obs_shape, feature_dim).to(device)
        self.pixel_encoder_target.load_state_dict(self.pixel_encoder.state_dict())
        self.pixel_decoder = PixelDecoder(obs_shape, feature_dim).to(device)
        self.actor = StochasticActor(feature_dim, action_shape[0], hidden_dim, linear_approx,
                                     actor_log_std_min, actor_log_std_max).to(device)

        self.critic = Critic(feature_dim, action_shape[0], hidden_dim, linear_approx).to(device)
        self.critic_target = Critic(feature_dim, action_shape[0], hidden_dim, linear_approx).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.pixel_encoder_opt = torch.optim.Adam(self.pixel_encoder.parameters(), lr=lr)
        self.pixel_decoder_opt = torch.optim.Adam(self.pixel_decoder.parameters(), lr=lr,
                                                  weight_decay=decoder_weight_lambda)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr,betas=(beta, 0.999))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(beta, 0.999))
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999))

        if self.use_aug:
            self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.pixel_encoder.train(training)
        self.pixel_decoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.pixel_encoder(obs.unsqueeze(0))

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

        obs = self.pixel_encoder(obs)
        q, _ = self.critic(obs, action)

        return {
            'state': obs.cpu().numpy()[0],
            'value': q.cpu().numpy()[0]
        }

    def update_critic(self, obs_embed, action, reward, discount, next_obs_embed, target_next_obs_embed, step):
        metrics = dict()

        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs_embed)

            target_Q1, target_Q2 = self.critic_target(target_next_obs_embed, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs_embed, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.critic_opt.zero_grad(set_to_none=True)
        self.pixel_encoder_opt.zero_grad()
        critic_loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 25)
        model_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.pixel_encoder.parameters(), 25)
        self.critic_opt.step()
        self.pixel_encoder_opt.step()

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

    def update_encoder_and_decoder(self, replay_iter, step):
        print("Updating encoder")
        metrics = dict()
        candidate_encoder = PixelEncoder(self.obs_shape, self.feature_dim).to(self.device)
        candidate_encoder_opt = torch.optim.Adam(candidate_encoder.parameters(), lr=self.lr)
        candidate_decoder = PixelDecoder(self.obs_shape, self.feature_dim).to(self.device)
        candidate_decoder_opt = torch.optim.Adam(candidate_decoder.parameters(), lr=self.lr,
                                                        weight_decay=self.decoder_weight_lambda)
        
        # wandb_run_inner = wandb.init(
        #     project=f"sacae_detach",
        #     group=f'test',
        #     name=f'{step}',
        # )

        for i in tqdm(range(min(20000, self.decoder_update_epoch * step // 256))):
            batch = next(replay_iter)
            obs, action, reward, discount, next_obs, _ = utils.to_torch(
                batch, self.device)

            obs = obs.float()
            target_obs = obs.float()

            #==================== Autoencoder Loss ====================#
            h = candidate_encoder(obs)
            if target_obs.dim() == 4:
                # preprocess images to be in [-0.5, 0.5] range
                target_obs = utils.preprocess_obs(target_obs)
            rec_obs = candidate_decoder(h)
            rec_loss = F.mse_loss(target_obs, rec_obs)

            # add L2 penalty on latent representation
            # see https://arxiv.org/pdf/1903.12436.pdf
            latent_loss = (0.5 * h.pow(2).sum(1)).mean()

            ae_loss = rec_loss + self.decoder_latent_lambda * latent_loss
            
            #==================== Distillation Loss ====================#
            pred_Q1, pred_Q2 = self.critic(h, action)
            _, pi, log_pi, log_std = self.actor(h)
            with torch.no_grad():
                target_h = self.pixel_encoder(obs)
                target_Q1, target_Q2 = self.critic(target_h, action)
                _, target_pi, target_log_pi, target_log_std = self.actor(target_h)
            q_dis_loss = F.mse_loss(pred_Q1, target_Q1) + F.mse_loss(pred_Q2, target_Q2)
            pi_dis_loss = F.mse_loss(pi, target_pi)
            distillation_loss = q_dis_loss + 5 * pi_dis_loss

            loss = ae_loss + 0.1 * distillation_loss

            candidate_encoder_opt.zero_grad(set_to_none=True)
            candidate_decoder_opt.zero_grad(set_to_none=True)
            loss.backward()
            model_grad_norm = torch.nn.utils.clip_grad_norm_(
                candidate_encoder.parameters(), 25)
            model_grad_norm = torch.nn.utils.clip_grad_norm_(
                candidate_decoder.parameters(), 25)
            candidate_encoder_opt.step()
            candidate_decoder_opt.step()

            # log_dict = {
            #     "train/ae_loss": ae_loss.item(),
            #     "train/q_dist_loss": q_dis_loss.item(),
            #     "train/pi_dis_loss": pi_dis_loss.item(),
            #     "train/tot_dis_loss": distillation_loss.item(),
            #     "train/tot_loss": loss.item()
            # }
            #wandb_run_inner.log(log_dict, i)
        #wandb_run_inner.finish()

        self.pixel_encoder = candidate_encoder
        self.pixel_encoder_target.load_state_dict(self.pixel_encoder.state_dict())
        self.pixel_encoder_opt = candidate_encoder_opt
        self.pixel_decoder = candidate_decoder
        self.pixel_decoder_opt = candidate_decoder_opt

        #metrics['ae_loss'] = loss.item()
        metrics = {
            "ae_loss": ae_loss.item(),
            "q_dist_loss": q_dis_loss.item(),
            "pi_dis_loss": pi_dis_loss.item(),
            "tot_dis_loss": distillation_loss.item(),
            "tot_loss": loss.item()
        }
        print("Encoder updated.")
        return metrics

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
        h = self.pixel_encoder(obs)
        with torch.no_grad():
            next_h = self.pixel_encoder(next_obs)
            target_next_h = self.pixel_encoder_target(next_obs)

        metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(h, action, reward, discount, next_h, target_next_h, step))

        # update actor
        if step % self.actor_update_freq == 0:
            metrics.update(self.update_actor_and_alpha(h.detach(), step))

        # update critic target
        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_target_tau)
            utils.soft_update_params(self.pixel_encoder, self.pixel_encoder_target,
                                     self.encoder_target_tau)

        # update encoder and decoder
        if step % self.decoder_update_freq == 0:
            metrics.update(self.update_encoder_and_decoder(replay_iter, step))

        return metrics

    def save(self, model_dir, step):
        model_save_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')
        model_save_dir.mkdir(exist_ok=True, parents=True)

        torch.save(self.actor.state_dict(), f'{model_save_dir}/actor.pt')
        torch.save(self.critic.state_dict(), f'{model_save_dir}/critic.pt')
        torch.save(self.pixel_encoder.state_dict(), f'{model_save_dir}/pixel_encoder.pt')
        torch.save(self.pixel_decoder.state_dict(), f'{model_save_dir}/pixel_decoder.pt')

    def load(self, model_dir, step):
        print(f"Loading the model from {model_dir}, step: {step}")
        model_load_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')

        self.actor.load_state_dict(
            torch.load(f'{model_load_dir}/actor.pt', map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f'{model_load_dir}/critic.pt', map_location=self.device)
        )
        self.pixel_encoder.load_state_dict(
            torch.load(f'{model_load_dir}/pixel_encoder.pt', map_location=self.device)
        )
        self.pixel_decoder.load_state_dict(
            torch.load(f'{model_load_dir}/pixel_decoder.pt', map_location=self.device)
        )
