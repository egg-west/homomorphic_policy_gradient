import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.utils as utils
from models.cnn import PixelEncoder, PixelDecoder, UniPixelDecoder

class UniAE(nn.Module):
    def __init__(self, obs_shape, a_dim, feature_dim):
        super().__init__()
        self.a_dim = a_dim[0]
        
        self.uni_obs_encoder = PixelEncoder(obs_shape, feature_dim)
        self.obs_decoder = UniPixelDecoder(obs_shape, feature_dim)
        #self.next_obs_decoder = UniPixelDecoder(obs_shape, feature_dim)
        
        self.a_encoder = nn.Sequential(
            nn.Linear(self.a_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
            # nn.Linear(feature_dim, feature_dim),
            # nn.LayerNorm(feature_dim),
            # nn.Tanh()
        )

        # self.a_decoder = nn.Sequential(
        #     nn.Linear(feature_dim, feature_dim),
        #     nn.LayerNorm(feature_dim),
        #     nn.ReLU(),
        #     nn.Linear(feature_dim, self.a_dim),
        # )

        self.mixer = nn.Sequential(
            nn.Linear(feature_dim*2, feature_dim),
        )

    def forward(self, obs, action):
        obs_embed = self.uni_obs_encoder(obs)
        #next_obs_embed = self.uni_obs_encoder(next_obs)
        action_embed = self.a_encoder(action)

        #h = torch.cat([obs_embed, next_obs_embed, action_embed], dim=1)
        h = torch.cat([obs_embed, action_embed], dim=1)
        h = self.mixer(h)

        obs_and_next_obs_recon = self.obs_decoder(h)

        return h, obs_and_next_obs_recon

    def obs_encode(self, obs):
        fake_actions = torch.zeros((obs.shape[0], self.a_dim)).to(obs.device)
        fake_action_embed = self.a_encoder(fake_actions)
        obs_embed = self.uni_obs_encoder(obs)
        #fake_action_embed = torch.zeros_like(obs_embed).to(obs.device)

        h = torch.cat([obs_embed, fake_action_embed], dim=1)
        h = self.mixer(h)

        return h

    def obs_action_encode(self, obs, actions):
        obs_embed = self.uni_obs_encoder(obs)
        action_embed = self.a_encoder(actions)

        h = torch.cat([obs_embed, action_embed], dim=1)
        h = self.mixer(h)

        return h

# class UniAE(nn.Module):
#     def __init__(self, obs_shape, a_dim, feature_dim):
#         super().__init__()
#         self.a_dim = a_dim[0]
        
#         self.uni_obs_encoder = PixelEncoder(obs_shape, feature_dim)
#         self.obs_decoder = PixelDecoder(obs_shape, feature_dim*2)
#         self.next_obs_decoder = PixelDecoder(obs_shape, feature_dim*2)
        
#         self.a_encoder = nn.Sequential(
#             nn.Linear(self.a_dim, feature_dim),
#             nn.LayerNorm(feature_dim),
#             nn.ReLU(),
#             nn.Linear(feature_dim, feature_dim),
#             nn.LayerNorm(feature_dim),
#             nn.Tanh()
#         )

#         self.a_decoder = nn.Sequential(
#             nn.Linear(feature_dim*2, feature_dim),
#             nn.LayerNorm(feature_dim),
#             nn.ReLU(),
#             nn.Linear(feature_dim, self.a_dim),
#         )

#         self.mixer = nn.Sequential(
#             nn.Linear(feature_dim*3, feature_dim*2),
#         )

#     def forward(self, obs, action, next_obs):
#         obs_embed = self.uni_obs_encoder(obs)
#         next_obs_embed = self.uni_obs_encoder(next_obs)
#         action_embed = self.a_encoder(action)

#         h = torch.cat([obs_embed, next_obs_embed, action_embed], dim=1)
#         h = self.mixer(h)

#         obs_recon = self.obs_decoder(h)
#         next_obs_recon = self.next_obs_decoder(h)
#         a_recon = self.a_decoder(h)
#         return h, obs_recon, a_recon, next_obs_recon

#     def obs_encode(self, obs):
#         fake_next_obs = torch.zeros_like(obs).to(obs.device)
#         fake_actions = torch.zeros((obs.shape[0], self.a_dim)).to(obs.device)
#         # print(f"{obs.device=}") cuda
#         #inputs = torch.cat([obs, fake_next_obs, fake_actions], dim=1)
#         #h, obs_recon, a_recon, next_obs_recon = self.forward(obs, fake_actions, fake_next_obs)
#         obs_embed = self.uni_obs_encoder(obs)
#         next_obs_embed = self.uni_obs_encoder(fake_next_obs)
#         action_embed = self.a_encoder(fake_actions)

#         h = torch.cat([obs_embed, next_obs_embed, action_embed], dim=1)
#         h = self.mixer(h)

#         return h

#     def obs_action_encode(self, obs, actions):
#         fake_next_obs = torch.zeros_like(obs).to(obs.device)
#         #inputs = torch.cat([obs, fake_next_obs, actions], dim=1)
#         #h, obs_recon, a_recon, next_obs_recon = self.forward(obs, actions, fake_next_obs)
#         obs_embed = self.uni_obs_encoder(obs)
#         next_obs_embed = self.uni_obs_encoder(fake_next_obs)
#         action_embed = self.a_encoder(actions)

#         h = torch.cat([obs_embed, next_obs_embed, action_embed], dim=1)
#         h = self.mixer(h)

#         return h