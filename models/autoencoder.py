import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.utils as utils
from models.cnn import PixelEncoder

# class PixelEncoder(nn.Module):
#     def __init__(self, obs_shape, feature_dim):
#         super().__init__()

#         assert len(obs_shape) == 3
#         self.repr_dim = 32 * 35 * 35

#         self.convnet = nn.Sequential(
#             nn.Conv2d(obs_shape[0], 32, 3, stride=2),
#             nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
#             nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
#             nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
#             nn.ReLU()
#         )
#         if feature_dim <= 10:
#             self.trunk = nn.Sequential(
#                 nn.Linear(self.repr_dim, 50),
#                 nn.LayerNorm(50),
#                 nn.Linear(50, feature_dim),
#                 nn.Tanh()
#             )
#         else:
#             self.trunk = nn.Sequential(
#                 nn.Linear(self.repr_dim, feature_dim),
#                 nn.LayerNorm(feature_dim),
#                 nn.Tanh()
#             )

#         self.apply(utils.weight_init)

#     def forward(self, obs):
#         obs = obs / 255.0 - 0.5
#         h = self.convnet(obs)
#         h = h.view(h.shape[0], -1)
#         h = self.trunk(h)
#         return h

class UniAE(nn.Module):
    def __init__(self, obs_shape, a_dim, feature_dim):
        super().__init__()
        self.uni_obs_encoder = PixelEncoder(obs_shape, feature_dim)
        self.obs_decoder = PixelDecoder(obs_shape, feature_dim*2)
        self.next_obs_decoder = PixelDecoder(obs_shape, feature_dim*2)
        
        self.a_encoder = nn.Sequential(
            nn.Linear(a_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(a_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )

        self.a_decoder = nn.Sequential(
            nn.Linear(feature_dim*2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, a_dim),
        )

        self.mixer = nn.Sequential(
            nn.Linear(feature_dim*3, feature_dim*2),
        )

    def forward(self, obs, action, next_obs):
        obs_embed = self.obs_encoder(obs)
        next_obs_embed = self.obs_encoder(next_obs)
        action_embed = self.a_encoder(action)
        
        h = torch.cat([obs_embed, next_obs_embed, action_embed], dim=1)
        h = self.mixer(h)

        obs_recon = self.obs_decoder(h)
        next_obs_recon = self.next_obs_decoder(h)
        a_recon = self.a_decoder(h)
        return h, obs_recon, a_recon, next_obs_recon