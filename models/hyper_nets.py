import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperNet(nn.Module):
    def __init__(self, cfg, state_dim, action_dim, abs_action_dim, hidden_dim, hypernet_layers=2):
        super(HyperNet, self).__init__()

        self.args = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.abs_action_dim = abs_action_dim

        self.embed_dim = hidden_dim

        #if getattr(args, "hypernet_layers", 1) == 1:
        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.action_dim)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim * self.abs_action_dim)
        elif hypernet_layers == 2:
            hypernet_embed = 64 #self.args.hypernet_embed #TODO configurable
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.action_dim))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.abs_action_dim))
        #elif getattr(args, "hypernet_layers", 1) > 2:
        elif hypernet_layers > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, states, actions):
        bs = states.shape[0]
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(actions, w1) + b1)
        # Second layer
        #w_final = torch.abs(self.hyper_w_final(states))
        w_final = self.hyper_w_final(states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, self.abs_action_dim)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, self.abs_action_dim)
        return q_tot