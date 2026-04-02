# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
from torch.distributions import Normal
from src.base.modules.normalization import Normalizer

def create_nn(input_size, output_size, hidden_size, num_layers, activation_fn=nn.ReLU, input_normalizer=None,
              final_activation_fn=None, hidden_init_fn=None, b_init_value=None, last_fc_init_w=None):
    # Optionally add a normalizer as the first layer
    if input_normalizer is None:
        input_normalizer = nn.Sequential()
    layers = [input_normalizer]

    # Create and initialize all layers except the last one
    for layer_idx in range(num_layers - 1):
        fc = nn.Linear(input_size if layer_idx == 0 else hidden_size, hidden_size)
        if hidden_init_fn is not None:
            hidden_init_fn(fc.weight)
        if b_init_value is not None:
            fc.bias.data.fill_(b_init_value)
        layers += [fc, activation_fn()]

    # Create and initialize  the last layer
    last_fc = nn.Linear(hidden_size, output_size)
    if last_fc_init_w is not None:
        last_fc.weight.data.uniform_(-last_fc_init_w, last_fc_init_w)
        last_fc.bias.data.uniform_(-last_fc_init_w, last_fc_init_w)
    layers += [last_fc]

    # Optionally add a final activation function
    if final_activation_fn is not None:
        layers += [final_activation_fn()]
    return nn.Sequential(*layers)


class Value(nn.Module):
    def __init__(self,args, hidden_dims=[256, 256], state_size=None, goal_size=None, action_size=None):
        super().__init__()
        self.a_range = args.max_action
        self.state_size = args.dim_state if state_size is None else state_size
        self.goal_size = args.dim_goal  if goal_size is None else goal_size
        self.action_size = args.dim_action if action_size is None else action_size
        print("action_size:",self.action_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
        input_size = self.state_size + self.goal_size

        fc = [nn.Linear(input_size, hidden_dims[0]), nn.ReLU()]
        for hidden_dim1, hidden_dim2 in zip(hidden_dims[:-1], hidden_dims[1:]):
            fc += [nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU()]
        fc += [nn.Linear(hidden_dims[-1], 1)]
        self.fc = nn.Sequential(*fc).to(device)

    def forward(self, state, goal):
        x = torch.cat([state, goal], -1)
        return self.fc(x)

class StochasticPolicy(nn.Module):
    def __init__(self, args, hidden_dims=[256, 256], state_size=None, goal_size=None, action_size=None):
        super().__init__()
        self.a_range = args.max_action
        self.state_size = args.dim_state if state_size is None else state_size
        self.goal_size = args.dim_goal  if goal_size is None else goal_size
        self.action_size = args.dim_action if action_size is None else action_size
        print("action_size:",self.action_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_size = self.state_size + self.goal_size

        fc = [nn.Linear(input_size, hidden_dims[0]), nn.ReLU()]
        for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
        self.fc = nn.Sequential(*fc).to(device)
        
        self.mean_linear = nn.Linear(hidden_dims[-1], self.action_size).to(device)
        self.logstd_linear = nn.Linear(hidden_dims[-1], self.action_size).to(device)

        self.LOG_SIG_MIN, self.LOG_SIG_MAX = -20, 2

    def forward(self, state, goal):
            x = self.fc(torch.cat([state, goal], -1))
            mean = self.mean_linear(x)
            log_std = self.logstd_linear(x)
            std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX).exp()
            normal = torch.distributions.Normal(mean, std)
            return normal

    def sample(self, state, goal):
            normal = self.forward(state, goal)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)
            mean = torch.tanh(normal.mean)
            return torch.clamp(action, -self.a_range, self.a_range).squeeze(0), log_prob, mean
        

class ReparamTrickPolicy(nn.Module):
    """ Gaussian policy which makes uses of the reparameterization trick to backprop gradients from a critic """
    def __init__(self, env, hidden_size, a_range=None, state_size=None, goal_size=None, action_size=None,
                 num_layers=4, normalize_inputs=False, min_logstd=-20, max_logstd=2,
                 hidden_init_fn=None, b_init_value=None, last_fc_init_w=None):
        super().__init__()
        self.a_range = env.action_range if a_range is None else a_range
        self.state_size = env.state_size if state_size is None else state_size
        self.goal_size = env.goal_size if goal_size is None else goal_size
        self.action_size = env.action_size if action_size is None else action_size

        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

        input_size = self.state_size + self.goal_size

        assert num_layers >= 2
        self.num_layers = int(num_layers)

        input_normalizer = Normalizer(input_size) if normalize_inputs else nn.Sequential()
        self.layers = create_nn(input_size=input_size, output_size=self.action_size * 2, hidden_size=hidden_size,
                                num_layers=self.num_layers, input_normalizer=input_normalizer,
                                hidden_init_fn=hidden_init_fn, b_init_value=b_init_value, last_fc_init_w=last_fc_init_w)

    def action_stats(self, s, g):
        x = torch.cat([s, g], dim=1) if g is not None else s
        action_stats = self.layers(x)
        mean = action_stats[:, :self.action_size]
        log_std = action_stats[:, self.action_size:]
        log_std = torch.clamp(log_std, self.min_logstd, self.max_logstd)
        std = log_std.exp()
        return mean, std

    def scale_action(self, logit):
        # Scale to the action range
        action = logit * self.a_range
        return action

    def forward(self, s, g=None, greedy=False, action_logit=None):
        mean, std = self.action_stats(s, g)
        m = torch.distributions.Normal(mean, std)

        # Sample.
        if action_logit is None:
            if greedy:
                action_logit_unbounded = mean
            else:
                action_logit_unbounded = m.rsample()  # for the reparameterization trick
            action_logit = torch.tanh(action_logit_unbounded)

            n_ent = -m.entropy().mean(dim=1)
            lprobs = m.log_prob(action_logit_unbounded) - torch.log(1 - action_logit.pow(2) + 1e-6)
            action = self.scale_action(action_logit)
            return action, action_logit_unbounded, lprobs, n_ent

        # Evaluate the action previously taken
        else:
            action_logit_unbounded = action_logit
            action_logit = torch.tanh(action_logit_unbounded)
            n_ent = -m.entropy().mean(dim=1)
            lprobs = m.log_prob(action_logit_unbounded) - torch.log(1 - action_logit.pow(2) + 1e-6)
            action = self.scale_action(action_logit)
            return lprobs, n_ent, action
