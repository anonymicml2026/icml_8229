import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.nn.functional as f
from functools import partial
from typing import List, Optional, Type
from src.common_1 import MLP, EnsembleMLP, LinearEnsemble

################################################################################
#
# Policy Network
#
################################################################################

# Initialize Policy weights
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weight_init(m: nn.Module, gain: int = 1) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    if isinstance(m, LinearEnsemble):
        for i in range(m.ensemble_size):
            # Orthogonal initialization doesn't care about which dim is first
            # Thus, we can just use ortho init as normal on each matrix.
            nn.init.orthogonal_(m.weight.data[i], gain=gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)

class Actor(nn.Module):
    """
    The policy network
    """
    def __init__(self, args):
        super(Actor, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal

        self.net = nn.Sequential(
            nn.Linear(dim_state+dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_action),
            nn.Tanh()
        )

    def forward(self, s, g):
        x = torch.cat([s, g], -1)
        actions = self.max_action * self.net(x)
        return actions
    
class GofarActor(nn.Module):
    """
    The policy network
    """
    def __init__(self, args):
        super(GofarActor, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal

        self.net = nn.Sequential(
            nn.Linear(dim_state+dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_action),
            nn.Tanh()
        )
    
    def forward(self, x):
        actions = self.max_action * self.net(x)
        return actions

class GaussianActor(nn.Module):
    """
    Gaussian Policy Network
    """
    def __init__(self, args):
        super(GaussianActor, self).__init__()
        self.max_action = args.max_action
        state_dim  = args.dim_state 
        hidden_dims = [args.dim_hidden,args.dim_hidden]
        action_dim = args.dim_action
        goal_dim   = args.dim_goal 
        fc = [nn.Linear(state_dim+goal_dim, hidden_dims[0]), nn.ReLU()]
        for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
            fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
        self.fc = nn.Sequential(*fc)

        self.mean_linear = nn.Linear(hidden_dims[-1], action_dim)
        self.logstd_linear = nn.Linear(hidden_dims[-1], action_dim)

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
        return action, log_prob, mean
    
################################################################################
#
# Critic Networks
#
################################################################################

class CriticMonolithic(nn.Module):
    """
    Monolithic Action-value Function Network (Q)
    """
    def __init__(self, args):
        super(CriticMonolithic, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal

        self.net = nn.Sequential(
            nn.Linear(dim_state+dim_goal+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, 1)
        )

    def forward(self, s, a, g):
        x = torch.cat([s, a/self.max_action, g], -1)
        q_value = self.net(x)
        return q_value

class GofarCritic(nn.Module):
    def __init__(self, args):
        super(GofarCritic, self).__init__()
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_hidden
        dim_action = args.dim_action
        dim_goal   = args.dim_goal

        self.net = nn.Sequential(
            nn.Linear(dim_state+dim_goal+dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, 1)
        )

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=-1)
        q_value = self.net(x)
        return q_value

class EnsembleCritic(nn.Module):
    def __init__(self, args, n_Q=2):
        super(EnsembleCritic, self).__init__()
        self.args = args
        ensemble_Q = [CriticMonolithic(self.args) for _ in range(n_Q)]			
        self.ensemble_Q = nn.ModuleList(ensemble_Q)
        self.n_Q = n_Q

    def forward(self, state, action, goal):
        Q = [self.ensemble_Q[i](state, action, goal) for i in range(self.n_Q)]
        Q = torch.cat(Q, dim=-1)
        return Q

class EnsembleContinuousMLPCritic(nn.Module):
    def __init__(
        self, args, ensemble_size=2, ortho_init = True, output_gain= None
    ):
        super().__init__()
        self.args = args
        self.state_dim = self.args.dim_state
        self.goal_dim = self.args.dim_goal
        self.act_dim = self.args.dim_action
        input_dim = self.state_dim+ self.act_dim + self.goal_dim
        self.ensemble_size = ensemble_size
        if self.ensemble_size > 1:
            self.mlp = EnsembleMLP(input_dim= input_dim, output_dim=1, ensemble_size=self.ensemble_size, hidden_layers=[256,256])
        else:
            self.mlp = MLP(input_dim= input_dim, output_dim=1, hidden_layers=[256,256])
        self.ortho_init = ortho_init
        self.output_gain = output_gain
        self.reset_parameters()

    def reset_parameters(self):
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))  # use the fact that True converts to 1.0
            if self.output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=self.output_gain))
        
    def forward(self, states, actions, goals):
        h = torch.cat((states, actions, goals), dim=-1)
        #print("q.shape:", self.q(h).shape)
        q = self.mlp(h).squeeze(-1)  # Remove the last dim
        if self.ensemble_size == 1:
            q = q.unsqueeze(0)  # add in the ensemble dim
        return q
    
class Value(nn.Module):
    def __init__(self, args):
        super(Value, self).__init__()
        self.args = args
        self.fc1 = nn.Linear( self.args.dim_state + self.args.dim_goal, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_out(x)

        return q_value
        
class MLPValue(nn.Module):
    def __init__(self, args, ensemble_size=1, ortho_init = True, output_gain= None):
        super().__init__()
        self.args = args
        self.state_dim = self.args.dim_state
        self.goal_dim = self.args.dim_goal
        self.act_dim = self.args.dim_action
        self.ensemble_size = ensemble_size

        input_dim = self.state_dim + self.goal_dim
        if self.ensemble_size > 1:
            self.mlp = EnsembleMLP(input_dim= input_dim, output_dim=1, ensemble_size=self.ensemble_size, hidden_layers=[256,256])
        else:
            self.mlp = MLP(input_dim= input_dim, output_dim=1,hidden_layers=[256,256])
        self.ortho_init = ortho_init
        self.output_gain = output_gain
        self.reset_parameters()

    def reset_parameters(self):
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))  # use the fact that True converts to 1.0
            if self.output_gain is not None:
                self.mlp.last_layer.apply(partial(weight_init, gain=self.output_gain))

    def forward(self, states, goals):
        h = torch.cat((states, goals), dim=-1)
        #print("v.shape:", self.mlp(h).shape)
        v = self.mlp(h).squeeze(-1)
        #print("after squeeze v.shape:", self.mlp(h).squeeze(-1).shape)
        if self.ensemble_size == 1:
           v = v.unsqueeze(0)
        return v
    
class NormalPolicy(nn.Module):	
	def __init__(self, goal_dim, hidden_dims=[256, 256]):	
		super(NormalPolicy, self).__init__()	
		fc = [nn.Linear(goal_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc = nn.Sequential(*fc)

		self.mean = nn.Linear(hidden_dims[-1], goal_dim)	
		self.log_scale = nn.Linear(hidden_dims[-1], goal_dim)	
		self.LOG_SCALE_MIN = -20	
		self.LOG_SCALE_MAX = 2	

	def forward(self,  goal):	
		h = self.fc(goal)	
		mean = self.mean(h)
		scale = self.log_scale(h).clamp(min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX).exp()	
		distribution = torch.distributions.Normal(mean, scale)
		return distribution
    
class MlpNetwork(nn.Module):
    """
    Basic feedforward network uesd as building block of more complex policies
    """
    def __init__(self, input_dim, output_dim=1, activ=f.relu, output_nonlinearity=None, n_units=64):
        super(MlpNetwork, self).__init__()
        # n_units = 512
        self.h1 = nn.Linear(input_dim, n_units)
        self.h2 = nn.Linear(n_units, n_units)
        # self.h3 = nn.Linear(n_units, n_units)
        self.out = nn.Linear(n_units, output_dim)
        self.out_nl = output_nonlinearity
        self.activ = activ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of network
        :param x:
        :return:
        """
        x = self.activ(self.h1(x))
        x = self.activ(self.h2(x))
        # x = self.activ(self.h3(x))
        x = self.out(x)
        if self.out_nl is not None:
            if self.out_nl == f.log_softmax:
                x = f.log_softmax(x, dim=-1)
            else:
                x = self.out_nl(x)
        return x
        
class DiscreteMLPDistance(nn.Module):
    def __init__(
        self,
        args,
        bins: int = 50,
        ortho_init: bool = True,
    ):
        super(DiscreteMLPDistance, self).__init__()
        self._bins = bins
        self.max_action = args.max_action
        dim_state  = args.dim_state
        dim_hidden = args.dim_hidden
        dim_goal   = args.dim_goal

        self.net = nn.Sequential(
            nn.Linear(dim_state+dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, self.bins)
        )

        self.ortho_init = ortho_init
        self.reset_parameters()

    def reset_parameters(self):
        if self.ortho_init:
            self.apply(partial(weight_init, gain=float(self.ortho_init)))  # use the fact that True converts to 1.0

    @property
    def bins(self):
        return self._bins

    def forward(self, s,g):
        x = torch.cat([s,  g], -1)
        v = self.net(x)
        v = v.unsqueeze(0)  # add in the ensemble dim
        return v
    

