import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import weights_init


class Actor(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            network_size: int,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        self.actor_fc1 = nn.Linear(in_dim, network_size)
        self.actor_ln1 = nn.LayerNorm(network_size)
        self.actor_fc2 = nn.Linear(network_size, network_size)
        self.actor_ln2 = nn.LayerNorm(network_size)
        self.actor_out = nn.Linear(network_size, out_dim)

        weights_init(self.actor_fc1, init_type='kaiming')
        weights_init(self.actor_fc2, init_type='kaiming')
        weights_init(self.actor_out, init_type='orthogonal')

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.actor_ln1(self.actor_fc1(state)))
        x = F.relu(self.actor_ln2(self.actor_fc2(x)))
        action = self.actor_out(x)

        return action


class Critic(nn.Module):
    def __init__(
            self,
            in_dim: int,
            network_size: int,
    ):
        """Initialize."""
        super(Critic, self).__init__()

        self.critic_fc1 = nn.Linear(in_dim, network_size)
        self.critic_ln1 = nn.LayerNorm(network_size)
        self.critic_fc2 = nn.Linear(network_size, network_size)
        self.critic_ln2 = nn.LayerNorm(network_size)
        self.critic_out = nn.Linear(network_size, 1)

        weights_init(self.critic_fc1, init_type='kaiming')
        weights_init(self.critic_fc2, init_type='kaiming')
        weights_init(self.critic_out, init_type='orthogonal')

    def forward(
            self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.critic_ln1(self.critic_fc1(x)))
        x = F.relu(self.critic_ln2(self.critic_fc2(x)))
        value = self.critic_out(x)

        return value
