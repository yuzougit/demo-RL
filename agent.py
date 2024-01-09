import logging
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.tensorboard import SummaryWriter

from satellite_engine import Satellite
from networks import Actor, Critic
from prioritized_buffers import ReplayBuffer
from train_utils import Quaternions2EulerAngles, RotationMatrix2Quaternions
from train_utils import RunningMeanStd, save_to_pkl, load_from_pkl, GaussianNoise
from train_utils import SixD2RotationMatrix

logging.basicConfig(filename='log_info.log', filemode='w', level=logging.INFO)


class DDPGAgent:
    """DDPG
    Attributes:
        n_timesteps (int): the number of timesteps for the simulation
        iterations (int): the number of iterations for the simulation
        network_size (int): the number of neurons in each layer
        learning_rate_actor (float): the learning rate for the actor network
        learning_rate_critic (float): the learning rate for the critic network
        memory_size (int): the size of the replay buffer
        batch_size (int): the batch size for training
        pretrain_step (int): the number of pretrain steps
        gamma (float): the discount factor
        initial_random_steps (int): the number of initial random steps
        weight_decay (float): the weight decay for the optimizer
        tau (float): the soft update rate
        path_save (str): the path to save the model
        path_load (str): the path to load the model
        path_tensorboard (str): the path to save tensorboard
    """

    def __init__(
            self,
            demo: list,
            n_timesteps: int,
            iterations: int,
            network_size: int,
            learning_rate_actor: float,
            learning_rate_critic: float,
            memory_size: int,
            batch_size: int,
            pretrain_step: int,
            gamma: float = 0.98,
            initial_random_steps: int = 1e4,
            weight_decay: float = 1e-5,
            tau: float = 1e-4,

            path_save: str = '',
            path_load: str = '',
            path_tensorboard: str = '',
    ):
        """Initialize."""
        self.n_timesteps = n_timesteps
        self.network_size = network_size
        self.iterations = iterations
        self.pretrain_step = pretrain_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.initial_random_steps = initial_random_steps

        # loss parameters
        self.tau = tau

        self.path_save = path_save
        self.path_load = path_load

        # environment setup
        game = Satellite(0, 0, 0, 0, 1, 650, n_timesteps)
        orbits, time_interval, self.lat, self.lon, self.alt, self.B_xyz = game.Orbit_propagation()
        _, state_RL, _, _, _ = game.initialize_sat(self.lat, self.lon, self.alt, self.B_xyz)
        obs_dim = len(state_RL)
        action_dim = 3

        self.action_max = 1
        self.action_min = -1
        self.action_clip = 1

        self.batch_size_expert = 0
        self.batch_size_agent = self.batch_size

        # agent buffer
        self.memory = ReplayBuffer(obs_dim, action_dim, memory_size)

        # expert buffer
        self.demo = demo
        self.demo_memory = ReplayBuffer(obs_dim, action_dim, len(demo))
        self.demo_memory.extend(demo)

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks
        print("obs_dim, action_dim:", obs_dim, action_dim)
        self.actor = Actor(obs_dim, action_dim, self.network_size).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim, self.network_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim + action_dim, self.network_size).to(self.device)
        self.critic_target = Critic(obs_dim + action_dim, self.network_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        if path_load:
            self.load_model()

        wandb.watch(self.actor, log='all', log_freq=100)
        wandb.watch(self.actor_target, log='all', log_freq=100)
        wandb.watch(self.critic, log='all', log_freq=100)
        wandb.watch(self.critic_target, log='all', log_freq=100)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=learning_rate_actor,
            weight_decay=weight_decay
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=learning_rate_critic,
            weight_decay=weight_decay
        )

        # gaussian noise
        self.exploration_noise = GaussianNoise(
            action_dim=action_dim, min_sigma=1e-5, max_sigma=1e-3, decay_period=n_timesteps)

        # transition to store in memory
        self.transition = list()

        # train steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

        # tensorboard
        self.writer = SummaryWriter(path_tensorboard)

        # running mean and std
        self.obs_rms = RunningMeanStd(shape=obs_dim)
        self.return_rms = RunningMeanStd(shape=())

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = np.array([np.random.uniform(self.action_min, self.action_max) / 100,
                                        np.random.uniform(self.action_min, self.action_max) / 100,
                                        np.random.uniform(self.action_min, self.action_max) / 100])

        else:
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            ).detach().cpu().numpy()

        # add gaussian noise for exploration during training
        if not self.is_test:
            noise_mag = self.action_clip
            noise = self.exploration_noise.sample(self.total_step % self.n_timesteps)
            selected_action = np.clip(selected_action + noise_mag * noise, -self.action_clip, self.action_clip)

        self.transition = [state, selected_action]

        return selected_action

    def update_model(self, is_pretrain=False) -> torch.Tensor:
        """Update the model by gradient descent."""
        device = self.device

        if is_pretrain:
            # if pretrain sample from expert
            samples = self.demo_memory.sample_batch(self.batch_size)
            state = torch.FloatTensor(samples["obs"]).to(device)
            next_state = torch.FloatTensor(samples["next_obs"]).to(device)
            action = torch.FloatTensor(samples["acts"]).to(device)
            reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
            done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        else:
            # sample from both agent and expert
            demo = self.demo_memory.sample_batch(self.batch_size_expert)
            agent = self.memory.sample_batch(self.batch_size_agent)

            state = torch.cat((torch.FloatTensor(demo["obs"]).to(device),
                               torch.FloatTensor(agent["obs"]).to(device)), dim=0)
            next_state = torch.cat((torch.FloatTensor(demo["next_obs"]).to(device),
                                    torch.FloatTensor(agent["next_obs"]).to(device)), dim=0)
            action = torch.cat((torch.FloatTensor(demo["acts"]).to(device),
                                torch.FloatTensor(agent["acts"]).to(device)), dim=0)
            reward = torch.cat((torch.FloatTensor(demo["rews"].reshape(-1, 1)).to(device),
                                torch.FloatTensor(agent["rews"].reshape(-1, 1)).to(device)), dim=0)
            done = torch.cat((torch.FloatTensor(demo["done"].reshape(-1, 1)).to(device),
                              torch.FloatTensor(agent["done"].reshape(-1, 1)).to(device)), dim=0)

        # Train Critic
        masks = 1 - done
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        target_return = reward + self.gamma * next_value * masks

        values = self.critic(state, action)
        critic_loss = F.mse_loss(values, target_return)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Train actor
        actor_loss = torch.mean(-self.critic(state, self.actor(state)))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        # Target update
        self._target_soft_update()

        wandb.log({'actor_loss': -actor_loss})
        wandb.log({'critic_loss': critic_loss})

        self.writer.add_scalar('actor_loss', -actor_loss, self.total_step)
        self.writer.add_scalar('critic_loss', critic_loss, self.total_step)

        hs = open(self.path_save + "actor_loss_" + str(self.iterations) + ".txt", "a")
        hs.write(str(-np.round_(actor_loss.data.item(), 5)))
        hs.write("\n")
        hs.close()

        hs = open(self.path_save + "critic_loss_" + str(self.iterations) + ".txt", "a")
        hs.write(str(np.round(critic_loss.data.item(), 5)))
        hs.write("\n")
        hs.close()

        return actor_loss.data, critic_loss.data

    def _pretrain(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Pretraining steps."""
        actor_losses = []
        critic_losses = []
        print("Pre-Train %d step." % self.pretrain_step)
        for step in range(1, self.pretrain_step + 1):
            self.total_step += 1
            if step % 1000 == 0:
                print("pretrain: step", step, "    iterations:", self.iterations)
            actor_loss, critic_loss = self.update_model(True)
            actor_losses.append(actor_loss.data)
            critic_losses.append(critic_loss.data)

        print("Pre-Train Complete!\n")
        return actor_losses, critic_losses

    def train(self, n_episode: int):
        """Train the agent."""
        self.is_test = False

        best_score = 0
        actor_losses, critic_losses = [], []

        n_timesteps = self.n_timesteps
        game = Satellite(0, 0, 0, 0, 1, 650, n_timesteps)
        control_trigger = 0
        pid = os.getpid()

        if self.demo:
            output = self._pretrain()
            actor_losses.extend(output[0])
            critic_losses.extend(output[1])

        for episode in range(1, n_episode + 1):
            score = 0
            score_episode = []
            state_solver, state_RL, controller_actions, action_set, time_overall = game.initialize_sat(
                self.lat, self.lon, self.alt, self.B_xyz
            )
            controller_vec = np.array([controller_actions[0], controller_actions[1], controller_actions[2]])
            self.action_clip = np.linalg.norm(controller_vec)

            for time_record in range(1, n_timesteps + 1):
                self.total_step += 1
                action = self.select_action(state_RL)
                next_state_RL, reward, next_state_solver, done, _, controller_actions, control_trigger = game.get_new_state(
                    state_solver, action, time_record - 1, self.lat, self.lon, self.B_xyz, control_trigger)

                if not self.is_test:
                    if not control_trigger:
                        self.transition += [reward, next_state_RL, done]
                    else:
                        self.transition = [state_RL, controller_actions, reward, next_state_RL, done]
                    self.memory.store(*self.transition)

                score += reward
                rot_6d = np.array([[state_RL[3], state_RL[4]], [state_RL[5], state_RL[6]], [state_RL[7], state_RL[8]]])
                rot_mat = SixD2RotationMatrix(rot_6d)
                quat_vec = RotationMatrix2Quaternions(rot_mat)
                euler_vec = Quaternions2EulerAngles(quat_vec)
                omega_vec = np.array([state_RL[9], state_RL[10], state_RL[11]])
                euler_norm = np.linalg.norm(euler_vec)
                omega_norm = np.linalg.norm(omega_vec)
                controller_vec = np.array([controller_actions[0], controller_actions[1], controller_actions[2]])
                self.action_clip = np.linalg.norm(controller_vec)

                wandb.log({'cumulated_reward': score})
                wandb.log({'reward': reward})
                wandb.log({'control_trigger': control_trigger})
                wandb.log({'euler': euler_norm})
                wandb.log({'eulerX': euler_vec[0]})
                wandb.log({'eulerY': euler_vec[1]})
                wandb.log({'eulerZ': euler_vec[2]})
                wandb.log({'omega': omega_norm})
                wandb.log({'omegaX': omega_vec[0]})
                wandb.log({'omegaY': omega_vec[1]})
                wandb.log({'omegaZ': omega_vec[2]})

                # if training is ready
                if (
                        len(self.memory) >= self.batch_size
                        and time_record > self.initial_random_steps
                ):
                    self.writer.add_scalar('cumulated_reward', score, self.total_step)
                    self.writer.add_scalar('reward', reward, self.total_step)
                    self.writer.add_scalar('action_clip', self.action_clip, self.total_step)
                    self.writer.add_scalar('euler', euler_norm, self.total_step)
                    self.writer.add_scalar('eulerX', euler_vec[0], self.total_step)
                    self.writer.add_scalar('eulerY', euler_vec[1], self.total_step)
                    self.writer.add_scalar('eulerZ', euler_vec[2], self.total_step)
                    self.writer.add_scalar('omega', omega_norm, self.total_step)
                    self.writer.add_scalar('omegaX', omega_vec[0], self.total_step)
                    self.writer.add_scalar('omegaY', omega_vec[1], self.total_step)
                    self.writer.add_scalar('omegaZ', omega_vec[2], self.total_step)

                state_RL = next_state_RL
                state_solver = next_state_solver

                if time_record == n_timesteps:
                    print('worker:', pid, ' episode: ', episode, '  |score %.1f' % score, '   eulers:',
                          [np.round_((euler_vec[0]), 4), np.round_((euler_vec[1]), 4), np.round_((euler_vec[2]), 4)],
                          '   omegas:',
                          [np.round_((omega_vec[0]), 3), np.round_((omega_vec[1]), 3), np.round_((omega_vec[2]), 3)],
                          'iterations:', self.iterations)
                    score_episode.append(score)
                    avg_score = np.mean(score_episode[-3:])

                    if avg_score > best_score:
                        best_score = avg_score
                        if not self.is_test:
                            self.save_model()

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target """
        tau = self.tau

        for t_param, l_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def load_model(self) -> None:
        """Load model weights """
        self.actor.load_state_dict(torch.load(self.path_load + 'model_actor.tp'))
        self.actor_target.load_state_dict(torch.load(self.path_load + 'model_actor_target.tp'))
        self.critic.load_state_dict(torch.load(self.path_load + 'model_critic.tp'))
        self.critic_target.load_state_dict(torch.load(self.path_load + 'model_critic_target.tp'))
        self.load_replay_buffer()

    def save_model(self) -> None:
        """Save model weights """
        torch.save(self.actor.state_dict(), self.path_save + 'model_actor' + str(self.iterations) + '.tp')
        torch.save(self.actor_target.state_dict(), self.path_save + 'model_actor_target' + str(self.iterations) + '.tp')
        torch.save(self.critic.state_dict(), self.path_save + 'model_critic' + str(self.iterations) + '.tp')
        torch.save(self.critic_target.state_dict(),
                   self.path_save + 'model_critic_target' + str(self.iterations) + '.tp')
        self.save_replay_buffer()

    def save_replay_buffer(self) -> None:
        """Save the replay buffer as a pickle file """
        assert self.memory is not None, "The replay buffer is not defined"
        save_to_pkl(self.path_save + 'replay_buffer' + str(self.iterations) + '.pkl', self.memory)

    def load_replay_buffer(self) -> None:
        """Load a replay buffer from a pickle file """
        self.memory = load_from_pkl(self.path_load + 'replay_buffer' + str(self.iterations) + '.pkl')
        assert isinstance(self.memory, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"
