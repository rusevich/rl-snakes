from itertools import accumulate
from collections import deque
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_snakes.env import SnakeEnv
from rl_snakes.logger import Logger

class VPG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.current_env = None

        HIDDEN_SIZE = 128

        obs_space, act_space = 11, 3 # change to variables instead of magic numbers

        self.policy_net = nn.Sequential(
            nn.Linear(obs_space, HIDDEN_SIZE),
            nn.GELU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE//2),
            nn.GELU(),
            nn.Linear(HIDDEN_SIZE//2, act_space) # add softmax?
        )
        self.policy_optim = torch.optim.Adam(params=self.policy_net.parameters())

        self.value_net = nn.Sequential(
            nn.Linear(obs_space, HIDDEN_SIZE),
            nn.GELU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE//2),
            nn.GELU(),
            nn.Linear(HIDDEN_SIZE//2, 1) 
        )
        self.value_optim = torch.optim.Adam(params=self.value_net.parameters())

        self.logger = Logger(stacked=["death_causes"], main="episode_returns")

    def take_action(self, obs):
        obs = torch.from_numpy(obs)
        logits = self.policy_net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().item(), logits

    def get_advantages(self, rewards, observations):
        rewards_to_go = torch.tensor(rewards).flip(0).cumsum(0).flip(0)
        states_values = self.value_net(torch.from_numpy(np.stack(observations))).squeeze(-1)
        advantages = [r - v for r, v in zip(rewards_to_go, states_values)]
        return advantages

    def policy_step(self, actions, logits, advantages):
        """
        actions - [N]
        logits - [N, action_space]
        advantages - [N]
        """
        self.policy_optim.zero_grad()

        policy_distribution = torch.distributions.Categorical(logits=torch.stack(logits, dim=0))
        action_log_probs = policy_distribution.log_prob(torch.tensor(actions))
        advantages = torch.stack(advantages, dim=0).detach() # detach is important here, for not causing double .backward()
        surrogate_loss = -torch.mean(action_log_probs * advantages)
        surrogate_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), float("inf"))
    
        self.policy_optim.step()

        ent = policy_distribution.entropy()

        policy_stats = {}
        policy_stats["entropy"] = ent.mean().item()
        policy_stats["grad_norm"] = grad_norm.item()
        policy_stats["policy_loss"] = surrogate_loss.item()

        return policy_stats

    def value_step(self, advantages):
        self.value_optim.zero_grad()
        advantages = torch.stack(advantages, dim=0)
        loss = torch.mean(advantages ** 2)
        loss.backward()
        self.value_optim.step()

        value_stats = {}
        value_stats["value_loss"] = loss.item()

        return value_stats

    def train(self):
        returns_window = deque(maxlen=self.args.ma_window)
        death_cause_window = deque(maxlen=self.args.ma_window)
        
        episode = 0
        while self.args.total_episodes == -1 or episode < self.args.total_episodes:
            # used for policy optimization
            batch_advantages = []
            batch_actions = []
            batch_logits = []

            # used for logging
            episodes_len = []
            episodes_causes_of_death = []
            episodes_scores = []
            episodes_total_rewards = []
            episodes_time = []

            for _ in range(self.args.batch_size):
                start = time.time()

                render_mode = None
                if episode % self.args.render_frequency == 0:
                    render_mode = "human"

                env = SnakeEnv(size=self.args.size, render_mode=render_mode)
                self.current_env = env

                obs, _ = env.reset()
                
                observations, actions, action_logits, rewards = [], [], [], []      
                observations.append(obs)    

                while True:
                    act, logits = self.take_action(obs)
                    obs, rew, term, trun, info = env.step(act)

                    observations.append(obs)
                    actions.append(act)
                    rewards.append(rew)
                    action_logits.append(logits)
                
                    if term or trun:
                        episode += 1
                        end = time.time()
                        elapsed_time = end - start

                        episodes_len.append(len(rewards))
                        episodes_causes_of_death.append(info["cause"])
                        episodes_scores.append(info["score"])
                        episodes_total_rewards.append(sum(rewards))
                        episodes_time.append(elapsed_time)

                        break

                batch_actions += actions
                batch_logits += action_logits
                batch_advantages += self.get_advantages(rewards, observations[:-1])           

                env.close()

            returns_window.extend(episodes_total_rewards)
            death_cause_window.extend(episodes_causes_of_death)

            policy_stats = self.policy_step(batch_actions, batch_logits, batch_advantages)
            value_stats = self.value_step(batch_advantages)

            episode_returns = {
                "mean": np.mean(episodes_total_rewards),
                "min": np.min(episodes_total_rewards),
                "max": np.max(episodes_total_rewards),
                "ma": np.mean(returns_window),
            }

            death_causes = {
                "wall": death_cause_window.count("wall"),
                "self": death_cause_window.count("self"),
                "timeout": death_cause_window.count("timeout"),
                "won": death_cause_window.count("won"),
            }

            self.logger.log(
                episode,
                episode_returns=episode_returns,
                death_causes=death_causes,
                entropy=policy_stats["entropy"],
                grad_norm=policy_stats["grad_norm"],
                value_loss=value_stats["value_loss"]
            )

            if episode % self.args.checkpoint_frequency == 0:
                torch.save(self.policy_net, self.logger.run_dir / "latest_checkpoint.pt")
