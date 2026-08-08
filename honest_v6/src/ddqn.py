"""Dependency-light Double DQN used by the Colab training script."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DDQNConfig:
    hidden_sizes: tuple[int, ...] = (256, 256)
    learning_rate: float = 1e-4
    gamma: float = 0.9995
    batch_size: int = 256
    replay_capacity: int = 300_000
    learning_starts: int = 10_000
    train_frequency: int = 4
    target_update_interval: int = 2_000
    gradient_clip_norm: float = 10.0


class ReplayBuffer:
    def __init__(self, capacity: int, observation_size: int, seed: int):
        self.capacity = capacity
        self.observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.bootstrap_discounts = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, observation, action, reward, next_observation, bootstrap_discount) -> None:
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.bootstrap_discounts[index] = float(bootstrap_discount)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        indices = self.rng.integers(self.size, size=batch_size)
        return (
            torch.as_tensor(self.observations[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_observations[indices], device=device),
            torch.as_tensor(self.bootstrap_discounts[indices], device=device),
        )


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        input_size = observation_size
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.ReLU()))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class DoubleDQNAgent:
    def __init__(self, observation_size: int, action_size: int, config: DDQNConfig, seed: int, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.config = config
        self.observation_size = observation_size
        self.action_size = action_size
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.online = QNetwork(observation_size, action_size, config.hidden_sizes).to(self.device)
        self.target = QNetwork(observation_size, action_size, config.hidden_sizes).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.gradient_steps = 0

    def select_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_size))
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(tensor).argmax(dim=1).item())

    def train_step(self, replay: ReplayBuffer) -> float:
        observations, actions, rewards, next_observations, bootstrap_discounts = replay.sample(
            self.config.batch_size, self.device
        )
        current_q = self.online(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_observations).argmax(dim=1, keepdim=True)
            next_q = self.target(next_observations).gather(1, next_actions).squeeze(1)
            target_q = rewards + bootstrap_discounts * next_q
        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "config": asdict(self.config),
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "gradient_steps": self.gradient_steps,
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "auto", seed: int = 0):
        map_location = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        config_data = checkpoint["config"]
        config_data["hidden_sizes"] = tuple(config_data["hidden_sizes"])
        agent = cls(
            int(checkpoint["observation_size"]),
            int(checkpoint["action_size"]),
            DDQNConfig(**config_data),
            seed,
            device,
        )
        agent.online.load_state_dict(checkpoint["online"])
        agent.target.load_state_dict(checkpoint["target"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.gradient_steps = int(checkpoint["gradient_steps"])
        return agent, dict(checkpoint.get("metadata", {}))
