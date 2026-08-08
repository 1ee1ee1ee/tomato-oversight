"""Small, dependency-light Double DQN implementation for grower policies."""

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
    hidden_sizes: tuple[int, ...] = (128, 128)
    learning_rate: float = 1e-4
    gamma: float = 0.999
    batch_size: int = 256
    replay_capacity: int = 200_000
    learning_starts: int = 5_000
    train_frequency: int = 4
    target_update_interval: int = 2_000
    gradient_clip_norm: float = 10.0


class ReplayBuffer:
    """Fixed-size replay memory stored as NumPy arrays."""

    def __init__(self, capacity: int, observation_size: int, seed: int) -> None:
        self.capacity = capacity
        self.observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.dones[index] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        indices = self.rng.integers(self.size, size=batch_size)
        return (
            torch.as_tensor(self.observations[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_observations[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device),
        )


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_sizes: tuple[int, ...]) -> None:
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
    """Independent online/target networks using the Double DQN target."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: DDQNConfig,
        seed: int,
        device: str = "auto",
    ) -> None:
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
        self.online_network = QNetwork(observation_size, action_size, config.hidden_sizes).to(self.device)
        self.target_network = QNetwork(observation_size, action_size, config.hidden_sizes).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=config.learning_rate)
        self.gradient_steps = 0

    def select_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_size))
        with torch.no_grad():
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            return int(self.online_network(observation_tensor).argmax(dim=1).item())

    def train_step(self, replay_buffer: ReplayBuffer) -> float:
        observations, actions, rewards, next_observations, dones = replay_buffer.sample(
            self.config.batch_size, self.device
        )

        current_q = self.online_network(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: online network selects; target network evaluates.
            next_actions = self.online_network(next_observations).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_observations).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.config.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()

        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        return float(loss.item())

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "config": asdict(self.config),
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "gradient_steps": self.gradient_steps,
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, device: str = "auto", seed: int = 0
    ) -> tuple["DoubleDQNAgent", dict[str, Any]]:
        if device == "auto":
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            map_location = device
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        config_data = checkpoint["config"]
        config_data["hidden_sizes"] = tuple(config_data["hidden_sizes"])
        agent = cls(
            observation_size=int(checkpoint["observation_size"]),
            action_size=int(checkpoint["action_size"]),
            config=DDQNConfig(**config_data),
            seed=seed,
            device=device,
        )
        agent.online_network.load_state_dict(checkpoint["online_network"])
        agent.target_network.load_state_dict(checkpoint["target_network"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.gradient_steps = int(checkpoint["gradient_steps"])
        return agent, dict(checkpoint.get("metadata", {}))
