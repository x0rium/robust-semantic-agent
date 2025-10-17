"""
Episode Data Structure and Logging
Feature: 002-full-prototype
Task: T040

Tracks episode data for analysis and reporting.

References:
- FR-013: JSONL episode logging
- SC-012: Test coverage requirement
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EpisodeStep:
    """Single timestep data."""

    timestep: int
    state: list  # True state (if available)
    action: list
    observation: list
    reward: float
    info: dict = field(default_factory=dict)


class Episode:
    """
    Episode trajectory with metadata.

    Stores full trajectory for analysis and logging.

    Attributes:
        episode_id: Unique episode identifier
        config_hash: Configuration hash for reproducibility
        steps: List of EpisodeStep objects
        total_return: Cumulative reward
        done: Episode completion flag

    Methods:
        add_step(state, action, obs, reward, info): Record timestep
        compute_return(): Calculate total return
        to_jsonl(): Serialize to JSONL format
        save(path): Write to file

    References:
        - FR-013: Episode logging specification
        - Task T040: Episode implementation
    """

    def __init__(self, episode_id: int, config_hash: str = ""):
        """
        Initialize episode.

        Args:
            episode_id: Unique episode number
            config_hash: Configuration identifier
        """
        self.episode_id = episode_id
        self.config_hash = config_hash
        self.steps: list[EpisodeStep] = []
        self.total_return = 0.0
        self.done = False

    def add_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        observation: np.ndarray,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        """
        Record episode step.

        Args:
            state: True state (2,)
            action: Control input (2,)
            observation: Noisy observation (2,)
            reward: Step reward
            info: Additional metadata
        """
        # Convert numpy arrays in info dict to lists for JSON serialization
        serializable_info = {}
        for key, value in info.items():
            if isinstance(value, np.ndarray):
                serializable_info[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_info[key] = value.item()
            else:
                serializable_info[key] = value

        step = EpisodeStep(
            timestep=len(self.steps),
            state=state.tolist(),
            action=action.tolist(),
            observation=observation.tolist(),
            reward=float(reward),
            info=serializable_info,
        )

        self.steps.append(step)
        self.total_return += reward

    def compute_return(self, discount: float = 1.0) -> float:
        """
        Compute discounted return.

        Args:
            discount: Discount factor γ ∈ [0, 1]

        Returns:
            Discounted cumulative reward
        """
        ret = 0.0
        for t, step in enumerate(self.steps):
            ret += (discount**t) * step.reward
        return ret

    def to_dict(self) -> dict[str, Any]:
        """Convert episode to dictionary."""
        return {
            "episode_id": self.episode_id,
            "config_hash": self.config_hash,
            "total_return": self.total_return,
            "num_steps": len(self.steps),
            "steps": [asdict(step) for step in self.steps],
        }

    def to_jsonl(self) -> str:
        """
        Serialize to JSONL format (one episode per line).

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict())

    def save(self, path: Path) -> None:
        """
        Save episode to JSONL file.

        Args:
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a") as f:
            f.write(self.to_jsonl() + "\n")

    def __repr__(self) -> str:
        return (
            f"Episode(id={self.episode_id}, "
            f"steps={len(self.steps)}, "
            f"return={self.total_return:.2f})"
        )
