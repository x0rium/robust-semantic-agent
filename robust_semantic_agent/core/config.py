"""
Configuration Management
Feature: 002-full-prototype
Task: T039

Load and validate YAML configuration files.

References:
- FR-012: Hyperparameters via YAML configs
- configs/default.yaml: Master configuration
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EnvConfig:
    """Environment configuration."""

    state_dim: int = 2
    action_dim: int = 2
    obstacle_radius: float = 0.3
    obstacle_center: list = field(default_factory=lambda: [0.0, 0.0])
    goal_region: list = field(default_factory=lambda: [0.8, 0.8])
    goal_radius: float = 0.1
    observation_noise: float = 0.1
    max_action: float = 0.15


@dataclass
class RiskConfig:
    """Risk management configuration."""

    mode: str = "cvar"
    alpha: float = 0.1
    nested: bool = False


@dataclass
class SafetyConfig:
    """Safety filter configuration."""

    cbf: bool = True
    barrier_alpha: float = 0.5
    qp_max_iter: int = 50
    qp_slack: float = 1e-3
    slack_penalty: float = 1000.0


@dataclass
class BeliefConfig:
    """Belief tracking configuration."""

    particles: int = 5000
    resample_threshold: float = 0.5
    process_noise: float = 0.01


@dataclass
class QueryConfig:
    """Query action configuration."""

    enabled: bool = False
    cost: float = 0.2
    delta_star: float = 0.15


@dataclass
class Configuration:
    """
    Master configuration object.

    Loads from YAML files and provides structured access.

    Attributes:
        seed: Random seed for reproducibility
        discount: Discount factor γ
        env: Environment configuration
        risk: Risk management configuration
        safety: Safety filter configuration
        belief: Belief tracking configuration
        query: Query action configuration

    Methods:
        from_yaml(path): Load configuration from YAML file
        validate(): Check configuration validity

    References:
        - FR-012: Configuration specification
        - configs/default.yaml: Default parameters
    """

    seed: int = 42
    discount: float = 0.98
    horizon: int = 50

    env: EnvConfig = field(default_factory=EnvConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    query: QueryConfig = field(default_factory=QueryConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Configuration":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            Configuration object

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Create configuration object
        config = cls()

        # Load top-level parameters
        config.seed = data.get("seed", 42)
        config.discount = data.get("discount", 0.98)
        config.horizon = data.get("horizon", 50)

        # Load nested configs
        if "env" in data:
            config.env = EnvConfig(**data["env"])

        if "risk" in data:
            config.risk = RiskConfig(**data["risk"])

        if "safety" in data:
            safety_data = data["safety"]
            # Flatten QP params
            if "qp" in safety_data:
                qp = safety_data.pop("qp")
                safety_data["qp_max_iter"] = qp.get("max_iter", 50)
                safety_data["qp_slack"] = qp.get("slack", 1e-3)
                safety_data["slack_penalty"] = qp.get("slack_penalty", 1000.0)
            config.safety = SafetyConfig(**safety_data)

        if "belief" in data:
            config.belief = BeliefConfig(**data["belief"])

        if "query" in data:
            config.query = QueryConfig(**data["query"])

        return config

    def validate(self) -> bool:
        """
        Validate configuration values.

        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check basic constraints
        assert 0 < self.discount <= 1.0, f"Invalid discount: {self.discount}"
        assert self.seed >= 0, f"Invalid seed: {self.seed}"
        assert self.belief.particles > 0, f"Invalid particle count: {self.belief.particles}"
        assert 0 < self.risk.alpha <= 1.0, f"Invalid CVaR alpha: {self.risk.alpha}"
        assert self.safety.barrier_alpha > 0, f"Invalid CBF alpha: {self.safety.barrier_alpha}"

        return True

    def __repr__(self) -> str:
        return (
            f"Configuration(seed={self.seed}, γ={self.discount}, "
            f"particles={self.belief.particles}, α={self.risk.alpha})"
        )
