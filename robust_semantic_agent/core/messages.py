"""
Message and Source Trust Management
Feature: 002-full-prototype
Tasks: T030, T031

Implements:
- Message dataclass for exogenous claims
- SourceTrust class for reliability tracking with Beta-Bernoulli updates

References:
- docs/theory.md §3: Message integration
- FR-003: Message multipliers specification
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .semantics import BelnapValue


@dataclass
class Message:
    """
    Exogenous message with claim, source, and Belnap status.

    Attributes:
        claim: Human-readable claim description
        source: Source identifier
        value: BelnapValue status (⊥, t, f, ⊤)
        A_c: Claim region indicator function: particles → bool array

    Example:
        message = Message(
            claim="x[0] > 0.5",
            source="beacon_1",
            value=BelnapValue.TRUE,
            A_c=lambda particles: particles[:, 0] > 0.5
        )

    References:
        - FR-003: Message integration via soft-likelihood
    """

    claim: str
    source: str
    value: BelnapValue
    A_c: Callable[[np.ndarray], np.ndarray]

    def __repr__(self) -> str:
        return f"Message(claim='{self.claim}', source='{self.source}', value={self.value})"


class SourceTrust:
    """
    Source reliability tracking with Beta-Bernoulli updates.

    Maintains reliability r_s ∈ (0, 1) for a message source.

    Attributes:
        r_s: Current reliability estimate ∈ (0, 1)
        alpha: Beta distribution α parameter (pseudo-count of successes)
        beta: Beta distribution β parameter (pseudo-count of failures)

    Methods:
        logit(): Convert r_s to logit space λ_s = log(r_s / (1 - r_s))
        update(): Bayesian update with success/failure evidence

    References:
        - docs/theory.md §3.2: Source trust dynamics
        - FR-003: Message multiplier M_{c,s,v}(x)
    """

    def __init__(self, r_s: float = 0.7, alpha: float = 7.0, beta: float = 3.0):
        """
        Initialize source trust with prior.

        Args:
            r_s: Initial reliability estimate (default 0.7)
            alpha: Beta prior α (successes pseudo-count)
            beta: Beta prior β (failures pseudo-count)

        Note:
            r_s = α / (α + β) for Beta(α, β) prior
        """
        self.alpha = alpha
        self.beta = beta
        self.r_s = r_s

    def logit(self) -> float:
        """
        Convert reliability to logit space.

        λ_s = log(r_s / (1 - r_s))

        Returns:
            Logit value (unbounded real)

        Note:
            Clips r_s to avoid log(0) or division by zero.
        """
        r_clipped = np.clip(self.r_s, 1e-6, 1 - 1e-6)
        return np.log(r_clipped / (1 - r_clipped))

    def update(self, success: bool, weight: float = 1.0) -> None:
        """
        Bayesian update of source reliability.

        Beta-Bernoulli update:
        - α' = α + weight (if success)
        - β' = β + weight (if failure)
        - r_s' = α' / (α' + β')

        Args:
            success: Whether source was correct
            weight: Evidence weight (can encode claim complexity)

        References:
            - docs/theory.md §3.2: Trust update dynamics
        """
        if success:
            self.alpha += weight
        else:
            self.beta += weight

        # Update r_s
        self.r_s = self.alpha / (self.alpha + self.beta)

    def __repr__(self) -> str:
        return (
            f"SourceTrust(r_s={self.r_s:.3f}, "
            f"α={self.alpha:.1f}, β={self.beta:.1f}, "
            f"λ_s={self.logit():.3f})"
        )
