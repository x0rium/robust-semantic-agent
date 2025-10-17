"""
Barrier Function for Forbidden Circle Environment
Feature: 002-full-prototype
Task: T034

Implements barrier function B(x) for circular forbidden zone.

Safe set: S = {x: ||x - c|| ≥ r} (outside circle)
Barrier: h(x) = ||x - c||² - r² (safe when h(x) ≥ 0)

References:
- docs/theory.md §4.3: CBF specification
- FR-007: CBF supermartingale property
"""

import numpy as np


class BarrierFunction:
    """
    Barrier function for circular forbidden zone.

    Safe set S = {x ∈ R²: ||x - center|| ≥ radius}
    Barrier h(x) = ||x - center||² - radius²

    Safety: h(x) ≥ 0

    Attributes:
        radius: Forbidden zone radius
        center: Forbidden zone center (2,)

    Methods:
        evaluate(x): Compute h(x)
        gradient(x): Compute ∇h(x)

    References:
        - FR-007: CBF-QP safety filter
        - SC-001: Zero violations requirement
    """

    def __init__(self, radius: float, center: np.ndarray):
        """
        Initialize barrier function.

        Args:
            radius: Forbidden circle radius (> 0)
            center: Circle center coordinates (2,)
        """
        self.radius = radius
        self.center = np.array(center)

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate barrier function h(x).

        h(x) = ||x - c||² - r²

        Args:
            x: State (2,)

        Returns:
            Barrier value (safe if ≥ 0, unsafe if < 0)
        """
        return np.dot(x - self.center, x - self.center) - self.radius**2

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient ∇h(x).

        ∇h(x) = 2(x - c)

        Args:
            x: State (2,)

        Returns:
            Gradient (2,)

        Note:
            For 2D integrator dynamics (ẋ = u), Lie derivatives are:
            - Lfh(x) = 0 (no drift)
            - Lgh(x) = ∇h(x)
        """
        return 2.0 * (x - self.center)

    def __repr__(self) -> str:
        return f"BarrierFunction(radius={self.radius}, center={self.center})"
