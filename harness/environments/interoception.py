import math
import random
from dataclasses import dataclass


@dataclass
class InteroceptionProfile:
    """Configuration for synthetic body state generation."""

    battery_start: float = 100.0
    battery_drain_per_step: float = 0.15  # ~660 steps to empty
    battery_noise: float = 0.02
    cpu_base_temp: float = 45.0
    cpu_temp_range: float = 30.0  # fluctuates 45-75C
    cpu_spike_prob: float = 0.05
    joint_degradation_rate: float = 0.001
    joint_failure_prob: float = 0.002


class SyntheticInteroception:
    """Generates randomised but plausible body state each step.

    :param profile: Configuration profile. Uses defaults if ``None``.
    """

    def __init__(self, profile: InteroceptionProfile | None = None):
        self._profile = profile or InteroceptionProfile()
        self._step = 0
        self._battery = self._profile.battery_start
        self._joint_health = 1.0
        self._joint_status = "nominal"

    def step(self) -> dict[str, str]:
        """Advance one step and return body state readings.

        :returns: Dict with keys ``battery``, ``cpu_temp``, ``joint_health``.
            Values are human-readable strings for :meth:`~emem.SpatioTemporalMemory.add_body_state`.
        """
        self._step += 1
        p = self._profile

        # Battery: steady drain with noise, never increases
        noise = random.gauss(0, p.battery_noise * p.battery_start)
        self._battery = max(
            0.0,
            min(
                self._battery,
                self._battery - p.battery_drain_per_step + noise,
            ),
        )

        # CPU temperature: sinusoidal base + random spikes
        base = (
            p.cpu_base_temp
            + (p.cpu_temp_range / 2) * (1 + math.sin(self._step / 50.0)) / 2
        )
        if random.random() < p.cpu_spike_prob:
            base += random.uniform(10, 25)
        cpu_temp = min(base, p.cpu_base_temp + p.cpu_temp_range)

        # Joint health: slow degradation with rare failures
        self._joint_health = max(0.0, self._joint_health - p.joint_degradation_rate)
        if random.random() < p.joint_failure_prob:
            self._joint_health = max(0.0, self._joint_health - 0.1)

        if self._joint_health > 0.8:
            self._joint_status = "nominal"
        elif self._joint_health > 0.5:
            self._joint_status = "degraded"
        elif self._joint_health > 0.2:
            self._joint_status = "warning"
        else:
            self._joint_status = "critical"

        return {
            "battery": f"battery: {self._battery:.0f}%",
            "cpu_temp": f"cpu: {cpu_temp:.0f}C",
            "joint_health": f"joints: {self._joint_status}",
        }

    @property
    def battery_level(self) -> float:
        return self._battery

    @property
    def joint_health(self) -> float:
        return self._joint_health

    def reset(self):
        """Reset to initial state."""
        self._step = 0
        self._battery = self._profile.battery_start
        self._joint_health = 1.0
        self._joint_status = "nominal"
