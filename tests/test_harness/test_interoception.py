"""Tests for synthetic interoception — no external dependencies."""

import pytest

from harness.environments.interoception import (
    InteroceptionProfile,
    SyntheticInteroception,
)


class TestSyntheticInteroception:
    def test_step_returns_all_keys(self):
        intero = SyntheticInteroception()
        state = intero.step()
        assert "battery" in state
        assert "cpu_temp" in state
        assert "joint_health" in state

    def test_battery_drains_over_time(self):
        intero = SyntheticInteroception()
        initial = intero.battery_level
        for _ in range(100):
            intero.step()
        assert intero.battery_level < initial

    def test_battery_never_negative(self):
        profile = InteroceptionProfile(battery_drain_per_step=10.0)
        intero = SyntheticInteroception(profile)
        for _ in range(100):
            intero.step()
        assert intero.battery_level >= 0.0

    def test_battery_format(self):
        intero = SyntheticInteroception()
        state = intero.step()
        assert state["battery"].startswith("battery: ")
        assert state["battery"].endswith("%")

    def test_cpu_temp_format(self):
        intero = SyntheticInteroception()
        state = intero.step()
        assert state["cpu_temp"].startswith("cpu: ")
        assert state["cpu_temp"].endswith("C")

    def test_joint_health_format(self):
        intero = SyntheticInteroception()
        state = intero.step()
        assert state["joint_health"].startswith("joints: ")
        status = state["joint_health"].replace("joints: ", "")
        assert status in ("nominal", "degraded", "warning", "critical")

    def test_joint_degradation(self):
        profile = InteroceptionProfile(
            joint_degradation_rate=0.05, joint_failure_prob=0.0
        )
        intero = SyntheticInteroception(profile)
        for _ in range(15):
            intero.step()
        assert intero.joint_health < 1.0

    def test_reset(self):
        intero = SyntheticInteroception()
        for _ in range(50):
            intero.step()
        intero.reset()
        assert intero.battery_level == 100.0
        assert intero.joint_health == 1.0

    def test_custom_profile(self):
        profile = InteroceptionProfile(
            battery_start=50.0,
            battery_drain_per_step=0.5,
            cpu_base_temp=60.0,
        )
        intero = SyntheticInteroception(profile)
        assert intero.battery_level == 50.0
        state = intero.step()
        # CPU base is higher
        temp_str = state["cpu_temp"].replace("cpu: ", "").replace("C", "")
        assert float(temp_str) >= 50.0  # at least close to base

    def test_deterministic_with_seed(self):
        """With same seed, should produce same sequence."""
        import random

        results_a = []
        random.seed(42)
        intero_a = SyntheticInteroception()
        for _ in range(10):
            results_a.append(intero_a.step())

        results_b = []
        random.seed(42)
        intero_b = SyntheticInteroception()
        for _ in range(10):
            results_b.append(intero_b.step())

        assert results_a == results_b
