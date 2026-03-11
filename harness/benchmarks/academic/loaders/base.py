from typing import Iterator, Protocol

from harness.benchmarks.academic.trajectory import BenchmarkSample


class DatasetLoader(Protocol):
    """Protocol for loading benchmark datasets."""

    def load(self) -> Iterator[BenchmarkSample]:
        """Yield BenchmarkSamples, grouped by scene for efficiency."""
        ...

    @property
    def name(self) -> str:
        """Dataset name for reporting."""
        ...
