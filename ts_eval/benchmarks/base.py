from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, Sequence, Union

Series = Sequence[float]
MultiSeries = Sequence[Series]


@dataclass
class QAItem:

    id: str
    series: Union[Series, MultiSeries]
    question: str
    gold: str
    scoring_type: str
    options: Union[list[str], dict[str, str], None] = None
    task_type: str | None = None
    domain: str | None = None
    source_benchmark: str = ""
    n_variates: int = field(default=1, init=False)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        first = self.series[0] if len(self.series) else None
        self.n_variates = len(self.series) if isinstance(first, (list, tuple)) else 1


class BenchmarkAdapter(Protocol):

    name: str
    license: str

    def load(self, split: str = "test", limit: int | None = None) -> Iterator[QAItem]:
        ...
