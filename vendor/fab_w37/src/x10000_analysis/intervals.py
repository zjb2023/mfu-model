from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence


@dataclass(frozen=True, order=True)
class Interval:
    start_ns: int
    end_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.start_ns, int) or not isinstance(self.end_ns, int):
            raise TypeError("interval boundaries must be int nanoseconds")
        if self.end_ns <= self.start_ns:
            raise ValueError("end_ns must be greater than start_ns")

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True, order=True)
class LabeledInterval(Interval):
    label: str


@dataclass(frozen=True)
class AtomicInterval(Interval):
    labels: frozenset[str]


@dataclass(frozen=True)
class ByteAllocation:
    tx_by_label: dict[str, int]
    rx_by_label: dict[str, int]
    unallocated_tx_bytes: int
    unallocated_rx_bytes: int


def overlap_ns(left: Interval, right: Interval) -> int:
    """Return exact integer overlap duration for two half-open intervals."""
    return max(0, min(left.end_ns, right.end_ns) - max(left.start_ns, right.start_ns))


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Merge overlapping or adjacent intervals."""
    ordered = sorted(intervals)
    if not ordered:
        return []
    merged: list[Interval] = [Interval(ordered[0].start_ns, ordered[0].end_ns)]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start_ns <= previous.end_ns:
            merged[-1] = Interval(previous.start_ns, max(previous.end_ns, current.end_ns))
        else:
            merged.append(Interval(current.start_ns, current.end_ns))
    return merged


def atomic_partition(
    window: Interval,
    intervals: Sequence[LabeledInterval],
) -> list[AtomicInterval]:
    """Partition a window into atomic intervals carrying the active label set."""
    clipped: list[LabeledInterval] = []
    boundaries = {window.start_ns, window.end_ns}
    for interval in intervals:
        start = max(window.start_ns, interval.start_ns)
        end = min(window.end_ns, interval.end_ns)
        if end <= start:
            continue
        clipped_interval = LabeledInterval(start, end, interval.label)
        clipped.append(clipped_interval)
        boundaries.add(start)
        boundaries.add(end)

    points = sorted(boundaries)
    result: list[AtomicInterval] = []
    for start, end in zip(points, points[1:]):
        labels = frozenset(
            interval.label
            for interval in clipped
            if interval.start_ns <= start and interval.end_ns >= end
        )
        result.append(AtomicInterval(start, end, labels))
    return result


def _validate_non_overlapping(targets: Sequence[LabeledInterval]) -> None:
    ordered = sorted(targets, key=lambda x: (x.start_ns, x.end_ns, x.label))
    for previous, current in zip(ordered, ordered[1:]):
        if current.start_ns < previous.end_ns:
            raise ValueError("targets must not overlap")


def _largest_remainder_allocation(
    total_bytes: int,
    durations: list[tuple[str, int]],
    denominator_ns: int,
) -> dict[str, int]:
    if total_bytes < 0:
        raise ValueError("byte counters must be non-negative")
    if denominator_ns <= 0:
        raise ValueError("denominator_ns must be positive")

    floors: dict[str, int] = {}
    remainders: list[tuple[Fraction, int, str]] = []
    for order, (key, duration_ns) in enumerate(durations):
        quota = Fraction(total_bytes * duration_ns, denominator_ns)
        floor_value = quota.numerator // quota.denominator
        floors[key] = floor_value
        remainders.append((quota - floor_value, -order, key))

    remaining = total_bytes - sum(floors.values())
    for _, _, key in sorted(remainders, reverse=True)[:remaining]:
        floors[key] += 1
    return floors


def allocate_bytes_by_overlap(
    counter: Interval,
    tx_bytes: int,
    rx_bytes: int,
    targets: Sequence[LabeledInterval],
) -> ByteAllocation:
    """Allocate integer counter bytes by temporal overlap, preserving byte totals exactly.

    Targets must be mutually exclusive. Time not covered by a target is returned as
    unallocated bytes instead of being silently discarded.
    """
    _validate_non_overlapping(targets)

    label_order: list[str] = []
    duration_by_label: dict[str, int] = {}
    covered_ns = 0
    for target in targets:
        duration = overlap_ns(counter, target)
        if duration == 0:
            continue
        if target.label not in duration_by_label:
            label_order.append(target.label)
            duration_by_label[target.label] = 0
        duration_by_label[target.label] += duration
        covered_ns += duration

    if covered_ns > counter.duration_ns:
        raise ValueError("targets must not overlap within the counter interval")
    unallocated_ns = counter.duration_ns - covered_ns
    components = [(label, duration_by_label[label]) for label in label_order]
    components.append(("__unallocated__", unallocated_ns))

    tx_alloc = _largest_remainder_allocation(tx_bytes, components, counter.duration_ns)
    rx_alloc = _largest_remainder_allocation(rx_bytes, components, counter.duration_ns)
    return ByteAllocation(
        tx_by_label={label: tx_alloc[label] for label in label_order},
        rx_by_label={label: rx_alloc[label] for label in label_order},
        unallocated_tx_bytes=tx_alloc["__unallocated__"],
        unallocated_rx_bytes=rx_alloc["__unallocated__"],
    )
