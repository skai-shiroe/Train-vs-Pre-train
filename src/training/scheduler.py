"""Learning rate schedules.

Every schedule starts with a linear warmup. A Transformer trained from scratch
is fragile during its first steps: the attention logits are near uniform and
the gradients are large, so stepping at the peak rate straight away routinely
diverges. The warmup ramps the rate from zero, and the decay that follows lets
the run settle instead of oscillating around the minimum.

All schedules are expressed as a multiplier applied to the peak rate, so the
same code drives every shape:

```text
linear        warmup, then a straight line down to zero
cosine        warmup, then a half cosine down to zero
inverse_sqrt  warmup, then sqrt(warmup / step), the original paper schedule
constant      warmup, then the peak rate held
```
"""

from __future__ import annotations

import math
from collections.abc import Callable

from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from src.training.config import SchedulerName

SCHEDULER_NAMES: tuple[str, ...] = ("linear", "cosine", "inverse_sqrt", "constant")


def _warmup_factor(step: int, warmup_steps: int) -> float:
    """Return the multiplier during the warmup.

    Args:
        step: Zero based step index.
        warmup_steps: Length of the warmup.

    Returns:
        A multiplier ramping from ``1 / warmup_steps`` up to one. It never
        returns zero: a first step at a rate of exactly zero is a wasted step.
    """
    return float(step + 1) / float(max(1, warmup_steps))


def _decay_progress(step: int, warmup_steps: int, total_steps: int) -> float:
    """Return how far the decay has progressed, in ``[0, 1]``.

    Args:
        step: Zero based step index.
        warmup_steps: Length of the warmup.
        total_steps: Total number of steps planned.

    Returns:
        Zero at the end of the warmup, one at the end of the run.
    """
    remaining = max(1, total_steps - warmup_steps)
    return min(1.0, float(step - warmup_steps) / float(remaining))


def build_lambda(
    name: SchedulerName, warmup_steps: int, total_steps: int
) -> Callable[[int], float]:
    """Return the multiplier function of a schedule.

    Exposed separately from :func:`build_scheduler` so that the shape of a
    schedule can be asserted without building an optimiser.

    Args:
        name: Schedule identifier.
        warmup_steps: Length of the warmup, in optimisation steps.
        total_steps: Total number of optimisation steps planned.

    Returns:
        A callable mapping a step index to a multiplier of the peak rate.

    Raises:
        ValueError: If the schedule is unknown, if ``total_steps`` is not
            strictly positive, or if the warmup is longer than the run.
    """
    if name not in SCHEDULER_NAMES:
        raise ValueError(f"Scheduler must be one of {SCHEDULER_NAMES}, got {name!r}.")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be strictly positive, got {total_steps}.")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non negative, got {warmup_steps}.")
    if warmup_steps > total_steps:
        raise ValueError(
            f"The warmup ({warmup_steps} steps) cannot be longer than the run "
            f"({total_steps} steps)."
        )

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return _warmup_factor(step, warmup_steps)

        if name == "constant":
            return 1.0

        if name == "inverse_sqrt":
            # sqrt(warmup / step). Without a warmup the schedule would divide
            # by zero on the first step, so it falls back to a flat rate.
            if warmup_steps == 0:
                return 1.0
            return math.sqrt(warmup_steps / float(step + 1))

        progress = _decay_progress(step, warmup_steps, total_steps)
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.0, 1.0 - progress)

    return schedule


def build_scheduler(
    optimizer: optim.Optimizer,
    name: SchedulerName,
    *,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Build the learning rate scheduler of a run.

    The returned scheduler steps once per optimisation step, not once per
    epoch. Stepping per epoch would make the schedule depend on the corpus
    size, which the ablation study varies on purpose.

    Args:
        optimizer: The optimiser whose rate is scheduled.
        name: Schedule identifier.
        warmup_steps: Length of the warmup, in optimisation steps.
        total_steps: Total number of optimisation steps planned.

    Returns:
        The configured scheduler.
    """
    return LambdaLR(optimizer, lr_lambda=build_lambda(name, warmup_steps, total_steps))
