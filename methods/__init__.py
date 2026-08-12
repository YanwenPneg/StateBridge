# Agent definitions adapted from LatentMAS (https://github.com/Gen-Verse/LatentMAS).
# Licensed under Apache-2.0. See THIRD_PARTY_NOTICES.md.

from dataclasses import dataclass
from typing import List


@dataclass
class Agent:
    name: str
    role: str


def default_agents() -> List[Agent]:
    return [
        Agent(name="Planner", role="planner"),
        Agent(name="Critic", role="critic"),
        Agent(name="Refiner", role="refiner"),
        Agent(name="Judger", role="judger"),
    ]


__all__ = ["Agent", "default_agents"]
