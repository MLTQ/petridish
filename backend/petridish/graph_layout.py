"""Reproducible semantic-to-physical port layouts for graph experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GraphLayout:
    """Fixed task metadata and semantic port assignments."""

    key: str
    title: str
    description: str
    input_count: int
    output_count: int
    input_side: str
    output_side: str
    flow_direction: int
    input_position_order: tuple[int, ...]
    output_position_order: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.flow_direction not in {-1, 1}:
            raise ValueError("flow_direction must be -1 or 1")
        if sorted(self.input_position_order) != list(range(self.input_count)):
            raise ValueError("input_position_order must permute every input port")
        if sorted(self.output_position_order) != list(range(self.output_count)):
            raise ValueError("output_position_order must permute every output port")


def _identity(size: int) -> tuple[int, ...]:
    return tuple(range(size))


def _permutation(size: int, seed: int) -> tuple[int, ...]:
    generator = torch.Generator().manual_seed(seed)
    return tuple(torch.randperm(size, generator=generator).tolist())


LAYOUTS: dict[str, GraphLayout] = {
    "mnist": GraphLayout(
        key="mnist",
        title="Spatial MNIST",
        description="Forty-nine patch sensors on the left classify into ten right-side outputs.",
        input_count=49,
        output_count=10,
        input_side="left",
        output_side="right",
        flow_direction=1,
        input_position_order=_identity(49),
        output_position_order=_identity(10),
    ),
    "associative_recall": GraphLayout(
        key="associative_recall",
        title="Associative Recall",
        description=(
            "A stream of key/value pairs is followed by a query key; fixed random "
            "token and answer ports test content-addressed retrieval."
        ),
        input_count=10,
        output_count=10,
        input_side="left",
        output_side="right",
        flow_direction=1,
        input_position_order=_permutation(10, 24_101),
        output_position_order=_permutation(10, 24_102),
    ),
    "tiny_language": GraphLayout(
        key="tiny_language",
        title="Tiny Language Model",
        description=(
            "Tokens arrive on the right and next-token predictions leave on the left, "
            "testing autoregressive context through a reversed directed substrate."
        ),
        input_count=10,
        output_count=10,
        input_side="right",
        output_side="left",
        flow_direction=-1,
        input_position_order=_identity(10),
        output_position_order=_permutation(10, 24_103),
    ),
}


def resolve_layout(layout: str | GraphLayout) -> GraphLayout:
    """Return one registered immutable layout or reject an unknown task key."""

    if isinstance(layout, GraphLayout):
        return layout
    try:
        return LAYOUTS[layout]
    except KeyError as error:
        raise ValueError(f"unknown graph layout: {layout}") from error


__all__ = ["GraphLayout", "LAYOUTS", "resolve_layout"]
