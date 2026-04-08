"""Container dataclass with shipping container presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Preset definitions (all dimensions in mm)
# ---------------------------------------------------------------------------

PRESETS: dict[str, tuple[float, float, float]] = {
    "20ft Standard":  (5898.0, 2352.0, 2393.0),
    "40ft Standard":  (12032.0, 2352.0, 2393.0),
    "40ft High Cube": (12032.0, 2352.0, 2698.0),
    "Custom":         (0.0, 0.0, 0.0),
}


@dataclass
class Container:
    """A rectangular packing container.

    All dimensions are stored internally in **millimetres**.

    Attributes:
        length: Internal length along X axis (mm).
        width:  Internal width along Y axis (mm).
        height: Internal height along Z axis (mm).
        name:   Human-readable label (preset name or "Custom").
    """

    length: float
    width: float
    height: float
    name: str = "Custom"

    # -----------------------------------------------------------------------
    # Factory helpers
    # -----------------------------------------------------------------------

    @classmethod
    def from_preset(cls, preset_name: str) -> "Container":
        """Create a Container from a named preset.

        Args:
            preset_name: One of the keys in PRESETS.

        Returns:
            A Container with the preset dimensions.

        Raises:
            KeyError: If preset_name is not recognised.
        """
        if preset_name not in PRESETS:
            raise KeyError(
                f"Unknown preset '{preset_name}'. "
                f"Available: {list(PRESETS.keys())}"
            )
        l, w, h = PRESETS[preset_name]
        return cls(length=l, width=w, height=h, name=preset_name)

    @classmethod
    def from_mm(cls, length: float, width: float, height: float, name: str = "Custom") -> "Container":
        """Create a Container from millimetre dimensions."""
        return cls(length=length, width=width, height=height, name=name)

    @classmethod
    def from_inches(cls, length: float, width: float, height: float, name: str = "Custom") -> "Container":
        """Create a Container from inch dimensions (converted internally to mm)."""
        MM_PER_INCH = 25.4
        return cls(
            length=length * MM_PER_INCH,
            width=width * MM_PER_INCH,
            height=height * MM_PER_INCH,
            name=name,
        )

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------

    @property
    def volume(self) -> float:
        """Internal volume in mm³."""
        return self.length * self.width * self.height

    def dimensions_mm(self) -> tuple[float, float, float]:
        """Return (length, width, height) in millimetres."""
        return (self.length, self.width, self.height)

    def dimensions_inches(self) -> tuple[float, float, float]:
        """Return (length, width, height) in inches."""
        MM_PER_INCH = 25.4
        return (
            self.length / MM_PER_INCH,
            self.width / MM_PER_INCH,
            self.height / MM_PER_INCH,
        )

    def __repr__(self) -> str:
        return (
            f"Container({self.name!r}, "
            f"{self.length:.0f}×{self.width:.0f}×{self.height:.0f}mm)"
        )
