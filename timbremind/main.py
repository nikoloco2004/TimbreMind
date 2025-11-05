"""Command line interface for generating audio-driven flow field art."""

from __future__ import annotations

import argparse
from pathlib import Path

from .audio import extract_audio_features
from .visualize import FlowFieldConfig, FlowFieldRenderer


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser describing the available CLI options."""

    parser = argparse.ArgumentParser(
        description=(
            "Transform a .wav file into a Perlin noise flow field visualisation. "
            "The resulting image is saved as a PNG file."
        )
    )
    parser.add_argument("input", type=Path, help="Path to the input .wav file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("timbremind_output.png"),
        help="Location where the generated image will be written",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed controlling the appearance of the noise field",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1920, 1080),
        help="Image resolution in pixels (width height)",
    )
    parser.add_argument(
        "--grid",
        type=int,
        nargs=2,
        metavar=("COLUMNS", "ROWS"),
        default=(80, 45),
        help="Number of flow lines across the width and height of the image",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = FlowFieldConfig(
        seed=args.seed,
        resolution=tuple(args.resolution),
        grid_size=tuple(args.grid),
    )

    features = extract_audio_features(args.input)
    renderer = FlowFieldRenderer(config)
    renderer.render(features, args.output)

    print(f"Flow field written to {args.output.resolve()}")


if __name__ == "__main__":  # pragma: no cover - direct CLI execution entry point
    main()
