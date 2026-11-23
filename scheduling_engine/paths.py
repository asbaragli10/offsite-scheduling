"""Replace these placeholders locally, or pass --data and --output-dir."""
from pathlib import Path


INPUT_CSV = Path("path/to/input/products.csv")
OUTPUT_DIRECTORY = Path("path/to/output")


def configure_io_paths(args) -> None:
    """Reject unconfigured paths before reading inputs or creating outputs."""
    if args.data == Path("path/to/input/products.csv"):
        raise ValueError(
            "Input path is a placeholder. Set INPUT_CSV in scheduling_engine/paths.py "
            "or pass --data with your product CSV path."
        )
    if args.output_dir == Path("path/to/output"):
        raise ValueError(
            "Output path is a placeholder. Set OUTPUT_DIRECTORY in "
            "scheduling_engine/paths.py or pass --output-dir with your results directory."
        )
    args.data = args.data.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
