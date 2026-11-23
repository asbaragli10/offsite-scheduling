# Offsite Scheduling

Multi-objective off-site production scheduling integrating fabrication sequencing,
daily workforce allocation and finite-trolley logistics. Parallel Pareto simulated
annealing searches for trade-offs between total tardiness, machine setup duration
and labour cost. Optional local MILP refinement improves machine sequencing and
assembly-jig staffing, with every proposal evaluated through the full simulator.

## Install

Use Python 3.11 or newer; the original requirements were tested on Python 3.12.
From this directory:

```sh
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on Linux/macOS), then install:

```sh
python -m pip install -r requirements.txt
```

For optional MILP refinement:

```sh
python -m pip install -r requirements-hybrid.txt
```

## Configure input and output paths

No input datasets, previous outputs, fitted distributions or calibration files are
included. The input CSV reader and result writers remain functional.

Edit these placeholders in `jssp_merit/paths.py`:

```python
INPUT_CSV = Path("path/to/input/products.csv")
OUTPUT_DIRECTORY = Path("path/to/output")
```

Alternatively, supply both paths on each invocation:

```sh
python main.py --data "PATH_TO_INPUT_CSV" --output-dir "PATH_TO_OUTPUT_DIRECTORY"
```

Replace the uppercase examples with actual paths. Relative paths are resolved
from the current working directory. Unedited default placeholders are rejected
before any input is read or output directory is created.

## Input format

The CSV must contain the following columns:

| Column | Meaning |
| --- | --- |
| Unique Identifier | Unique product identifier |
| Coil Size | Non-negative integer identifying the coil/setup family |
| Machine Work Time | Positive fabrication duration, in seconds by default |
| Screws | Non-negative integer assembly-work quantity |
| Studs | Non-negative integer handling-work quantity |
| Production Deadline | Product deadline; use YYYY-MM-DD |

Set `--machine-work-time-unit minutes` or `hours` when needed. Without
`--start-date YYYY-MM-DD`, the production start is inferred as seven calendar days
before the earliest deadline; `--start-offset-days` changes that offset.

## Run

Use `python main.py --help` to inspect all model and search settings.

```sh
python main.py --data "PATH_TO_INPUT_CSV" --output-dir "PATH_TO_OUTPUT_DIRECTORY" --move-config 3 --seed 42 --workers 1
```

For hybrid simulated annealing and MILP:

```sh
python main.py --data "PATH_TO_INPUT_CSV" --output-dir "PATH_TO_OUTPUT_DIRECTORY" --milp-mode both
```

MILP modes are `none`, `machine`, `jigs` and `both`. The original resource and
search defaults are retained. In particular, MILP archive selection defaults to
setup durations of 3,600-14,400 seconds and labour costs of 39,000-42,000; adjust
`--milp-setup-min-seconds`, `--milp-setup-max-seconds`, `--milp-labour-min` and
`--milp-labour-cap` for your instance, otherwise checkpoints may be skipped.

Each run creates a unique subdirectory under your configured output root with
Pareto schedules, staffing plans, KPIs, logs, calibration and a source snapshot.
Future runs also export a portable copy of the selected input for reproducibility.
Calibration is computed from your own dataset unless you explicitly provide
`--calibration-file`. Keep inputs and generated results outside version control;
common local data/output locations and tabular data files are ignored.

## Code layout

- `main.py`: optimiser entry point.
- `jssp_merit/paths.py`: input/output placeholders.
- `jssp_merit/model/`: coupled fabrication, trolley and assembly simulation.
- `jssp_merit/neighbourhoods/`: sequencing and workforce search moves.
- `jssp_merit/optimisation/`: Pareto annealing, archive and MILP integration.
- `jssp_merit/data/`: CSV loading and validation code, with no datasets.
- `jssp_merit/reporting/`: schedule exports, KPIs and provenance.
- `MILP_post_processing/{machine,jigs}/`: MILP solvers used by the optimiser.
- `tests/`: portable hybrid integration checks.

The original Python package name `jssp_merit` is retained to preserve imports.
Paper-specific campaigns, remote deployment scripts, input data and previous
results are excluded from this standalone copy.

## Checks

After installing the hybrid requirements:

```sh
python -m unittest discover -s tests -v
python -m unittest MILP_post_processing.machine.test_milp_neighborhood -v
```

These checks use small synthetic in-memory instances and require no case-study data.
