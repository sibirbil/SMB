from pathlib import Path

here = Path(__file__).parent.absolute()
RESULT_DIR = Path.home() / "smb_experiments_results"
DEFAULT_PARAMETERS = here / "experimentconfigurations" / "debug_experiment_parameters.yaml"
PAPER_BENCHMARKS = here / "experimentconfigurations" / "paper_benchmarks.yaml"
