import argparse
from pathlib import Path

from smbexperiments import RESULT_DIR
from smbexperiments.experimentlogger import get_logger
from smbexperiments.utils.outputs import save_plots_for_dataset_model

logger = get_logger(__name__)


def plot_results_from_jsons(results_path: Path = RESULT_DIR) -> None:
    parser = argparse.ArgumentParser(
        prog='run experiments',
        description='run at once several experiments to benchmark SMB',
        epilog='HELP')
    """ can point to  another parameter file by providing -p option"""

    parser.add_argument('-i', '--input_dir',
                        type=str,
                        required=False,
                        help="results  path where jsons are stored")

    parser.add_argument('--opts',
                        type=str,
                        nargs='+',
                        help="give a sublist of optimizers e.g. --opts SLS ADAM SMB"
                        )

    parser.add_argument('--epochs',
                        type=int,
                        required=False,
                        help="restricts plot domain to [0, epocshs]")

    args = parser.parse_args()
    if args.input_dir is not None:
        results_path = Path(args.input_dir)

    for dir_ in results_path.iterdir():
        save_plots_for_dataset_model(path_save=dir_, optimizer_list=args.opts, epochs=args.epochs)


if __name__ == "__main__":
    plot_results_from_jsons()
