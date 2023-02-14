import argparse
from pathlib import Path

from smbexperiments import DEFAULT_PARAMETERS as debug_parameters_file
from smbexperiments import PAPER_BENCHMARKS as paper_benchmarks_file
from smbexperiments import RESULT_DIR as path_results
from smbexperiments.experimentlogger import get_logger
from smbexperiments.getsetupresults import get_setup_result
from smbexperiments.localtypes.trainingsettings import TrainingSettings
from smbexperiments.utils import use_GPU
from smbexperiments.utils.metrics import softmax_loss
from smbexperiments.utils.outputs import save_result, save_plots_for_dataset_model

logger = get_logger(__name__)


def train_experiments(parameters_file: Path,
                      path_save: Path
                      ) -> None:
    settings = TrainingSettings.from_yaml(yaml_file=parameters_file)
    use_gpu = use_GPU
    seed = settings.seed
    TrainingSettings.set_seeds(seed=seed)

    logger.info(f"using gpu is {use_gpu}")
    logger.info(f"seed set from {seed}")
    logger.info(f"processing a total of {len(settings.setups)} setup")

    for setup in settings.setups:
        result_dict = setup.init_result_dict()
        model = setup.model()
        configuration = setup.optimizer_configuration().copy()
        train_set, test_set, train_loader = setup.organize_training_data()

        if "SMB" in setup.optimizer_name() or "SLS" in setup.optimizer_name():
            configuration['n_batches_per_epoch'] = len(train_loader)

        if use_gpu:
            model.cuda()

        optimizer = setup.set_optimizer_with_model_parameters(params=model.parameters(), config=configuration)

        criterion = softmax_loss  # loss function

        evaluation_metrics = {"train_loss_list": [],
                              "test_acc_list": [],
                              "run_time_list": [],
                              }
        epochs = setup.epochs
        result_dict['run_configuration'] = configuration
        logger.info(f"Starting training with {result_dict['name']} for {epochs} epochs on dataset "
                    f"{setup.dataset()} with network {result_dict['model']}")
        try:
            setup_result = get_setup_result(setup=setup,
                                            model=model,
                                            epochs=epochs,
                                            train_set=train_set,
                                            test_set=test_set,
                                            train_loader=train_loader,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            evaluation_metrics=evaluation_metrics,
                                            result_dict=result_dict)
        except Exception as train_err:
            logger.error(f"cannot run setup, because {train_err}, passing")
            result_dict['status'] = "failed"
            continue

        try:
            save_result(results=setup_result, path_save=path_save)
        except Exception as save_err:
            logger.error(f"cannot save results for setup {setup.__str__()} because {save_err}, skipping")
            continue
    if not path_save.exists():
        logger.error("No result can be produced")
        raise FileNotFoundError
    else:
        logger.info("I am making plots from results""")
        for dir_ in path_save.iterdir():
            try:
                save_plots_for_dataset_model(path_save=dir_)
            except Exception as plot_err:
                logger.error(f"cannot plot results setup {setup.__str__()} because {plot_err}, skipping")


def run_experiments() -> None:
    parser = argparse.ArgumentParser(
        prog='run experiments',
        description='run at once several experiments to benchmark SMB',
        epilog='HELP')
    """ can point to  another parameter file by providing -p option"""

    parser.add_argument('-o', '--output_dir',
                        type=str,
                        required=False,
                        help="output results and plots path")

    parser.add_argument('-p', '--parameters',
                        type=str,
                        help='path to experiments parameters, optional',
                        required=False)
    args = parser.parse_args()

    if args.parameters is not None:
        path_parameters = Path(args.parameters)
    else:
        path_parameters = debug_parameters_file

    if args.output_dir is not None:
        path_save = Path(args.output_dir)
    else:
        path_save = path_results

    logger.info(f'running with settings from {path_parameters.absolute()}')
    logger.info(f'I will save results and plots to {path_save.absolute()}')
    train_experiments(parameters_file=path_parameters, path_save=path_save)


def reproduce_paper() -> None:
    logger.info(f'running with settings from {paper_benchmarks_file.absolute()}')
    logger.info(f'I will save results and plots to {path_results.absolute()}')
    train_experiments(parameters_file=paper_benchmarks_file, path_save=path_results)


def dummy_test_smb() -> None:
    logger.info(f'running with settings from {debug_parameters_file.absolute()}')
    logger.info(f'I will save results and plots to {path_results.absolute()}')
    train_experiments(parameters_file=debug_parameters_file, path_save=path_results)


if __name__ == "__main__":
    run_experiments()
