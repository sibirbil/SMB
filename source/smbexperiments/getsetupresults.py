from typing import Dict
from smbexperiments.localtypes.trainingsetups import TrainingSetup
from smbexperiments.experimentlogger import get_logger

logger = get_logger(__name__)


def get_setup_result(
        setup: TrainingSetup,
        model,
        epochs,
        train_set,
        train_loader,
        test_set,
        criterion,
        optimizer,
        evaluation_metrics,
        result_dict
) -> Dict:
    def _log_accuracy(epoch: int):
        logger.info(
            f'Epoch: {epoch}   -   Training Loss: {round(evaluation_metrics["train_loss_list"][epoch - 1], 6)}  -  '
            f'Test Accuracy: {round(evaluation_metrics["test_acc_list"][epoch - 1], 6)}'
            f'  -  Time: {round(evaluation_metrics["run_time_list"][epoch - 1], 2)}')

    if "SMB" in setup.test_optimizer().name:
        for epoch in range(1, epochs + 1):
            setup.train_epoch_with_smb(model=model,
                                       train_set=train_set,
                                       test_set=test_set,
                                       train_loader=train_loader,
                                       criterion=criterion,
                                       metric_lists=evaluation_metrics,
                                       optimizer=optimizer)
            _log_accuracy(epoch=epoch)

    elif setup.test_optimizer().name == "SLS":
        train_iter_loss_list = []
        for epoch in range(1, epochs + 1):
            setup.train_epoch_with_sls(model=model,
                                       train_set=train_set,
                                       test_set=test_set,
                                       train_loader=train_loader,
                                       train_iter_loss_list=train_iter_loss_list,
                                       criterion=criterion,
                                       metric_lists=evaluation_metrics,
                                       optimizer=optimizer)
            _log_accuracy(epoch=epoch)

    elif setup.test_optimizer().name == "ADAM":
        for epoch in range(1, epochs + 1):
            setup.train_epoch_with_adam(model=model,
                                        train_set=train_set,
                                        test_set=test_set,
                                        train_loader=train_loader,
                                        criterion=criterion,
                                        metric_lists=evaluation_metrics,
                                        optimizer=optimizer)
            _log_accuracy(epoch=epoch)

    elif setup.test_optimizer().name == "SGD":
        for epoch in range(1, epochs + 1):
            setup.train_epoch_with_sgd(model=model,
                                       train_set=train_set,
                                       test_set=test_set,
                                       train_loader=train_loader,
                                       criterion=criterion,
                                       metric_lists=evaluation_metrics,
                                       optimizer=optimizer)
            _log_accuracy(epoch=epoch)

    else:
        logger.error("no compatible optimizer configuration has worked, stopping")
        raise ValueError

    result_dict.update({'train_loss': evaluation_metrics["train_loss_list"],
                        'test_acc': evaluation_metrics["test_acc_list"],
                        'run_time': evaluation_metrics["run_time_list"],
                        })
    return result_dict
