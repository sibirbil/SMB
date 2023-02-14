import datetime
import json
from pathlib import Path
from typing import Dict, List

from matplotlib import pyplot as plt

from smbexperiments.experimentlogger import get_logger
from . import PLOT_COLOR_MAP as colors

logger = get_logger(__name__)


def load_json(file_name: Path) -> Dict:
    with open(file_name, "r") as fp:
        return json.load(fp)


def get_result_list(results_path: Path, optimizer_list) -> List:
    all_results_files = results_path.glob("*.json")
    all_results = [load_json(_file) for _file in all_results_files]

    if optimizer_list is not None:
        results = []
        for result in all_results:
            if result['name'] in optimizer_list:
                results.append(result)
        return results
    else:
        return all_results


def get_plot_range(sample_list: List, p_range: int):
    if p_range is not None:
        return p_range
    else:
        return len(sample_list)


def show_loss_acc_graph(opt_out_list, graph_title, save_path, epochs: int) -> None:
    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(18, 7))
    fig.suptitle(graph_title, fontsize=25)

    legend_list = []
    plot_range = get_plot_range(sample_list=opt_out_list[0]["train_loss"], p_range=epochs)

    for opt_out in opt_out_list:
        legend_list.append('{}'.format(opt_out['name']))

    for opt_out in opt_out_list:
        if opt_out['name'] == "SMB":
            linewidth = 2
        else:
            linewidth = 1
        for idx, ax in enumerate(axs.ravel()):
            ax.grid(True)
            color = colors.get(opt_out['name'])

            if idx == 0:
                ax.semilogy(opt_out['train_loss'][:plot_range], color=color,
                            linewidth=linewidth)
                ax.set_ylabel("Training - Softmax Loss (log)", fontsize=24)
                ax.set_xlabel("Epochs", fontsize=24)
                # ax.legend(legend_list, loc="upper right", fontsize=15)
            if idx == 1:
                ax.plot(opt_out['test_acc'][:plot_range], linewidth=linewidth,
                        color=color)  # , color='black', marker='s', markevery=5, markersize=5)
                ax.set_ylabel('Test - Accuracy', fontsize=24)
                ax.set_xlabel("Epochs", fontsize=20)
                # ax.legend(legend_list, loc="lower right",fontsize=15)
    fig.legend(legend_list, loc='lower right', fontsize=18)

    fig.savefig(save_path)
    plt.close(fig)


def get_run_time(opt_out: list):
    opt_time = []
    cumulative = 0

    for i in range(len(opt_out['run_time'])):
        cumulative += opt_out['run_time'][i]
        opt_time.append(cumulative)

    return opt_time


def show_time_graph(opt_out_list, graph_title, save_path, epochs: int):
    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(graph_title, fontsize=25)

    legend_list = []
    plot_range = get_plot_range(sample_list=opt_out_list[0]["train_loss"], p_range=epochs)


    for opt_out in opt_out_list:
        legend_list.append('{}'.format(opt_out['name']))


    for opt_out in opt_out_list:
        opt_time = get_run_time(opt_out=opt_out)
        if opt_out['name'] == "SMB":
            linewidth = 2
        else:
            linewidth = 1
        for idx, ax in enumerate(axs.ravel()):
            ax.grid(True)
            color = colors.get(opt_out['name'])
            if idx == 0:
                ax.semilogy(opt_time[:plot_range], opt_out['train_loss'][:plot_range],
                            linewidth=linewidth, color=color)
                ax.set_ylabel("Training - Softmax Loss (log)", fontsize=24)
                ax.set_xlabel("Run Time (s)",fontsize=24)
                # ax.legend(legend_list, loc="upper right")

            if idx == 1:
                ax.plot(opt_time[:plot_range], opt_out['test_acc'][:plot_range],
                        linewidth=linewidth, color=color)
                ax.set_ylabel('Test - Accuracy',fontsize=24)
                ax.set_xlabel("Run Time (s)",fontsize=24)
                # ax.legend(legend_list, loc="lower right")
    fig.legend(legend_list, loc='lower right', fontsize=18)

    fig.savefig(save_path)
    plt.close(fig)


def save_result(results: Dict, path_save: Path) -> None:
    path_save.mkdir(exist_ok=True)
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    dataset_name = results['data']
    model_name = results['model']
    name = results['name']
    results_dir_data_model = path_save / f"{dataset_name}-{model_name}"
    results_dir_data_model.mkdir(exist_ok=True)

    results_file_name = "{}_{}_{}_{}.json".format(name,
                                                  dataset_name,
                                                  model_name,
                                                  date_time
                                                  )

    results_file = results_dir_data_model / results_file_name

    logger.info(f"saving results to {results_file.absolute()}")

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=6)


def save_plots_for_dataset_model(path_save: Path, optimizer_list=None, epochs=None):
    try:
        results = get_result_list(results_path=path_save, optimizer_list=optimizer_list)
        graph_title = f"{path_save.stem.upper()}"
        plots_path = path_save / "plots"
        plots_path.mkdir(exist_ok=True)
        save_path_acc = plots_path / Path(path_save.stem + "_accuracy.png")
        save_path_time = plots_path / Path(path_save.stem + "_run_times.png")
        show_loss_acc_graph(opt_out_list=results, graph_title=graph_title, save_path=save_path_acc, epochs=epochs)
        show_time_graph(opt_out_list=results, graph_title=graph_title, save_path=save_path_time, epochs=epochs)
    except Exception as err:
        logger.error(f"Can make plot(s) \n {err}")
