import math
import itertools

import numpy as np
import matplotlib.pyplot as plt
from .train_results import FitResult
import seaborn as sns


def plot_fit(fit_res: FitResult,output_name, fig=None, log_loss=False, legend=None):
    """
    Plots a FitResult object.
    Creates four plots: train loss, test loss, train acc, test acc.
    :param fit_res: The fit result to plot.
    :param output_name: Output name for the graph file.
    :param fig: A figure previously returned from this function. If not None,
        plots will the added to this figure.
    :param log_loss: Whether to plot the losses in log scale.
    :param legend: What to call this FitResult in the legend.
    :return: The figure.
    """
    if fig is None:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10),
                                 sharex='col', sharey=False)
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()

    p = itertools.product(['train', 'test'], ['loss', 'acc'])
    for idx, (traintest, lossacc) in enumerate(p):
        ax = axes[idx]
        attr = f'{traintest}_{lossacc}'
        data = getattr(fit_res, attr)
        h = ax.plot(np.arange(1, len(data) + 1), data, label=legend)
        ax.set_title(attr)
        if lossacc == 'loss':
            ax.set_xlabel('Iteration #')
            ax.set_ylabel('Loss')
            if log_loss:
                ax.set_yscale('log')
                ax.set_ylabel('Loss (log)')
        else:
            pos_attr = f'{traintest}_pos_acc'
            neg_attr = f'{traintest}_neg_acc'
            data_pos = getattr(fit_res, pos_attr)
            data_neg = getattr(fit_res, neg_attr)

            h_pos = ax.plot(np.arange(1, len(data) + 1), data_pos, label='pos')
            h_neg = ax.plot(np.arange(1, len(data) + 1), data_neg, label='neg')
            
            ax.set_xlabel('Epoch #')
            ax.set_ylabel('Accuracy (%)')
        if legend:
            ax.legend()
        ax.grid(True)
    

    if output_name.split('_')[0] == 'Attention':
        title = 'Attention Graph'
    else:      
        title = 'Baseline Graph'
    fig.suptitle(title, fontsize=22)
    plt.savefig(output_name)

    return fig, axes


def plot_attention_map(heatmap_res, dataset):
    """
    Generates Attention Maps graph
    :param heatmap_res: attention weights data
    :param dataset: dataset matching the heat_map param
    :return: 
    """
    idx_range = 50

    for j, start in enumerate(range(0,900,30)):
        fig, ax =plt.subplots(1,2)
        indices = heatmap_res[2][start:start+idx_range]
        y_vals = [dataset[i] for i in indices]
        x_vals = [list(heatmap_res[1][i]) for i in range(start,start+idx_range)]
        
        sns.heatmap(x_vals, ax=ax[0])
        sns.heatmap(y_vals, ax=ax[1])

        plt.savefig('heatmap\\heatmap_{}'.format(j))


def plot_both_models(fit_attention, fit_base, output_name):
    """
    Plot a graph with comparison between the 2 models
    :param fit_attention: The fit result to plot for the Attention model.
    :param fit_base: The fit result to plot for the Baseline model.
    :param output_name: Output name for the graph file.
    :return: The figure
    """
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10),
                             sharex='col', sharey=False)
    axes = axes.reshape(-1)

    p = itertools.product(['train', 'test'], ['loss', 'acc'])
    for idx, (traintest, lossacc) in enumerate(p):
        ax = axes[idx]
        attr = f'{traintest}_{lossacc}'
        data_atten = getattr(fit_attention, attr)
        data_base = getattr(fit_base, attr)
        h_atten = ax.plot(np.arange(1, len(data_atten) + 1), data_atten, label='attention')
        h_base = ax.plot(np.arange(1, len(data_base) + 1), data_base, label='baseline')

        ax.set_title(attr)
        if lossacc == 'loss':
            ax.set_xlabel('Iteration #')
            ax.set_ylabel('Loss')

        else:
            ax.set_xlabel('Epoch #')
            ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True)
    title = 'Models Comparison'
    fig.suptitle(title, fontsize=22)

    plt.savefig(output_name)

    return fig, axes
