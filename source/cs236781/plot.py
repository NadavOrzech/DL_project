import math
import itertools

import numpy as np
import matplotlib.pyplot as plt
from .train_results import FitResult
import seaborn as sns


# def tensors_as_images(tensors, nrows=1, figsize=(8, 8), titles=[],
#                       wspace=0.1, hspace=0.2, cmap=None):
#     """
#     Plots a sequence of pytorch tensors as images.
#
#     :param tensors: A sequence of pytorch tensors, should have shape CxWxH
#     """
#     assert nrows > 0
#
#     num_tensors = len(tensors)
#
#     ncols = math.ceil(num_tensors / nrows)
#
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
#                              gridspec_kw=dict(wspace=wspace, hspace=hspace),
#                              subplot_kw=dict(yticks=[], xticks=[]))
#     axes_flat = axes.reshape(-1)
#
#     # Plot each tensor
#     for i in range(num_tensors):
#         ax = axes_flat[i]
#
#         image_tensor = tensors[i]
#         assert image_tensor.dim() == 3  # Make sure shape is CxWxH
#
#         image = image_tensor.numpy()
#         image = image.transpose(1, 2, 0)
#         image = image.squeeze()  # remove singleton dimensions if any exist
#
#         # Scale to range 0..1
#         min, max = np.min(image), np.max(image)
#         image = (image-min) / (max-min)
#
#         ax.imshow(image, cmap=cmap)
#
#         if len(titles) > i and titles[i] is not None:
#             ax.set_title(titles[i])
#
#     # If there are more axes than tensors, remove their frames
#     for j in range(num_tensors, len(axes_flat)):
#         axes_flat[j].axis('off')
#
#     return fig, axes
#
#
# def dataset_first_n(dataset, n, show_classes=False, class_labels=None,
#                     random_start=True, **kw):
#     """
#     Plots first n images of a dataset containing tensor images.
#     """
#
#     if random_start:
#         start = np.random.randint(0, len(dataset) - n)
#         stop = start + n
#     else:
#         start = 0
#         stop = n
#
#     # [(img0, cls0), ..., # (imgN, clsN)]
#     first_n = list(itertools.islice(dataset, start, stop))
#
#     # Split (image, class) tuples
#     first_n_images, first_n_classes = zip(*first_n)
#
#     if show_classes:
#         titles = first_n_classes
#         if class_labels:
#             titles = [class_labels[cls] for cls in first_n_classes]
#     else:
#         titles = []
#
#     return tensors_as_images(first_n_images, titles=titles, **kw)


def plot_fit(fit_res: FitResult,output_name, fig=None, log_loss=False, legend=None):
    """
    Plots a FitResult object.
    Creates four plots: train loss, test loss, train acc, test acc.
    :param fit_res: The fit result to plot.
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
    plt.savefig(output_name)

    return fig, axes


def plot_attention_map(heatmap_res, dataset):
    fig, ax =plt.subplots(1,2)

    idx_range = 50

    indices = heatmap_res[2][:idx_range]
    y_vals = [dataset[i] for i in indices]
    x_vals = [list(heatmap_res[1][i]) for i in range(idx_range)]
    
    sns.heatmap(x_vals, ax=ax[0])
    sns.heatmap(y_vals, ax=ax[1])

    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)
    # sns.heatmap(data=x_list, ax=ax1)

    plt.savefig('heatmap')


def plot_both_models(fit_attention, fit_base, output_name):
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
    plt.savefig(output_name)

    return fig, axes
