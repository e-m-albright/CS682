import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from src import PLOTS_DIR


# TODO hardcode names begone
def plot_loss(losses: list, num_batches: int, name: str):
    # Use num_batches to smooth/avg?
    ax = sns.lineplot(
        range(len(losses)),
        losses,
    )
    ax.set_title('Loss')
    ax.set_ylabel('Cross Entropy Loss')
    ax.set_xlabel('Epoch/Batch Iterations')

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name + "_loss.png")
    plt.savefig(path)


# TODO other metrics, test/val
def plot_accuracies(accuracies: list, name: str):
    ax = sns.lineplot(
        range(len(accuracies)),
        accuracies,
    )
    ax.set_title('Validation Accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name + "_accuracy.png")
    plt.savefig(path)


# TODO
def load_show_slices():
    """
    def imshow(inp, title=None):
        inp = inp.numpy().transpose((1, 2, 0))
        # plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    def show_databatch(inputs, classes):
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[class_names[x] for x in classes])

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders[TRAIN]))
    show_databatch(inputs, classes)
    """

    from src.data.ml import Dataset

    d = Dataset()
    t, l = d.train
    s = t[0]
    show_slices([s[15, :, :], s[:, 28, :], s[:, :, 32]])

    # Show single
    # plt.imshow(s[0, :, :].T, cmap="gray")


def show_slices(slices):
    """Function to display row of image slices"""
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

    plt.suptitle("Slices")


def load_show_experimental_scans():
    from nilearn import plotting
    from src.data import experiment

    layout = experiment.get_food_temptation_data()

    # subject data is S_ID, hi-res, exp-data, events
    subject_data = experiment.get_all_subject_data(layout)

    # take first subject, exp-data
    scans = subject_data[0][2]

    scans.orthoview()

    _show_variance(scans)

    # pick a single scan to show
    single_scan = scans.slicer[:, :, :, 0]

    plotting.plot_stat_map(
        single_scan,
        # threshold=30,
        # display_mode="z",
        # cut_coords=1,
        # colorbar=False,
    )
    plotting.plot_img(single_scan)
    plotting.plot_epi(single_scan)
    plotting.plot_roi(single_scan)
    plotting.plot_anat(single_scan)


def _show_variance(scans):
    d = scans.get_data()
    # Where in the image is there actually any information for classification?

    flat_d = d.reshape(-1, d.shape[-1])
    normal_d = (d - flat_d.mean(axis=0)) / flat_d.std(axis=0)
    v = np.var(normal_d, axis=3)

    halfway = [int(i / 2) for i in v.shape]

    show_slices([v[halfway[0], :, :], v[:, halfway[1], :], v[:, :, halfway[2]]])
