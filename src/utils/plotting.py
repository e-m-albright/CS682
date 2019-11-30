import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_loss(losses: list):
    ax = sns.lineplot(range(len(losses)), losses)
    plt.show()


def plot_accuracies(accuracies: list):
    ax = sns.lineplot(range(len(accuracies)), accuracies)
    plt.show()


# from src.data.ml import Dataset
# from src.utils.plotting import show_slices
# d = Dataset()
# t,l = d.train
# s = t[0]
# show_slices(s[15, :, :], s[:, 28, :], s[:, :, 32])
#
# avg_scan = s.mean(dim=0)
# # plt.imshow(  tensor_image.permute(1, 2, 0)  )
# # plt.imshow(avg_scan.view(1, 64, 64))

"""
s.shape
from src.data.ml import Dataset
from src.utils.plotting import show_slices
d = Dataset(dimensions="2d",scale=5./6)
t,l = d.train
s = t[0]
show_slices([s[15, :, :], s[:, 20, :], s[:, :, 24]])
s.shape
from src.data.ml import Dataset
from src.utils.plotting import show_slices
d = Dataset(dimensions="2d")
t,l = d.train
s = t[0]
show_slices([s[:, 16, :], s[:, :, 16]])
s.shape
from matplotlib import pyplot as plt
plt.imshow(s.T, cmap="gray")

# SHOW THE AVERAGE
plt.imshow(s[0,:,:].T, cmap="gray")
30 * 44 * 40


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

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

    plt.suptitle("Center slices for EPI image")


def show_experimental_scans():
    from nilearn import plotting
    from nilearn import image
    from src.data import experiment

    layout = experiment.get_food_temptation_data()

    # subject data is S_ID, hi-res, exp-data, events
    subject_data = experiment.get_all_subject_data(layout)

    # take first subject, exp-data
    scans = subject_data[0][2]

    scans.orthoview()

    show_variance(scans)

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


def show_variance(scans):
    d = scans.get_data()
    # Where in the image is there actually any information for classification?

    flat_d = d.reshape(-1, d.shape[-1])
    normal_d = (d - flat_d.mean(axis=0)) / flat_d.std(axis=0)
    v = np.var(normal_d, axis=3)

    halfway = [int(i / 2) for i in v.shape]

    show_slices([v[halfway[0], :, :], v[:, halfway[1], :], v[:, :, halfway[2]]])
