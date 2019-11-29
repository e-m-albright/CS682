import seaborn as sns
import matplotlib.pyplot as plt


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
"""

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

    plt.suptitle("Center slices for EPI image")
