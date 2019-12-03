import argparse


parser = argparse.ArgumentParser()


# -----------------------
# Run mode selection
# -----------------------
parser.add_argument(
    "-m", "--model",
    nargs="?",
    choices=["svm", "mlp", "fc", "2d", "conv2d", "3d", "conv3d"],
    default="3d",
    type=str,
)
parser.add_argument(
    "-s", "--save",
    action="store_true",
    default=False,
)
parser.add_argument(
    "-l", "--load",
    type=str,
)

# -----------------------
# Hyperparameters
# -----------------------
parser.add_argument(
    "-e", "--epochs",
    default=3,
    type=int,
)
parser.add_argument(
    "--learning-rate",
    default=None,
    type=float,
)

# -----------------------
# Data selection
# -----------------------
parser.add_argument(
    "--subjects",
    default=6,
    type=int,
)

# -----------------------
# Operation selection
# -----------------------
parser.add_argument(
    "-p", "--plot",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--print-freq",
    default=2,
    type=int,
)
parser.add_argument(
    "--use-cpu",
    action="store_true",
    default=False,
)


iargs, _ = parser.parse_known_args()
