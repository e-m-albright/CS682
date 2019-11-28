import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "-m", "--model",
    nargs="?",
    choices=["svm", "mlp", "fc"],
    default="fc",
    type=str,
)
parser.add_argument(
    "-e", "--epochs",
    default=3,
    type=int,
)
parser.add_argument(
    "--use-cpu",
    action="store_true",
    default=False,
)

iargs, _ = parser.parse_known_args()
