import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "-m", "--model",
    nargs="?",
    choices=["svm", "fc"],
    default="fc",
    type=str,
)
parser.add_argument(
    "--print-every",
    default=10,
    type=int,
)

iargs = parser.parse_args()
