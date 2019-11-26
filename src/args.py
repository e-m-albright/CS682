import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "-m", "--model",
    nargs="?",
    choices=["svm", "fc"],
    default="fc",
    type=str,

)

iargs = parser.parse_args()
