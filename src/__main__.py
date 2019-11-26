"""
TODO

Awesome!

Some TODO's to work on in the future
- More subjects (memory management)
- Cut timesteps that could either belong to one event or another (boundary dispute) as I'm not 100% confident in event assignment

- Deep learn, VGG
- Find convolution on non flattened image (3d convolution could get brain regions better?)

- Review HW for good ideas in training, data manipulation, etc

"""


def run():
    from src.args import iargs

    if iargs.model == "svm":
        from src.models import svm

        results = svm.test_ml_data()
        print(results)

    elif iargs.model == "fc":
        from src.models import fc


if __name__ == "__main__":
    run()
