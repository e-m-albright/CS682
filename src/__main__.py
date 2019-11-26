"""
TODO

Awesome!

Some TODO's to work on in the future
- More subjects (memory management)
- Cut timesteps that could either belong to one event or another (boundary dispute) as I'm not 100% confident in event assignment

- Deep learn, VGG
- Find convolution on non flattened image (3d convolution could get brain regions better?)

- Review HW for good ideas in training, data manipulation, etc

# TODO is this shit from notebooks import errors?
    # package_dir = os.path.dirname(os.path.realpath(__file__))
    # parent_dir = os.path.join('..', package_dir)
    # sys.path.insert(0, parent_dir)

"""

from src.args import iargs


def run():

    if iargs.model == "svm":
        from src.models import svm

        results = svm.test_ml_data()
        print(results)

    elif iargs.model == "fc":
        from src.models import fc

        # TODO


if __name__ == "__main__":
    run()
