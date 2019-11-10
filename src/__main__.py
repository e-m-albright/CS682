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

import os
import sys


if __name__ == "__main__":
    """
    TODO I don't love the usage
    python -m src
    kinda weird
    
    python -m <NAME>? can I keep the dir as src and use a diff name?
    """
    package_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.join('..', package_dir)
    sys.path.insert(0, parent_dir)

    from src.models import sanity_check

    results = sanity_check.test_ml_data()
    print(results)
