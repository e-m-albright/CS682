"""
TODO
1.
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
