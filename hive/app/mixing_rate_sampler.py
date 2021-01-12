"""This is a non-essential module used for convex optimization prototyping.

This functionality tests and compares the mixing rate of various
markov matrices.

You can start a test by executing the following command::

    $ python mixing_rate_sampler.py --samples=1000

You can also specify the names of the functions used to generate markov
matrices like so::

    $ python mixing_rate_sampler.py -s 10 -f afunc,anotherfunc,yetanotherfunc

Note:
    Default functions set { "new_mh_transition_matrix",
    "new_sdp_mh_transition_matrix", "new_go_transition_matrix",
    "new_mgo_transition_matrix" }

"""

from __future__ import annotations

import collections
import getopt
import importlib
import json
import os
import sys
import ast
from typing import List, Any, OrderedDict, Tuple

import numpy as np
from cvxpy.error import SolverError, DCPError

import domain.helpers.matrices as mm
from domain.helpers.matlab_utils import MatlabEngineContainer
from environment_settings import MIXING_RATE_SAMPLE_ROOT

_SizeResultsDict: OrderedDict[str, List[float]]
_ResultsDict: OrderedDict[str, _SizeResultsDict]


def __no_matlab__():
    sys.exit("MatlabEngineContainer not available. "
             "Do you have matlab packages installed?")


def main():
    """Compares the mixing rate of the markov matrices generated by all
    specified `functions`, `samples` times.

    The execution of the main method results in a JSON file outputed to
    :py:const:`~app.environment_settings.MIXING_RATE_SAMPLE_ROOT` folder.
    """

    matlab_engine = MatlabEngineContainer.get_instance()
    if not matlab_engine:
        __no_matlab__()

    try:
        import EngineError
    except ModuleNotFoundError:
        __no_matlab__()

    os.makedirs(MIXING_RATE_SAMPLE_ROOT, exist_ok=True)

    results: _ResultsDict = collections.OrderedDict()

    for size in network_sizes:
        print(f"\nTesting matrices of size: {size}.")
        size_results: _SizeResultsDict = collections.OrderedDict()
        for name in functions:
            size_results[name] = []
        for i in range(1, samples + 1):
            print(f"    Sample {i}.")
            m = mm.new_symmetric_connected_matrix(
                size, allow_sloops, enforce_sloops)
            v_ = np.abs(np.random.uniform(0, 100, size))
            v_ /= v_.sum()

            for name in functions:
                try:
                    _, mixing_rate = getattr(module, name)(m, v_)
                    size_results[name].append(mixing_rate)
                except (DCPError, SolverError, EngineError):
                    size_results[name].append(float('inf'))

        results[str(size)] = size_results

    json_string = json.dumps(results, indent=4)
    dir_contents = os.listdir(MIXING_RATE_SAMPLE_ROOT)
    fid = len([*filter(lambda x: "sample" in x, dir_contents)])
    file_path = f"{MIXING_RATE_SAMPLE_ROOT}/sample_{fid + 1}.json"
    with open(file_path, 'w+') as file:
        file.write(json_string)


if __name__ == "__main__":
    samples: int = 30
    network_sizes: Tuple = (8, 16)
    module: Any = "domain.helpers.matrices"
    functions: List[str] = [
        "new_mh_transition_matrix",
        "new_sdp_mh_transition_matrix",
        "new_go_transition_matrix",
        "new_mgo_transition_matrix"
    ]

    allow_sloops = 1
    enforce_sloops = 1

    try:
        short_opts = "s:n:m:f:a:e:"
        long_opts = ["samples=", "network_sizes=", "module=", "functions=",
                     "allow_self_loops=", "enforce_loops="]

        args, values = getopt.getopt(sys.argv[1:], short_opts, long_opts)
        for arg, val in args:
            if arg in ("-s", "--samples"):
                samples = int(str(val).strip()) or samples
            if arg in ("-n", "--network_sizes"):
                network_sizes = ast.literal_eval(str(val).strip())
            if arg in ("-m", "--module"):
                module = str(val).strip()
            if arg in ("-f", "--functions"):
                function_names = str(val).strip().split(',')
            if arg in ("a", "--allow_self_loops"):
                allow_sloops = int(str(val).strip()) or allow_sloops
            if arg in ("e", "--enforce_loops"):
                enforce_sloops = int(str(val).strip()) or enforce_sloops

        module = importlib.import_module(module)
        main()
    except getopt.GetoptError:
        sys.exit("Usage: python mixing_rate_sampler.py -s 1000 -f a_matrix_generator")
    except ValueError:
        sys.exit("Execution arguments should have the following data types:\n"
                 "  --samples -s (int)\n"
                 "  --network_size -n (comma seperated list of int)\n"
                 "  --module -m (str)\n"
                 "  --functions -f (comma seperated list of str)\n"
                 "  --allow_self_loops (int) in {0, 1}\n"
                 "  --enforce_loops (int) in {0, 1}\n")
    except (ModuleNotFoundError, ImportError):
        sys.exit(f"Module '{module}' does not exist or can not be imported.")
    except AttributeError:
        sys.exit(f"At least a function does not exist in module '{module}'.")
