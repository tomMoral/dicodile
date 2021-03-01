|Build Status| |codecov|

This package is still under development. If you have any trouble running this code,
please `open an issue on GitHub <https://github.com/tomMoral/dicodile/issues>`_.

DiCoDiLe
--------

Package to run the experiments for the preprint paper `Distributed Convolutional Dictionary Learning (DiCoDiLe): Pattern Discovery in Large Images and Signals <https://arxiv.org/abs/1901.09235>`__.

Installation
^^^^^^^^^^^^

All the tests should work with python >=3.6. This package depends on the python
library ``numpy``, ``matplotlib``, ``scipy``, ``mpi4py``, ``joblib``. The
package can be installed with the following command run from the root of the
package.

.. code:: bash

    pip install  -e .

Or using the conda environment:

.. code:: bash

    conda env create -f dicodile_env.yml

To build the doc use:

.. code:: bash

    pip install  -e .[doc]
    cd docs
    make html

To run the tests:

.. code:: bash

    pip install  -e .[test]
    pytest .

Usage
^^^^^

All experiments are with ``mpi4py`` and will try to spawned workers depending on the parameters set in the experiments. If you need to use an ``hostfile`` to configure indicate to MPI where to spawn the new workers, you can set the environment variable ``MPI_HOSTFILE=/path/to/the/hostfile`` and it will be automatically detected in all the experiments. Note that for each experiments you should provide enough workers to allow the script to run.

All figures can be generated using scripts in ``benchmarks``. Each script will generate and save the data to reproduce the figure. The figure can then be plotted by re-running the same script with the argument ``--plot``. The figures are saved in pdf in the ``benchmarks_results`` folder. The computation are cached with ``joblib`` to be robust to failures.

.. |Build Status| image:: https://github.com/tomMoral/dicodile/workflows/unittests/badge.svg
.. |codecov| image:: https://codecov.io/gh/tomMoral/dicodile/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/tomMoral/dicodile
