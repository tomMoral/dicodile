This package is still under development. If you have any trouble running this code, please contact <thomas.moreau.2010@gmail.com>


## DiCoDiLe

Package to run the experiments for the ICML submission [Distributed Convolutional Dictionary Learning (DiCoDiLe): Pattern Discovery in Large Images and Signals](#).

#### Requirements

All the tests were done with python3.6.
This package depends on the python library `numpy`, `matplotlib`, `scipy`, `mpi4py`, `joblib`.
The package can be installed with the following command run from the root of the package.

```bash
pip install  -e .
```

#### Usage

All experiments are with `mpi4py` and will try to spawned workers depending on the parameters set in the experiments. If you need to use an `hostfile` to configure indicate to MPI where to spawn the new workers, you can set the environment variable `MPI_HOSTFILE=/path/to/the/hostfile` and it will be automatically detected in all the experiments. Note that for each experiments you should provide enough workers to allow the script to run.

All figures can be generated using scripts in `benchmarks`. Each script will generate and save the data to reproduce the figure. The figure can then be plotted by re-running the same script with the argument `--plot`. The figures are saved in pdf in the `benchmarks_results` folder. The computation are cached with `joblib` to be robust to failures.
