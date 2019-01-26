
if [[ "$DISTRIB" == "CONDA" ]]; then
    sudo apt-get install -y libfftw3-dev
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda config --set always_yes yes --set changeps1 no
    conda config --add channels conda-forge
    conda update -q conda
    # Useful for debugging any issues with conda
    conda info -a

    conda create -q -n ompi python=3.6 openmpi pytest joblib mpi4py matplotlib numpy scipy pytest joblib
    source activate ompi
else
    python -m pip install pytest joblib
fi
