from setuptools import setup


packages = ['dicodile',
            'dicodile.workers',
            'dicodile.utils',
            'dicodile.update_z',
            'dicodile.data']

setup(name='dicodile',
      version='0.1.dev0',
      packages=packages,
      install_requires=[
          'numpy',
          'numba',
          'scipy',
          'matplotlib',
          'mpi4py',
          'threadpoolctl',
          'joblib'
      ],
      )
