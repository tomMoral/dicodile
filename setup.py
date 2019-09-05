from setuptools import setup, find_packages


setup(name='dicodile',
      version='0.1.dev0',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'mpi4py',
          'joblib'
      ],
      )
