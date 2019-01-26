from setuptools import setup, find_packages


setup(name='dicodile',
      version='0.1.dev',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'mpi4py',
          'joblib'
      ],
      )
