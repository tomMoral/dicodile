from setuptools import setup
import setuptools_scm  # noqa: F401
import toml  # noqa: F401

descr = """Distributed Convolutional Dictionary Learning"""

DISTNAME = 'dicodile'
DESCRIPTION = descr
MAINTAINER = 'Thomas Moreau'
MAINTAINER_EMAIL = 'thomas.moreau@inria.fr'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/tomMoral/dicodile.git'

packages = ['dicodile',
            'dicodile.workers',
            'dicodile.utils',
            'dicodile.update_d',
            'dicodile.update_z',
            'dicodile.data']

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      download_url=DOWNLOAD_URL,
      long_description=open('README.rst').read(),
      classifiers=[
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'License :: OSI Approved',
          'Programming Language :: Python',
          'Topic :: Software Development',
          'Topic :: Scientific/Engineering',
          'Operating System :: POSIX',
          'Operating System :: Unix',
      ],
      platforms='any',
      packages=packages,
      install_requires=[
          'numpy',
          'numba',
          'scipy',
          'matplotlib',
          'mpi4py',
          'threadpoolctl',
          'joblib',
          'download'
      ],
      )
