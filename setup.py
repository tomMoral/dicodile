from pkg_resources import packaging
from setuptools import setup
import setuptools

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

min_setuptools_ver = "46.4.0"
if packaging.version.parse(setuptools.__version__) < \
        packaging.version.parse(min_setuptools_ver):
    raise ValueError(f"""Expected setuptools >= {min_setuptools_ver},
                      found {setuptools.__version__} instead.
                      Please update setuptools.""")

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
