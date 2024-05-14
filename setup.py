from SPLASH._version import __version__
from setuptools import find_packages, setup

setup(
    name='astro_SPLASH',
    version=__version__,
    author='Adam Boesky',
    author_email='apboesky@gmail.com',
    description='SPLASH (Supernova classification Pipeline Leveraging Attributes of Supernova Hosts)',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=['torch',
                      'numpy',
                      'scikit-learn'],
    include_package_data=True
)

# To republish:
# 1. bump version nuber in SPLASH/_version.py
# 2. python setup.py sdist
# 3. twine upload dist/*
