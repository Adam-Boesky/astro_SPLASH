from setuptools import find_packages, setup

setup(
    name='astro_SPLASH',
    version='0.1.0',
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
# 1. bump version nuber
# 2. python setup.py sdist
# 3. twine upload dist/*
