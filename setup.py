from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='ccgowl is a free software package for structure based on the Python programming language. '
                'It can be used with the interactive Python interpreter or executing Python scripts. Its main '
                'purpose is to allow for experimenting with state-of-the-art methods for learning structure in '
                'precision matrix estimation. This package includes implementations of GRAB, GOWL, and ccGOWL models.',
    author='Cody Mazza-Anthony',
    license='MIT',
)
