from setuptools import find_packages, setup

"""
Python version: Python 3.6.5
virtualenv -p /usr/local/bin/python3 env
source env/bin/activate
pip install numpy pandas scipy rpy2 sklearn statsmodels matplotlib seaborn

Open repl and run
from rpy2.robjects.packages import importr
utils = importr('utils')
utils.install_packages('QUIC', repos='https://cloud.r-project.org')
utils.install_packages('stringr', repos='https://cloud.r-project.org')
utils.install_packages('glasso', repos='https://cloud.r-project.org')
utils.install_packages('gdata', repos='https://cloud.r-project.org')
utils.install_packages('psych', repos='https://cloud.r-project.org')
utils.install_packages('MGL', repos='https://cloud.r-project.org')
"""

setup(
    name='ccgowl',
    packages=find_packages(),
    version='0.1.0',
    description='ccgowl is a free software package for structure based on the Python programming language. '
                'It can be used with the interactive Python interpreter or executing Python scripts. Its main '
                'purpose is to allow for experimenting with state-of-the-art methods for learning structure in '
                'precision matrix estimation. This package includes implementations of GRAB, GOWL, and ccGOWL models.',
    author='Cody Mazza-Anthony',
    license='MIT',
)
