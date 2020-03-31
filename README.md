# ccgowl: Gaussian Graphical Model library

ccgowl is a Python library for estimating inverse covariance matrices when pre-existing structure exists between features.
This code was used in the paper available [here](https://cmazzaanthony.github.io/files/1906.02719.pdf).

## Documentation

Sphinx Documentation is available [here](https://cmazzaanthony.github.io/ccgowl/).

## Installation

```bash
git clone git@github.com:cmazzaanthony/ccgowl.git
cd ccgowl
pip install -e .
```

## Usage

```python
import numpy as np
from ccgowl.models.ccgowl import CCGOWLModel

from ccgowl.data.make_synthetic_data import generate_theta_star_gowl, standardize

p = 10
n = 100
n_blocks = 1
theta_star_eps, blocks, theta_star = generate_theta_star_gowl(p=p,
                                                              alpha=0.5,
                                                              noise=0.1,
                                                              n_blocks=n_blocks,
                                                              block_min_size=2,
                                                              block_max_size=6)

theta_star_eps = theta_star_eps[0]  # by default we generate 1 trial, but for simulations we generate many trials
sigma = np.linalg.inv(theta_star_eps)
n = 100
X = np.random.multivariate_normal(np.zeros(p), sigma, n)
X = standardize(X)  # Standardize data to have mean zero and unit variance.
S = np.cov(X.T)

lam1 = 0.05263158  # controls sparsity
lam2 = 0.05263158  # encourages equality of coefficients

model = CCGOWLModel(X, lam1, lam2)
model.fit()
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── ccgowl             <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)