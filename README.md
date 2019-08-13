# NBSVM

An sklearn-compatible classifier for benchmarking NLP classification problems.
The model used is the NBSVM described in section 2.3 of the paper
[Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf). The authors provide their own (matlab) [implementation](https://github.com/sidaw/nbsvm).

## Installation

Simply clone the repo, `cd` into the project root directory and install into a python environment with `pip install .`
For example:

```bash
python3 -m venv venv
source venv/bin/activate
git clone git@github.com:fastforwardlabs/nbsvm.git
cd nbsvm
pip install .
```

## Usage

The NBSVM classifier is intended to be used on features transformed by either `CountVectorizer` or `TfidfVectorizer`.

Example usage looks like this:

```python
from nbsvm import NBSVM

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

news = fetch_20newsgroups()

vectorizer = CountVectorizer(binary=True)

X = vectorizer.fit_transform(news.data)
y = news.target

model = NBSVM()
model.fit(X, y)
model.predict(X)
```

## Tests

There are a handful of unit tests for the public interface of the NBSVM class.
To run these locally, install the dependencies in `requirements.txt` into a clean environment and simply call `pytest` in the root directory of the project.
The first time the tests run, they will fetch a subset of the 20newsgroups dataset, which may take a few moments.
Tests should run in seconds after the initial download.
By default, the data will download to `~/scikit_learn_data` (in your home directory), which can be changed by modifying the source.