import pytest
import scipy

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nbsvm import NBSVM


def newsgroups(vectorizer):
	train = fetch_20newsgroups(
		subset="train",
		categories=["alt.atheism", "sci.space", "talk.politics.misc"]
	)
	test = fetch_20newsgroups(
		subset="test",
		categories=["alt.atheism", "sci.space", "talk.politics.misc"]
	)
	train_X = vectorizer.fit_transform(train.data)
	train_y = train.target
	test_X = vectorizer.transform(test.data)
	test_y = test.target
	return train_X, train_y, test_X, test_y


@pytest.fixture
def count_newsgroups():
	return newsgroups(CountVectorizer(binary=True))


@pytest.fixture
def tfidf_newsgroups():
	return newsgroups(TfidfVectorizer())


def test_NBSVM_initializes_with_params():
	clf = NBSVM(alpha=0.1, beta=0.2, C=0.3)
	assert clf.alpha == 0.1
	assert clf.beta == 0.2
	assert clf.C == 0.3


def test_NBSVM_raises_on_dense_negative_X():
	X = [[1., -1.]]
	y = [1]
	clf = NBSVM()
	with pytest.raises(ValueError):
		clf.fit(X,y)


def test_NBSVM_raises_on_sparse_negative():
	X = scipy.sparse.csr_matrix([[1., -1.]])
	y = scipy.sparse.csr_matrix([1])
	clf = NBSVM()
	with pytest.raises(ValueError):
		clf.fit(X,y)


def test_NBSVM_extracts_classes(count_newsgroups):
	X, y, _, _ = count_newsgroups
	clf = NBSVM()
	clf.fit(X, y)
	assert hasattr(clf, 'classes_')
	assert len(clf.classes_) == 3


def test_NBSVM_scores_well_on_test(count_newsgroups):
	train_X, train_y, test_X, test_y = count_newsgroups
	clf = NBSVM()
	clf.fit(train_X, train_y)
	p = clf.predict(test_X)
	assert accuracy_score(p, test_y) > 0.9


def test_NBSVM_scores_well_on_test(tfidf_newsgroups):
	train_X, train_y, test_X, test_y = tfidf_newsgroups
	clf = NBSVM()
	clf.fit(train_X, train_y)
	p = clf.predict(test_X)
	assert accuracy_score(p, test_y) > 0.9