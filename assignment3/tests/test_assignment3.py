import libs.logistic_regression
import libs.svm


def test_version():
    assert libs.logistic_regression.__version__ == '0.1.0'
    assert libs.svm.__version__ == '0.1.0'
