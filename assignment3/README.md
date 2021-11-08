# CS57300 Assignment 3 - Logistic Regression and Linear SVM

Shuang Wu (wu1716@purdue.edu)

## Getting Started

We use [Poetry](https://python-poetry.org/) to manage the dependencies and
environment. Run the following command to setup developing environment.

```sh
poetry install --no-root
```

Remember to activate the virtual environment if not automatically loaded.

```sh
source ./assignment3/.venv/bin/activate
```

## Scripts

Note that the following scripts are interdependent. They need to be executed _in
order_.

### Preprocessing

To create dataset for logistic regression, linear-SVM and naive bayesian
classifier, run the following command.

```sh
python preprocess-assg3.py
```

The script should generate 4 processed data:

- `testSet_NBC.csv`
- `trainingSet_NBC.csv`
- `testSet.csv`
- `trainingSet.csv`

### Classify with Logistic Regression

Run the following script to display the training and test accuracies with
logistic regression.

```sh
python lr_svm.py trainingSet.csv testSet.csv 1
```

### Classify with linear-SVM

Run the following script to display the training and test accuracies with
linear-SVM.

```sh
python lr_svm.py trainingSet.csv testSet.csv 2
```

### Generate Performance Comparison

Run the following to display performance comparison between logistic regression,
linear-SVM and naive bayesian classifier.

```sh
python cv.py
```

## Further Details and Examples

See [`evaluation.ipynb`](./evaluation.ipynb) for simple walk through on how to
use the scripts in this project.
