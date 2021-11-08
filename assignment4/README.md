# CS 57300 Data Mining Assignment 4

Shuang Wu (wu1716@purdue.edu)

## Getting Started

We use [Poetry](https://python-poetry.org/) to manage the dependencies and
environment. Run the following command to setup developing environment.

```sh
poetry install --no-root
```

Remember to activate the virtual environment if not automatically loaded.

```sh
source ./assignment4/.venv/bin/activate
```

## Scripts

Note that the following scripts are interdependent. They need to be executed _in
order_.

### Preprocessing

To create dataset for decision tree, bagging and random forest, run the
following command.

```sh
python preprocess-assg4.py
```

The script should generate 2 processed data:

- `testSet.csv`
- `trainingSet.csv`

### Classify with Decision Tree

Run the following script to display the training and test accuracies with the
decision tree classifier.

```sh
python trees.py trainingSet.csv testSet.csv 1
```

### Classify with Bagging

Run the following script to display the training and test accuracies with
bagging classifier.

```sh
python trees.py trainingSet.csv testSet.csv 2
```

### Generate Performance Comparison

Run the following to display performance comparison between different
hyperparamters.

```sh
python cv_depth.py
python cv_frac.py
python cv_numtrees.py
```

### Classify with Multi-Layer Perceptrons (Neural Network)

Run the following script to display the training and test accuracies with
multi-layer perceptrons classifier.

```sh
python bonus.py trainingSet.csv testSet.csv
```

## Further Details and Examples

See [`evaluation.ipynb`](./evaluation.ipynb) for simple walk through on how to
use the scripts in this project.
