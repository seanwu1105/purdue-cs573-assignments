# CS57300 Assignment 2 - Naive Bayes Classifier

Shuang Wu (wu1716@purdue.edu)

## Getting Started

We use [Poetry](https://python-poetry.org/) to manage the dependencies and
environment. Run the following command to setup developing environment.

```sh
poetry install --no-root
```

## Scripts

Note that the following scripts are interdependent. They need to be executed _in
order_.

### Preprocessing

To normalize the dataset, run the following command.

```sh
python preprocess.py dating-full.csv dating.csv
```

### Visualization

To show visualizations on the normalized dataset, run one of the following
commands.

To show the relation between gender and preference scores of participants:

```sh
python 2_1.py dating.csv
```

To show the relation between sucess rate and ratings of partner from
participants:

```sh
python 2_2.py
```

### Categorize Continuous Attributes

To categorize the continuous attributes in `dating.csv`:

```sh
python discretize.py dating.csv dating-binned.csv
```

Note that the categorization boundary is defined in
[`./naive_bayes_classifier/definitions.py`](./naive_bayes_classifier/definitions.py).

### Training-Test Data Splitting

To generate two mutual exclusive data set for training and test:

```sh
python split.py dating-binned.csv testSet.csv training.csv
```

### Classify with NBC

#### Default Result with Naive Bayes Classifier

Run the following script to display the training and test accuracies.

```sh
python 5_1.py
```

The model is trained with whole training set with bin size = 5.

#### Different Results with Different Bin Sizes

Run the following script to display the accuracies in different bin size. A plot
will display in the end of the execution.

```sh
python 5_2.py
```

#### Different Results with Different Fraction

Run the following script to display the accuracies in different fraction of
training set used to train the model. A plot will display in the end of the
execution.

```sh
python 5_3.py
```

## Further Details and Examples

See [`evaluation.ipynb`](./evaluation.ipynb) for simple walk through on how to
use the scripts in this project.
