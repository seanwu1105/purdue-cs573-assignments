import pandas as pd
from matplotlib import pyplot as plt

from libs import Classifier
from libs.cv import CrossValidation
from libs.decision_tree import Dt
from libs.random_forest import Rf


def main():
    training_set = pd.read_csv('trainingSet.csv')
    test_set = pd.read_csv('testSet.csv')

    depths = (3, 5, 7, 9)
    stats = {
        'dt': {'test_acc': [], 'std_err': []},
        'bt': {'test_acc': [], 'std_err': []},
        'rf': {'test_acc': [], 'std_err': []}
    }

    for depth in depths:
        print('Depth:', depth)
        test_acc, std_err = cross_validate(Dt(max_depth=depth),
                                           training_set, test_set,
                                           name='Decision Tree')
        stats['dt']['test_acc'].append(test_acc)
        stats['dt']['std_err'].append(std_err)

        test_acc, std_err = cross_validate(
            Rf(max_depth=depth, attributes_downsampling=False),
            training_set, test_set, name='Bagging')
        stats['bt']['test_acc'].append(test_acc)
        stats['bt']['std_err'].append(std_err)

        test_acc, std_err = cross_validate(
            Rf(max_depth=depth, attributes_downsampling=True),
            training_set, test_set, name='Random Forest')
        stats['rf']['test_acc'].append(test_acc)
        stats['rf']['std_err'].append(std_err)

    _, ax = plt.subplots()
    ax.errorbar(depths, stats['dt']['test_acc'],
                yerr=stats['dt']['std_err'], label='Decision Tree')
    ax.errorbar(depths, stats['bt']['test_acc'],
                yerr=stats['bt']['std_err'], label='Bagging')
    ax.errorbar(depths, stats['rf']['test_acc'],
                yerr=stats['rf']['std_err'], label='Random Forest')
    ax.legend()
    plt.show()


def cross_validate(classifier: Classifier,
                   training_set: pd.DataFrame, test_set: pd.DataFrame,
                   name: str):
    validation = CrossValidation(training_set)
    best_model, avg_acc, std_err = validation.validate(classifier, t_frac=0.5)
    test_acc = best_model.test(test_set)
    print(f'[{name}] Test Accuracy:', test_acc)
    print(f'[{name}] CV Average Accuracy:', avg_acc)
    print(f'[{name}] CV Standard Error:', std_err)
    return test_acc, std_err


if __name__ == '__main__':
    main()
