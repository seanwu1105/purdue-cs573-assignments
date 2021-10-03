from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from naive_bayes_classifier import definitions


def calculate_grouped_means(filename: str) -> Dict[str, List[float]]:
    df: pd.DataFrame = pd.read_csv(filename)

    means: Dict[str, List[float]] = {'male': [], 'female': []}
    gender_encoding = {v: i for i, v in enumerate(sorted({'male', 'female'}))}

    for col in definitions.preference_scores_of_participant:
        grouped = df.groupby('gender')[col].mean()
        means['male'].append(grouped[gender_encoding['male']])
        means['female'].append(grouped[gender_encoding['female']])

    return means


def save_plot(means: Dict[str, List[float]]):
    width = 0.3
    x = np.arange(len(definitions.preference_scores_of_participant))

    _, ax = plt.subplots()

    ax.bar(x - width/2, means['male'], width, label='male')
    ax.bar(x + width/2, means['female'], width, label='female')

    ax.set_ylabel('Scores Mean')
    ax.set_xticks(x)
    ax.set_xticklabels((s.replace('_important', '').replace('_', '\n')
                        for s in definitions.preference_scores_of_participant))
    ax.legend()

    plt.savefig('2_1.png')


def main():
    means = calculate_grouped_means('dating.csv')
    save_plot(means)


if __name__ == '__main__':
    main()
