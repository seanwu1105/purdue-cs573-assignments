from typing import Dict, Generic, Optional, Set, TypeVar

import pandas as pd

__version__ = '0.1.0'

T = TypeVar('T')


# pylint: disable=too-few-public-methods
class NaiveBayesClassifier(Generic[T]):
    def __init__(self,
                 df: pd.DataFrame,
                 target_name: str,
                 target_sample_space: Set[T],
                 feature_sample_spaces: Dict[str, Set]):
        self.df = df
        self.target_name = target_name
        self.target_sample_space = target_sample_space
        self.features_sample_space = feature_sample_spaces

        self.prior: Optional[pd.Series] = None
        self.likelihoods: Optional[Dict[str, pd.Series]] = None

    def predict(self, features: pd.Series) -> T:
        self._train()

        assert self.prior is not None
        assert self.likelihoods is not None

        max_target: Optional[T] = None
        max_posterior = 0
        for target in self.df[self.target_name].unique():
            if max_target is None:
                max_target = target

            prior = self.prior[target]
            likelihood = 1
            for feature_name, feature_value in features.items():
                try:
                    likelihood *= self.likelihoods[feature_name][target][feature_value]
                # If there is a feature or target value value not in the sample
                # space, set its likelihood to 0.
                except KeyError:
                    likelihood *= 0
                    break

            posterior = prior * likelihood
            if posterior > max_posterior:
                max_target = target
                max_posterior = posterior

        assert max_target is not None
        return max_target

    def _train(self):
        if self.prior is None:
            self._calculate_prior()
        if self.likelihoods is None:
            self._calculate_likelihoods()

    def _calculate_prior(self):
        grouped_count = self.df.groupby(self.target_name)[
            self.target_name].count()
        self.prior = grouped_count / grouped_count.sum()

    def _calculate_likelihoods(self):
        self.likelihoods = {}
        for feature_name in self.features_sample_space:

            # Make sure all possible values of the features are in the table.
            multi_idx = pd.MultiIndex.from_product((
                self.target_sample_space,
                self.features_sample_space[feature_name]
            ), names=[self.target_name, feature_name])

            grouped_count = (self.df
                             .groupby([self.target_name, feature_name])[feature_name]
                             .count()
                             # Fill 0 for missing values in the sample space.
                             .reindex(multi_idx, fill_value=0))

            smoothed_grouped_count = grouped_count + 1

            self.likelihoods[feature_name] = {}

            for target in smoothed_grouped_count.keys().unique(level=0):
                subgroup = smoothed_grouped_count[target]
                self.likelihoods[feature_name][target] = subgroup / \
                    subgroup.sum()
