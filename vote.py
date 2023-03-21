
# import pandas as pd
from typing import List, Union, Tuple, Any

import numpy as np

from .agreement import cohens_kappa


### UNWEIGHTED VOTE

def _majority_vote_row(row: List, prob: bool) -> Union[str, List[Tuple[str, float]]]:
    """ Calculates the majority vote for one sample rated by different annotators.

    :param row: represents a single item rated by n different annotators (with n the length of row).
        The shape of this list is (nb_annotators,).
    :param prob: boolean indicating when probabilities should be returned instead of a hard vote.
    :return: the (probabilistic) majority vote for the row.
        If prob = False, the hard vote is returned as a string. When there is a tie, the tied labels are concatenated
        and separated by a '-'.
        If prob = True, the probabilistic vote is returned as a list of tuples (label, proba). The shape of this list
        is (nb_unique_row_labels,), where nb_unique_row_labels is the number of unique labels in the given row.
    """
    # https://stackoverflow.com/questions/11620914/removing-nan-values-from-an-array
    votes = list(filter(lambda v: v == v, np.array(row).ravel().tolist()))
    label_counts = [(label, votes.count(label)) for label in set(votes)]
    if prob:  # Return list of tuples (label, probability)
        return [(label, count / len(votes)) for label, count in label_counts]
    # Create the hard vote if not probabilistic
    max_count = -1
    max_label = []
    for label, count in label_counts:
        if count > max_count:
            max_label = [label]
            max_count = count
        elif count == max_count:
            max_label.append(label)
    max_label = sorted(max_label)
    return '-'.join(max_label)


def majority_vote(labels: List[List], prob: bool = False) -> List:
    """ Computes the (probabilistic) majority vote for all the samples in the given labels.
    This is an unweighted majority vote (i.e., each annotator is weighted equally).
    NOTE; this method supports different number of annotators for the samples (as we use lists and not np.ndarray).


    :param labels: the list of sample ratings. It is a list of lists, where the inner list contains the ratings.
        The shape of this list of lists is (nb_samples, nb_annotators), where nb_annotators can vary.
    :param prob: boolean indicating when probabilities should be returned instead of hard votes.
    :return: list containing the majority votes (if prob = False), otherwise the list contains tuples (label, proba).
        The shape of this list is (nb_samples,) if prob = False.
        The shape of this list is (nb_samples, nb_unique_row_labels) if prob = True, where nb_unique_row_labels varies.
    """
    return [_majority_vote_row(row, prob) for row in labels]


# def majority_vote_df(df : pd.DataFrame, label_cols: Union[None, List[str]] = None, prob: bool = False) -> List:
#     if label_cols is None:
#         label_cols = df.columns
#     return list(df[label_cols].apply(lambda row: _majority_vote_row(row, prob), axis=1))



### WEIGHTED VOTE

def _get_annotator_weights(labels: np.ndarray, biased: bool = True) -> List[float]:
    """ Calculates the weights for each annotator, the weight of an annotator is expressed as the Cohen's kappa score.

    :param labels: the labels of the annotators.
        The shape of this np.ndarray is (nb_samples, nb_annotators).
    :param biased: boolean indicating whether the biased kappa score should be used as weight.
    :return: the weights of the annotators.
    """
    assert np.array(labels).ndim == 2, 'The labels must have consistent number of annotators!'
    weights = []
    for idx in range(np.array(labels).shape[1]):
        annotator_labels = labels[:, idx]
        other_labels = np.array(labels)
        if not biased:  # If unbiased => remove annotator column from other_labels
            other_labels = np.hstack([other_labels[:, :idx], other_labels[:, idx+1:]])
        weights.append(cohens_kappa(annotator_labels, other_labels))
    return weights


def _majority_vote_weighted_row(row: np.ndarray, weights: List[float], prob: bool) -> Union[Any,Tuple[Any,float]]:
    """ Calculates the (probabilistic) weighted majority vote for one sample rated by different annotators.

    :param row: represents a single item rated by n different annotators (with n the length of row).
        The shape of this np.ndarray is (nb_annotators,).
    :param weights: a list of weights, each weight indicates the 'weight' of the corresponding annotator's vote.
    :param prob: boolean indicating when probabilities should be returned instead of a hard vote.
    :return: the (probabilistic) weighted majority vote for the row.
        If prob = False, the hard vote is returned.
        If prob = True, the probabilistic vote is returned as a list of tuples (label, proba). The shape of this list
        is (nb_unique_row_labels,), where nb_unique_row_labels is the number of unique labels in the given row.
    """
    assert len(row) == len(weights), 'The weights must have the same length as the number of annotators!'
    label_votes = {label: 0 for label in np.unique(row)}
    for idx, label in enumerate(row):  # Perform the weighted voting
        label_votes[label] += weights[idx]
    if prob:  # Return list of tuples (label, probability)
        return [(label, count / sum(weights)) for label, count in label_votes.items()]
    return max(label_votes, key=label_votes.get)  # Return label with largest vote


def majority_vote_weighted(labels: np.ndarray, prob: bool = False) -> List:
    """ Computes the (probabilistic) weighted majority vote for all the samples in the given labels.
    The majority vote is weighted by the biased Cohen's kappa of each annotator.
    NOTE; As the type of labels is np.ndarray, each sample (i.e., row) needs to have the same amount of annotations
    (i.e., columns).

    :param labels: the labels of the annotators.
        The shape of this np.ndarray is (nb_samples, nb_annotators).
    :param prob: boolean indicating when probabilities should be returned instead of hard votes.
    :return: list containing the weighted majority votes (where each annotators vote is weighted by its Cohen's kappa)
    if prob = True, otherwise the list contains tuples (label, proba).
        The shape of this list is (nb_samples,) if prob = False.
        The shape of this list is (nb_samples, nb_unique_row_labels) if prob = True, where nb_unique_row_labels varies.
    """
    weights = _get_annotator_weights(labels)
    return [_majority_vote_weighted_row(row, weights, prob) for row in labels]


# def majority_vote_weighted_df(df: pd.DataFrame, label_cols: Union[None, List[str]] = None ) -> List:
#     if label_cols is None:
#         label_cols = df.columns
#     weights = _get_annotator_weights(df[label_cols].values)
#     return list(df[label_cols].apply(lambda row: _majority_vote_weighted_row(row, weights), axis=1))
