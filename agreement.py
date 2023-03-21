# -*- coding: utf-8 -*-
__author__ = 'Jeroen Van Der Donckt'

from typing import List, Tuple, Any, Union

import numpy as np
import scipy.special
import statsmodels.stats.inter_rater as inter_rater_stats
from functional import seq
from krippendorff import krippendorff


######## KAPPA'S

def _labels_to_binary(labels: np.ndarray, category: Any) -> np.ndarray:
    """ Helper method that converts the array of labels to an array of same size that contains binary values indicating
    where in the given array the category is present (1) and where not (0).

    :param labels: the labels of the annotators.
        The shape of this np.ndarray is (nb_samples, nb_annotators).
    :return: np.ndarray of sanme shape where labels are encoded binary; 1 if label == category, else 0
    """
    binary_labels = np.zeros(labels.shape, dtype=int)
    binary_labels[np.where(labels == category)] = 1
    return binary_labels


def annotator_accuracy(annotator_labels: np.ndarray, other_labels: np.ndarray) -> float:
    """ Computes the accuracy of the labels for the given annotator against the other labels (of the other annotators).
    This annotator accuracy is the relative observed agreement among raters.

    :param annotator_labels: the labels of the annotator for which we want to obtain its accuracy.
        The shape of this np.ndarray is (nb_samples,).
    :param other_labels: the labels of the other annotators.
        The shape of this np.ndarray is (nb_samples, nb_annotators).
    :return: Cohen's kappa score.
    """
    assert annotator_labels.size == other_labels.shape[0], 'The provided labels have the wrong shape!'
    other_labels = other_labels.T  # Transpose the array to enable numpy method broadcasting
    return np.sum(annotator_labels == other_labels) / other_labels.size


#### CHOHEN'S KAPPA

def cohens_kappa(annotator_labels: np.ndarray, other_labels: np.ndarray) -> float:
    """ Computes Cohen's kappa for the labels of the given annotator against other annotator(s) there score.
    NOTE; in case of just 2 annotators (i.e., annotator_labels and other_labels have both dimension 1) the score will
    be the same independent of which annotator is passed as either first or second argument.
    NOTE; in comparison with other implementations (e.g., sklearn, statsmodels) this implementations allows to calculate
    Cohen's kappa for the labels of one annotator (annotator_labels) against multiple other annotators (other_labels).

    :param annotator_labels: the labels of the annotator for which we want to obtain its Cohen's kappa.
        The shape of this np.ndarray is (nb_samples,).
    :param other_labels: the labels of the other annotators.
        The shape of this np.ndarray is (nb_samples, nb_annotators).
    :return: the Cohen's kappa score.
    """
    assert annotator_labels.size == other_labels.shape[0], 'The provided labels have the wrong shape!'
    # Po is the relative observed agreement among raters (identical to accuracy)
    Po = annotator_accuracy(annotator_labels, other_labels)
    # Pe is the hypothetical probability of chance agreement
    Pe = 0
    other_labels = other_labels.T  # Transpose the array to enable numpy method broadcasting
    for label in np.unique(annotator_labels):
        Plabel_annotator = np.sum(annotator_labels == label) / annotator_labels.size
        Plabel_other = np.sum(other_labels == label) / other_labels.size
        Pe += Plabel_annotator * Plabel_other
    return (Po - Pe) / (1 - Pe)


def cohens_kappa_category(annotator_labels: np.ndarray, other_labels: np.ndarray, category: Any) -> float:
    """ Computes Cohen's kappa for the given category.

    :param annotator_labels: the labels of the annotator for which we want to obtain its Cohen's kappa.
        The shape of this np.ndarray is (nb_samples,).
    :param other_labels: the labels of the other annotators.
        The shape of this np.ndarray is (nb_samples, nb_annotators).
    :param category: the category for which we want to calculate the score.
    :return: the Cohen's kappa score for the given category.
    """
    assert category in np.unique(annotator_labels), f'The given category ({category}) is not present in the labels!'
    # Work binary -> either category or NOT_category as labels
    binary_annotator_labels = _labels_to_binary(annotator_labels, category)
    binary_other_labels = _labels_to_binary(other_labels, category)
    return cohens_kappa(binary_annotator_labels, binary_other_labels)


#### FLEISS' KAPPA

def fleiss_kappa(labels: np.ndarray) -> float:
    """Computes Fleiss' kappa for group of annotators.

    :param labels: the labels of the annotators.
        The shape of this np.ndarray is (nb_samples, nb_annotators).
    :return: the Fleiss' kappa score.

    Algorithm
    ---------
    labels = np.atleast_2d(labels)
    nb_samples, nb_annotators = labels.shape[0], labels.shape[1]
    # Pbar is the (mean of the) extent to which raters agree for the samples
    Pbar = 0
    for row in labels:
        _, counts = np.unique(row, return_counts=True)
        Prow = (np.sum(counts ** 2) - nb_annotators) / (nb_annotators*(nb_annotators-1))
        Pbar += Prow
    Pbar /= nb_samples
    # PbarE is the squared sum of the proportion of all assignments for the categories
    PbarE = 0
    for label in np.unique(labels):
        plabelE = np.sum(labels == label) / labels.size
        PbarE += plabelE ** 2
    return (Pbar - PbarE) / (1 - PbarE)
    """
    assert labels.ndim == 2 and labels.shape[1] > 1, 'The provided labels have an invalid shape!'
    # Convert data labels with shape (samples, annotator) to (samples, cat_counts)
    category_table, _ = inter_rater_stats.aggregate_raters(labels)
    return inter_rater_stats.fleiss_kappa(category_table)


def fleiss_kappa_category(labels: np.ndarray, category: Any) -> float:
    """ Computes Fleiss' kappa for the given category.

    :param labels: the labels of the annotators.
        The shape of this np.ndarray is (nb_samples, nb_annotators).
    :param category: the category for which we want to calculate the score.
    :return: the Fleiss' kappa score for the given category.
    """
    assert category in np.unique(labels), f'The given category ({category}) is not present in the labels!'
    # Work binary -> either category or NOT_category as labels
    binary_labels = _labels_to_binary(labels, category)
    return fleiss_kappa(binary_labels)


#### KRIPPENDROFF'S ALPHA

def krippendorffs_alpha(labels: np.ndarray, ignore_labels: Union[None, List[Any]] = None) -> float:
    """ Computes Krippendorff's alpha for group of annotators.

    :param labels: the labels of the annotators.
        The shape of this np.ndarray is (nb_samples, nb_annotators).
    :param ignore_labels: list of labels that have to be ignored. These labels will be not considered in the calculation
    of the score.
    :return: the Krippendorff's alpha score.
    """
    if ignore_labels is None:
        ignore_labels = []
    # Reduce the ignore labels to the ones that are present in the given labels
    ignore_labels = set(seq(ignore_labels).filter(lambda el: el in labels))  # Filter the relevant labels
    # The underlying implementation of Krippendorff's alpha requires that the labels datatype is numerical!
    # => Encode the labels if they are not numerical or when some labels have to be ignored
    if not np.issubdtype(labels.dtype, np.number) or len(ignore_labels) > 0:
        unique_labels = np.unique(labels)
        encoded_labels = np.zeros(labels.shape, dtype=float)
        for code, label in enumerate(set(unique_labels).difference(ignore_labels)):
            encoded_labels[np.where(labels == label)] = code
        for label in ignore_labels:
            encoded_labels[np.where(labels == label)] = np.nan  # Is joker
        return krippendorff.alpha(encoded_labels.T, level_of_measurement='nominal')
    return krippendorff.alpha(labels.T, level_of_measurement='nominal')


def krippendorffs_alpha_category(labels: np.ndarray, category: Any) -> float:
    """ Computes Krippendorff's alpha for the given category.

    :param labels: the labels of the annotators.
        The shape of this np.ndarray is (nb_samples, nb_annotators).
    :param category: the category for which we want to calculate the score.
    :return: the Krippendorff's alpha score for the given category.
    """
    assert category in np.unique(labels), f'The given category ({category}) is not present in the labels!'
    # Work binary -> either category or NOT_category as labels
    binary_labels = _labels_to_binary(labels, category)
    return krippendorffs_alpha(binary_labels)



######## AGREEMENTS

# Inspired by: https://github.com/AlessandroChecco/percent-agreement/blob/master/percent_agreement.py
def _agreement_row(row: np.ndarray) -> float:
    """ Computes the agreement of one sample rated by different annotators.

    :param row: represents a single item rated by n different annotators (with n the length of row).
        The shape of this np.ndarray is (nb_annotators,).
    :return: the percent agreement of the row.
    """
    _, counts = np.unique(row, return_counts=True)
    # Number of agreeing combinations (i.e., combination w/ same rating) (of size 2)
    nb_agreement_combs = scipy.special.comb(counts, 2).sum()
    # Total number of possible combinations (of size 2)
    tot_nb_combs = scipy.special.comb(len(row), 2)
    return nb_agreement_combs / tot_nb_combs


def percentage_agreement(labels: np.ndarray):
    """ Computes the percent agreement of a list of samples rated by a possibly different number of raters.

    :param labels: the labels of the annotators.
        The shape of this np.ndarray is (nb_samples, nb_annotators).
    :return: the percent agreement of all the annotators.
    """
    agreements = []
    for row in labels:
        agreements.append(_agreement_row(row))
    return np.mean(agreements)


def agreement_count_categories(labels: List[List], categories: List[int], include_ties: bool = True) -> Tuple[dict, dict]:
    """ Computes the count of the annotators per stage (based on the majority vote).
    NOTE; this method supports different number of annotators for the samples (as we use lists and not np.ndarray).

    :param labels: the list of sample ratings. It is a list of lists, where the inner list contains the ratings.
        The shape of this list of lists is (nb_samples, nb_annotators), where nb_annotators can vary.
    :param categories: list of the different possible ratings (i.e., categories).
    :param include_ties: boolean indicating when ties (in majority vote) should be included or ignored.
    :return: a tuple containing two dicts; (1) a dict with as key the majority label and as value a dict containing
                                               with as keys the labels and as value the count (= nb annotators) who
                                               rated the label (key of inner dict) for this majority label (key of outer
                                               dict).
                                           (2) a dict with as key the majority label and as value the count of samples
                                               for which (i.e., the number of times) the label was the majority label.
    """
    majority_label_counts = {category: 0 for category in categories}
    label_agreements = {category: {sub_label: 0 for sub_label in categories} for category in categories}
    for row in labels:  # Iterate over the samples
        vals, counts = np.unique(row, return_counts=True)
        max_idx = counts == np.max(counts)
        row_majority_labels = vals[max_idx]
        if len(row_majority_labels) > 1 and not include_ties:
            continue  # Skip the row when there is a tie
        for majority_label in row_majority_labels:  # Iterate over the majority labels
            majority_label_counts[majority_label] += 1
            for val, count in zip(vals, counts):
                label_agreements[majority_label][val] += count
    return label_agreements, majority_label_counts
