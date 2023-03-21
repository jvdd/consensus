# Consensus
Contains methods for (1) evaluating inter-rater reliability and (2) constructing a (majority) vote based on multiple annotations per sample.

## Agreement
`agreement.py` provides metrics for evaluating inter-rater reliability.  
Each method takes the multi-annotator labels as a list or np.ndarray, the shape of this list is (nb_samples, nb_annotators).

The supported metrics are;               
* `annotator_accuracy`: Computes the accuracy of the labels for the given annotator against the other labels (of the other annotators).
    This annotator accuracy is the relative observed agreement among raters.
* `cohens_kappa`: Computes Cohen's kappa for the labels of the given annotator against other annotator(s) there score.  
    *Note*; in case of just 2 annotators the score will be the same independent of which annotator is passed as either first or second argument.  
    *Note*; in comparison with other implementations (e.g., sklearn, statsmodels) this implementations allows to calculate Cohen's kappa for the labels of one annotator against multiple other annotators.
* `cohens_kappa_category`: Computes Cohen's kappa for the given category.
* `fleiss_kappa`: Computes Fleiss' kappa for group of annotators.
* `fleiss_kappa_category`: Computes Fleiss' kappa for the given category.
* `krippendorffs_alpha`: Computes Krippendorff's alpha for group of annotators.
* `krippendorffs_alpha_category`: Computes Krippendorff's alpha for the given category.
* `percentage_agreement`: Computes the percent agreement of a list of samples rated by a possibly different number of raters.
* `agreement_count_categories`: Computes the count of the annotators per stage (based on the majority vote).  
    *Note*; this method supports different number of annotators for the samples.



## Vote
`vote.py` provides methods for aggregating multi-annotator votes to a majority vote (per sample).  
Each method takes the multi-annotator labels as a list or np.ndarray, the shape of this list is (nb_samples, nb_annotators).

The supported methods are;
* `majority_vote`: Computes the (probabilistic) majority vote for all the samples in the given labels.
    This is an unweighted majority vote (i.e., each annotator is weighted equally).  
    *Note*; this method supports different number of annotators for the samples.
* `majority_vote_weighted`: Computes the (probabilistic) weighted majority vote for all the samples in the given labels.
    The majority vote is weighted by the biased Cohen's kappa of each annotator.  
    NOTE; this method requires that each sample has the same amount of annotations (i.e., columns).