import numpy as np
from numpy.random import default_rng


def entropy(dataset):
    """
    Calculates the entropy of the data
    :param dataset: The dataset
    :return: entropy of the data
    """
    num = dataset.shape[0]
    labels, label_counts = np.unique(dataset, return_counts=True)
    probs = label_counts / num
    entropy = -1 * np.sum(probs * np.log2(probs))
    return entropy



def remainder(left, right):
    """
    Calculates the remainder of the data
    :param left: left dataset
    :param right: right dataset
    :return: remainder of left and right datasets
    """
    left_n = left.shape[0]
    right_n = right.shape[0]
    total_n = left_n + right_n
    h_l = entropy(left)
    h_r = entropy(right)
    rem = left_n / total_n * h_l + right_n / total_n * h_r

    return rem


def gain(total, left, right):
    """
    Calculates the information gain from splitting total into left and right
    :param total: The data pre spilt
    :param left: All data smaller than the split
    :param right: All data larger than the split
    :return: information gain from splitting on the the split point
    """
    return entropy(total) - remainder(left, right)


def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """
    Splits the dataset into k splits
    :param n_folds: Number of folds to use
    :param n_instances: Number of rows of the set
    :param randon_generator: A random generator
    :return: information gain from splitting on the the split point
    """
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        test_indices = split_indices[k]
        train_indices = np.concatenate([split_indices[i] for i in range(
            n_folds) if i != k]).astype(int)

        folds.append([train_indices, test_indices])

    return folds



def train_val_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """
    Split the dataset into train, validation and test set in k ways using k folds
    :param n_folds: Number of folds to use
    :param n_instances: Number of rows of the set
    :param randon_generator: A random generator
    :return: indices of the entries in the initial dataset
    """    
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        for h in range(n_folds):
            if(k == h):
                continue

            test_indices = split_indices[k].astype(int)
            val_indices = split_indices[h].astype(int)
            train_indices = np.concatenate([split_indices[i] for i in range(
                n_folds) if i != h and i != k]).astype(int)

            folds.append([train_indices, val_indices, test_indices])

    return folds


def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    """
    Splits the dataset in k parts
    param n_folds: Number of folds to use
    :param n_instances: Number of rows of the set
    :param randon_generator: A random generator
    :return split_indices: Set of split indices
    """
    shuffled_indices = random_generator.permutation(n_instances)
    split_indices = np.array_split(shuffled_indices, n_splits)
    return split_indices
