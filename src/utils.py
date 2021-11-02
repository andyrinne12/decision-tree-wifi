import numpy as np
from numpy.random import default_rng
# Compute the entropy of a dataset


def entropy(dataset):
    num = dataset.shape[0]
    labels, label_counts = np.unique(dataset, return_counts=True)
    probs = label_counts / num
    entropy = -1 * np.sum(probs * np.log2(probs))
    return entropy

# Compute the weighted entropy of two datasets


def remainder(left, right):
    left_n = left.shape[0]
    right_n = right.shape[0]
    total_n = left_n + right_n
    h_l = entropy(left)
    h_r = entropy(right)
    rem = left_n / total_n * h_l + right_n / total_n * h_r

    return rem

# Compute the information gain for a dataset and a split


def gain(total, left, right):
    return entropy(total) - remainder(left, right)

# Split the dataset into train, validation and test set in k ways using k folds
# Returns indices of the entries in the initial dataset


def train_val_test_k_fold(n_folds, n_instances, random_generator=default_rng()):

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        val_k = (k + 1) % n_folds
        test_indices = split_indices[k].astype(int)
        val_indices = split_indices[val_k].astype(int)
        train_indices = np.concatenate([split_indices[i] if i != val_k and i != k else np.array(
            []) for i in range(n_folds)]).astype(int)

        folds.append([train_indices, val_indices, test_indices])

    return folds


def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    shuffled_indices = random_generator.permutation(n_instances)
    split_indices = np.array_split(shuffled_indices, n_splits)
    return split_indices
