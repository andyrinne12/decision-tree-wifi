import utils
import numpy as np


class DecisionTree:
    def __init__(self):
        self.depth = 0
        self.root = None

    def fit(self, x_train=None, y_train=None, training_set=None):
        if(not x_train and not y_train and not training_set):
            raise Exception("No data passed to train the model")
        if(training_set):
            self.root, self.depth = __decision_tree_learning(training_set)
        else:
            self.root, self.depth = __decision_tree_learning(
                np.hstack(x_train, y_train))

    def predict(self, x_test):
        if(not self.root):
            raise Exception("Model has to be trained first")

        predicted = []
        for entry in x_test:
            predicted.append(predict_entry(self.root, entry))

        return np.array(predicted)

# Predict the label using the given tree and data entry in a recursive manner


def predict_entry(root, data):
    if(root.left == None and root.right == None):
        return root.label

    if(data[root.attribute] <= root.value):
        return predict_entry(root.left, data)
    else:
        return predict_entry(root.right, data)


def __decision_tree_learning(training_set, depth=0):
    labels = np.unique(training_set[:, -1])

    # There is only one unique label
    if(labels.shape[0] == 1):
        return Node(0, 0, None, None, labels[0]), depth

    attr_max, val_max, left, right = find_split(training_set)

    l_branch, l_depth = __decision_tree_learning(left, depth + 1)
    r_branch, r_depth = __decision_tree_learning(right, depth + 1)

    node = Node(attr_max, val_max, l_branch, r_branch)

    return node, max(l_depth, r_depth)

# Finds and returns the split with the highest information gain


def find_split(dataset):
    attrs = dataset.shape[1] - 1
    h_max, attr_max, val_max = -1, None, None

    for i in range(attrs):
        points = np.unique(np.sort(dataset[:, i]))
        splits = np.sum(np.vstack((points[:-1], points[1:])), axis=0) / 2
        for val in splits:

            left = dataset[dataset[:, i] <= val]
            right = dataset[dataset[:, i] > val]
            left_labels = left[:, -1]
            right_labels = right[:, -1]

            h = gain(dataset[:, -1], left_labels, right_labels)

            if(h > h_max):
                h_max, attr_max, val_max = h, i, val

    left = dataset[dataset[:, attr_max] <= val_max]
    right = dataset[dataset[:, attr_max] > val_max]
    return (attr_max, val_max, left, right)

# Recursive helper to draw the tree


def __draw_tree(root, depth=1):
    if(root.left == None and root.right == None):
        return "|  " * (depth - 1) + ">> " + "class " + str(root.label)

    return ("|  ") * (depth - 1) + ("*  ") + "feature " + str(root.attribute) + " <= " + str(root.value) + '\n' + _draw_tree(root.left, depth + 1) + '\n' + ("|  ") * (depth - 1) + ("*  ") + "feature " + str(root.attribute) + " > " + str(root.value) + '\n' + _draw_tree(root.right, depth + 1)

# The text representation of the decision tree


def __str__(tree):
    return __draw_tree(tree)


# Node class used to build the decision tree
# Leaves have the left and right attributes None and have a label set

class Node:
    def __init__(self, attribute, value, left, right, label=None):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.label = label

    def is_leaf(self):
        return self.left == None and self.right == None
