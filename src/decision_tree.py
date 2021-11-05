import utils
import numpy as np
import model_eval


class DecisionTree:
    def __init__(self):
        self.depth = 0
        self.root = None

    def fit(self, x_train=None, y_train=None, training_set=None):
        if(x_train is None and y_train is None and training_set is None):
            raise Exception("No data passed to train the model")
        if(training_set is not None):
            self.root, self.depth = self.__decision_tree_learning(training_set)
        else:
            self.root, self.depth = self.__decision_tree_learning(
                np.hstack(x_train, y_train))

    def get_avg_depth(self):
        total_depths, total_branches = self.__get_avg_depth(self.root)
        return total_depths / total_branches

    def predict(self, x_test):
        if(not self.root):
            raise Exception("Model has to be trained first")

        predicted = []
        for entry in x_test:
            predicted.append(predict_entry(self.root, entry))

        return np.array(predicted)

    def prune(self, training_set, validation_set):
        self.depth = self.__prune(self.root, training_set, validation_set)

    # The text representation of the decision tree

    def __str__(self):
        return self.__draw_tree(self.root)

    def __prune(self, root, training_set, validation_set):
        if(root is None):
            return 0

        left_set = training_set[training_set[:, root.attribute] <= root.value]
        right_set = training_set[training_set[:, root.attribute] > root.value]

        l_depth = self.__prune(root.left, left_set, validation_set)
        r_depth = self.__prune(root.right, right_set, validation_set)

        left = root.left
        right = root.right

        if(left is not None and right is not None and left.label is not None and right.label is not None):
            acc0 = model_eval.evaluate(validation_set, self)

            if(left_set.shape[0] > right_set.shape[0]):
                root.label = left.label
            else:
                root.label = right.label
            root.left = None
            root.right = None

            acc1 = model_eval.evaluate(validation_set, self)

            #print(root.attribute, root.value, acc0, acc1)

            if(acc1 < acc0):
                root.left = left
                root.right = right
                root.label = None
                return 1

            return 0

        return max(l_depth, r_depth) + 1

    def __decision_tree_learning(self, training_set, depth=0):
        labels = np.unique(training_set[:, -1])

        # There is only one unique label
        if(labels.shape[0] == 1):
            return Node(0, 0, None, None, labels[0]), depth

        attr_max, val_max, left, right = find_split(training_set)

        l_branch, l_depth = self.__decision_tree_learning(left, depth + 1)
        r_branch, r_depth = self.__decision_tree_learning(right, depth + 1)

        node = Node(attr_max, val_max, l_branch, r_branch)

        return node, max(l_depth, r_depth)

    # Recursive helper to draw the tree

    def __draw_tree(self, root, depth=1):
        if(root.left == None and root.right == None):
            return "|  " * (depth - 1) + ">> " + "class " + str(root.label)

        return ("|  ") * (depth - 1) + ("*  ") + "feature " + str(root.attribute) + " <= " + str(root.value) + '\n' + self.__draw_tree(root.left, depth + 1) + '\n' + ("|  ") * (depth - 1) + ("*  ") + "feature " + str(root.attribute) + " > " + str(root.value) + '\n' + self.__draw_tree(root.right, depth + 1)

    def __get_avg_depth(self, root):
        if(root.left is None and root.right is None):
            return 1, 1

        else:
            l_nodes, l_branches = self.__get_avg_depth(root.left)
            r_nodes, r_branches = self.__get_avg_depth(root.right)

            return l_nodes + r_nodes + 1, l_branches + r_branches

# Predict the label using the given tree and data entry in a recursive manner


def predict_entry(root, data):
    if(root.left == None and root.right == None):
        return root.label

    if(data[root.attribute] <= root.value):
        return predict_entry(root.left, data)
    else:
        return predict_entry(root.right, data)

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

            h = utils.gain(dataset[:, -1], left_labels, right_labels)

            if(h > h_max):
                h_max, attr_max, val_max = h, i, val

    left = dataset[dataset[:, attr_max] <= val_max]
    right = dataset[dataset[:, attr_max] > val_max]
    return (attr_max, val_max, left, right)


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
        return self.left is None and self.right is None
