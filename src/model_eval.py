import numpy as np

import extra
import utils


def eval_cross_validation(dataset, model, n_folds=10):
    folds = utils.train_test_k_fold(n_folds, dataset.shape[0])
    total_folds = len(folds)

    x = dataset[:, :-1]
    y = dataset[:, -1]

    class_labels = np.unique(y)
    avg_depth = 0
    confusion = np.zeros((len(class_labels), len(class_labels)))

    for train_indices, test_indices in folds:
        train = dataset[train_indices]
        x_test = x[test_indices, :]
        y_test = y[test_indices]

        model.fit(training_set=train)
        confusion += evaluate_confusion(x_test, y_test, model, class_labels)
        avg_depth += model.depth

    confusion /= total_folds
    avg_depth /= total_folds

    return metrics_from_confusion(confusion, avg_depth)


def eval_prune_nested_cross_validation(dataset, model, n_folds=10):
    folds = utils.train_val_test_k_fold(n_folds, dataset.shape[0])
    total_folds = len(folds)

    x = dataset[:, :-1]
    y = dataset[:, -1]

    class_labels = np.unique(y)
    avg_depth = 0
    confusion = np.zeros((len(class_labels), len(class_labels)))

    for train_indices, valid_indices, test_indices in folds:
        train = dataset[train_indices]
        valid = dataset[valid_indices]
        x_test = x[test_indices, :]
        y_test = y[test_indices]

        model.fit(training_set=train)
        model.prune(train, valid)
        confusion += evaluate_confusion(x_test, y_test, model, class_labels)
        avg_depth += model.depth

    confusion /= total_folds
    avg_depth /= total_folds

    return metrics_from_confusion(confusion, avg_depth)


def evaluate(test_db, trained_tree):
    x_test = test_db[:, :-1]
    y_test = test_db[:, -1]
    y_predict = trained_tree.predict(x_test)

    return np.nonzero(y_test == y_predict)[0].shape[0] / y_test.shape[0]


def evaluate_metrics(x_test, y_test, model):
    confusion = evaluate_confusion(x_test, y_test, model)
    return metrics_from_confusion(confusion, model.depth)


def evaluate_confusion(x_test, y_test, model, class_labels=None):
    y_predict = model.predict(x_test)
    return confusion_matrix(y_test, y_predict, class_labels)


def confusion_matrix(y_gold, y_prediction, class_labels=None):
    if class_labels is None:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

    y_gold = y_gold.astype(int)
    y_prediction = y_prediction.astype(int)

    for i in range(len(y_gold)):
        confusion[y_gold[i] - 1, y_prediction[i] - 1] += 1

    return confusion


def metrics_from_confusion(confusion, depth):

    # Compute the precision per class

    if np.sum(confusion) > 0:
        a = np.sum(np.diagonal(confusion)) / np.sum(confusion)
    else:
        a = 0.

    p = []
    r = []
    f = []

    n = confusion.shape[0]
    for i in range(n):
        preds_t = np.sum(confusion[:, i], axis=0)
        actual_t = np.sum(confusion[i])

        if(actual_t == 0):
            rec = 0
        else:
            rec = confusion[i][i] / actual_t

        if(preds_t == 0):
            prec = 0
        else:
            prec = confusion[i][i] / preds_t

        if(prec == 0 and rec == 0):
            f1 = 0
        else:
            f1 = 2 * prec * rec / (prec + rec)

        p.append(prec)
        r.append(rec)
        f.append(f1)

    p = np.array(p)
    r = np.array(r)
    f = np.array(f)

    p_macro = np.sum(p) / n
    r_macro = np.sum(r) / n
    f_macro = np.sum(f) / n

    return EvalMetric(confusion, a, (p, p_macro), (r, r_macro), (f, f_macro), depth)


class EvalMetric:
    def __init__(self, confusion, accuracy, precision, recall, f1, depth):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.depth = depth
        self.confusion = confusion
