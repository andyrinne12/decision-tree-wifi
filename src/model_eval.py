import numpy as np


def get_metrics(y_gold, y_prediction, class_labels=None):
    confusion = confusion_matrix(y_gold, y_prediction, class_labels)

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

    return EvalMetric(a, (p, p_macro), (r, r_macro), (f, f_macro))


def confusion_matrix(y_gold, y_prediction, class_labels=None):
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

    for i in range(len(y_gold)):
        confusion[y_gold[i]][y_prediction[i]] += 1

    return confusion


class EvalMetric:
    def __init__(self, accuracy, precision, recall, f1):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
