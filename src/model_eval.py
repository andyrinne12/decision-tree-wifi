import numpy as np
import utils
import extra


def get_metrics_cross_validation(dataset, model, n_folds=10):
    folds = utils.train_test_k_fold(n_folds, dataset.shape[0])

    x = dataset[:, :-1]
    y = dataset[:, -1]

    metrics_list = []

    for train_indices, test_indices in folds:
        train = dataset[train_indices]
        x_test = x[test_indices, :]
        y_test = y[test_indices]

        model.fit(training_set=train)
        metrics = evaluate_metrics(x_test, y_test, model)

        metrics_list.append(metrics)

        # print(extra.print_metrics(metrics))

    metrics_list = np.array(metrics_list)

    metrics_mean = get_metrics_mean(metrics_list)

    return metrics_mean


def evaluate(test_db, trained_tree):
    x_test = test_db[:, :-1]
    y_test = test_db[:, -1]
    y_predict = trained_tree.predict(x_test)

    return np.nonzero(y_test == y_predict)[0].shape[0] / y_test.shape[0]


def evaluate_metrics(x_test, y_test, trained_tree):
    y_predict = trained_tree.predict(x_test)
    return get_metrics(y_test, y_predict)


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

    y_gold = y_gold.astype(int)
    y_prediction = y_prediction.astype(int)

    for i in range(len(y_gold)):
        confusion[y_gold[i] - 1, y_prediction[i] - 1] += 1

    return confusion


def get_metrics_mean(metrics_list):
    m1 = metrics_list[0]

    m1_acc = m1.accuracy
    m1_prec, m1_macro_prec = m1.precision
    m1_rec, m1_macro_rec = m1.recall
    m1_f1, m1_macro_f1 = m1.f1

    n = metrics_list.shape[0]

    for i in range(1, metrics_list.shape[0]):

        m = metrics_list[i]
        m_acc = m.accuracy
        m_prec, m_macro_prec = m.precision
        m_rec, m_macro_rec = m.recall
        m_f1, m_macro_f1 = m.f1

        m1_acc += m_acc
        m1_macro_prec += m_macro_prec
        m1_macro_rec += m_macro_rec
        m1_macro_f1 += m_macro_f1

        m1_prec = m1_prec + m_prec
        m1_rec = m1_rec + m_rec
        m1_f1 = m1_f1 + m_f1

    return EvalMetric(m1_acc / n, (m1_prec / n, m1_macro_prec / n), (m1_rec / n, m1_macro_rec / n), (m1_f1 / n, m1_macro_f1 / n))


class EvalMetric:
    def __init__(self, accuracy, precision, recall, f1):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
