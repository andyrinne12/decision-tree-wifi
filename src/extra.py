import pandas as pd


def print_metrics(metrics):
    a = metrics.accuracy
    p, p_macro = metrics.precision
    r, r_macro = metrics.recall
    f, f_macro = metrics.f1
    depth = metrics.depth
    df1 = pd.DataFrame.from_dict(dict(
        ("Room " + str(i + 1), [p[i], r[i], f[i], None, None]) for i in range(p.shape[0])), orient="index")
    df1 = df1.append(
        pd.Series([p_macro, r_macro, f_macro, a, depth], name="Macro"))
    df1 = df1.set_axis(["Precission", "Recall", "F1",
                       "Accuracy", "Avg Depth"], axis=1, inplace=False)
    return df1


def print_confusion(confusion):
    df = pd.DataFrame(confusion)
    df = df.set_axis(["Room " + str(i + 1) +
                      " (P)" for i in range(confusion.shape[0])], axis=1, inplace=False)
    df = df.set_axis(
        ["Room " + str(i + 1)
         + " (A)" for i in range(confusion.shape[0])], axis=0, inplace=False)
    return df
