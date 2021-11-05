from decision_tree import DecisionTree
import model_eval
import numpy as np
import extra
import sys


def main(args):

    model = DecisionTree()

    model.fit(adwwawd)
    # model.predict(awdwda)

    accuracy = model_eval.evalluate(test, model)
    confusion =

    extra.print_confusion()
    extra.print_metrics()

    print(args)

    clean_dataset = np.loadtxt("datasets/clean_dataset.txt")
    noisy_dataset = np.loadtxt("datasets/noisy_dataset.txt")

    model = DecisionTree()

    cv_clean_metrics = model_eval.eval_cross_validation(
        clean_dataset, model, 10)

    print(extra.print_metrics(cv_clean_metrics))


if __name__ == '__main__':
    main(sys.argv[1:])
