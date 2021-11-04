from decision_tree import DecisionTree
import model_eval
import numpy as np
import extra

if __name__ == '__main__':
    clean_dataset = np.loadtxt("datasets/clean_dataset.txt")
    noisy_dataset = np.loadtxt("datasets/noisy_dataset.txt")

    model = DecisionTree()

    cv_clean_metrics = model_eval.eval_cross_validation(
        clean_dataset, model, 10)

    print(extra.print_metrics(cv_clean_metrics))
