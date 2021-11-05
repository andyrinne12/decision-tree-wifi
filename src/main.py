import numpy as np

import extra
import model_eval
from decision_tree import DecisionTree

if __name__ == '__main__':

    # Load datasets as numpy arrays
    clean_dataset = np.loadtxt("datasets/clean_dataset.txt")
    noisy_dataset = np.loadtxt("datasets/noisy_dataset.txt")

    # The DecisionTree has to be first instantiated
    model = DecisionTree()

    model.fit(training_set=clean_dataset)

    # Draw the tree in a text-based manner
    # print(model)
    # print('\n')

    x_test = noisy_dataset[:, :-1]
    y_test = noisy_dataset[:, -1]

    # The model HAS to be trained first, predicts a list of labels
    y_predict = model.predict(x_test)

    # Generate confusion matrix for the prediction
    confusion = model_eval.confusion_matrix(y_test, y_predict)

    # Generate metrics based on the confusion matrix
    metrics = model_eval.metrics_from_confusion(confusion, model.depth)

    # To prune the TRAINED model using the set used for training and a validation set
    # You can use the evaluation means above to test the performance difference after pruning
    # model.prune(clean_dataset, noisy_dataset[:1000])

    # Print the confusion matrix and the extracted metrics using pandas dfs
    print(extra.print_confusion(confusion))
    print('\n')
    print(extra.print_metrics(metrics))
    print('\n')

    print('\n')

    ##
    # The whole pipeline above is implemented for the two tasks of the coursework

    # Perform cross validation on both datasets
    cv_clean_metrics = model_eval.eval_cross_validation(
        clean_dataset, model, 10)
    cv_noisy_metrics = model_eval.eval_cross_validation(
        noisy_dataset, model, 10)

    print(extra.print_metrics(cv_clean_metrics))
    print('\n')
    print(extra.print_confusion(cv_noisy_metrics.confusion))
    print('\n')

    print('\n')

    print(extra.print_metrics(cv_clean_metrics))
    print('\n')
    print(extra.print_confusion(cv_noisy_metrics.confusion))
    print('\n')

    print('\n')

    # Perform nested cross validation on both datasets while pruning, having the pruning process as a `hyperparamter`
    ncv_prune_clean_metrics = model_eval.eval_prune_nested_cross_validation(
        clean_dataset, model, 10)
    ncv_prune_noisy_metrics = model_eval.eval_prune_nested_cross_validation(
        noisy_dataset, model, 10)

    print(extra.print_metrics(ncv_prune_clean_metrics))
    print('\n')
    print(extra.print_confusion(ncv_prune_clean_metrics.confusion))
    print('\n')

    print('\n')

    print(extra.print_metrics(ncv_prune_noisy_metrics))
    print('\n')
    print(extra.print_confusion(ncv_prune_noisy_metrics.confusion))
    print('\n')

    print('\n')
