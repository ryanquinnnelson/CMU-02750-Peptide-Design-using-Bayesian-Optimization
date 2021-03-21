"""
Defines grid search process for hyperparameter tuning.
"""
import copy
import csv
import packages.activelearning.activelearning as al


def _append_results_to_file(filename, fields=None, rows=None):
    """
    Appends results to csv file. Creates file if it does not exist.
    :param filename: Name of the file.
    :param fields: Header row for csv file. Default is None.
    :param rows: Data rows for csv file. Default is None.
    :return: None
    """
    with open(filename, 'a') as f:

        write = csv.writer(f)

        if fields:
            write.writerow(fields)

        if rows:
            write.writerows(rows)


def _create_row_gs1(kernel, n_learner, n_initial, n_queries, row_prefix, details_func, r2):
    """
    Builds a 2D list representing a new row of data for a csv file. This is used with grid_search_1.

    :param kernel: Kernel used by regressor.
    :param n_learner: Number of members in Committee.
    :param n_initial: Number of initial training instances.
    :param n_queries: Number of queries.
    :param row_prefix: List, unchanging portion of data row in csv file.
    :param details_func: Python function returning a list of hyperparameter details. Accepts kernel as argument.
    :param r2: R2 score.
    :return: 2D list.
    """
    kernel_hyperparms = details_func(kernel)
    meta = [n_learner, n_initial, n_queries, r2]
    row = row_prefix + kernel_hyperparms + meta

    return [row]


def _create_row_gs2(n_estimator, max_depth, n_initial, n_queries, row_prefix, r2):
    """
    Builds a 2D list representing a new row of data for a csv file. This is used with grid_search_2.

    :param n_estimator: Number of estimators in the random forest.
    :param max_depth: Depth limit for the random forest.
    :param n_initial: Number of initial training instances.
    :param n_queries: Number of queries.
    :param row_prefix: List, unchanging portion of data row in csv file.
    :param r2: R2 score.
    :return: 2D list.
    """
    hyperparms = [n_estimator, max_depth]
    meta = [1, n_initial, n_queries, r2]
    row = row_prefix + hyperparms + meta

    return [row]


def grid_search_1(kernels, n_learners, n_initials, X_pool, y_pool, X_test, y_test, n_queries,
                  seed, filename, fields, row_prefix, details_func):
    """
    Grid search over (kernels, n_learners, n_initials) using active learning.
    Uses CommitteeRegressor for active learner with GaussianProcessRegressors as members.
    Appends results to file after each run combination. Experiment can be halted at any time without losing results.

    :param kernels: Collection of kernels to use in grid search.
    :param n_learners: Collection of n_learner counts (size of committee) to use in grid search.
    :param n_initials: Collection of n_initial counts (number of initial training instances) to use in grid search.
    :param X_pool: Training set.
    :param y_pool: Regression values for training set.
    :param X_test: Test set.
    :param y_test: Regression values for test set.
    :param n_queries: Number of queries to perform in active learning.
    :param seed: Random seed for reproducibility.
    :param filename: Name of the file to save grid search results.
    :param fields: Header row for csv file of results.
    :param row_prefix: Unchanging portion of data row in csv file.
    :param details_func: Python function returning a list of hyperparameter details. Accepts kernel as argument.
    :return: None
    """

    # append fields as first row in file
    _append_results_to_file(filename, fields)

    # perform grid search
    for kernel in kernels:
        for n_learner in n_learners:
            for n_initial in n_initials:
                # make a copy of the data for use in this experiment
                X_pool_gs = copy.deepcopy(X_pool)
                y_pool_gs = copy.deepcopy(y_pool)

                # build learner
                committee = al.build_committee(kernel, n_learner, n_initial, X_pool_gs, y_pool_gs, seed)

                # perform active learning
                al.run_active_learner_regression(committee, X_pool_gs, y_pool_gs, n_queries)

                # score model
                r2 = al.score_regression_model(committee, X_test, y_test)

                # create row for file
                row = _create_row_gs1(kernel, n_learner, n_initial, n_queries, row_prefix, details_func, r2)

                # append to file
                _append_results_to_file(filename, rows=row)

                # output to console for tracking progress
                print('{}|{}|{}|{}|{}'.format(kernel, n_learner, n_initial, n_queries, r2))


def grid_search_2(n_estimators, max_depths, n_initials, X_pool, y_pool, X_test, y_test, n_queries,
                  seed, filename, fields, row_prefix):
    """
    Grid search over (kernels, n_learners, n_initials) using active learning.
    Uses Random Forest Regressor for active learner.
    Appends results to file after each run combination. Experiment can be halted at any time without losing results.

    :param n_estimators: Collection of n_estimators values to use in grid search.
    :param max_depths: Collection of max_depth values to use in grid search.
    :param n_initials: Collection of n_initial counts (number of initial training instances) to use in grid search.
    :param X_pool: Training set.
    :param y_pool: Regression values for training set.
    :param X_test: Test set.
    :param y_test: Regression values for test set.
    :param n_queries: Number of queries to perform in active learning.
    :param seed: Random seed for reproducibility.
    :param filename: Name of the file to save grid search results.
    :param fields: Header row for csv file of results.
    :param row_prefix: Unchanging portion of data row in csv file.
    :return:
    """
    # append fields as first row in file
    _append_results_to_file(filename, fields)

    # perform grid search
    for n_estimator in n_estimators:
        for max_depth in max_depths:
            for n_initial in n_initials:
                # make a copy of the data for use in this experiment
                X_pool_gs = copy.deepcopy(X_pool)
                y_pool_gs = copy.deepcopy(y_pool)

                # build learner
                regressor = al.build_random_forest_regressor(n_estimator, max_depth, n_initial, X_pool_gs, y_pool_gs,
                                                             seed)
                # perform active learning
                al.run_active_learner_regression(regressor, X_pool_gs, y_pool_gs, n_queries)

                # score model
                r2 = al.score_regression_model(regressor, X_test, y_test)

                # create row for file
                row = _create_row_gs2(n_estimator, max_depth, n_initial, n_queries, row_prefix, r2)

                # append to file
                _append_results_to_file(filename, rows=row)

                # output to console for tracking progress
                print('RandomForestRegressor(n_estimators={},max_depth={})|{}|{}|{}'.format(n_estimator, max_depth,
                                                                                            n_initial, n_queries, r2))
