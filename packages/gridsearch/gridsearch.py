"""
Defines grid search process for hyperparameter tuning.
"""
import copy
import csv
import packages.activelearning.activelearning as al


def _append_results_to_file(filename, fields=None, rows=None):
    with open(filename, 'a') as f:

        write = csv.writer(f)  # using csv.writer method from CSV package

        if fields:
            write.writerow(fields)

        if rows:
            write.writerows(rows)


def _create_row_gs1(kernel, n_learner, n_initial, n_queries, row_prefix, details_func, r2):
    kernel_hyperparms = details_func(kernel)
    meta = [n_learner, n_initial, n_queries, r2]
    row = row_prefix + kernel_hyperparms + meta

    return [row]


def _create_row_gs2(n_estimator,max_depth, n_initial, n_queries, row_prefix, r2):
    hyperparms = [n_estimator, max_depth]
    meta = [1, n_initial, n_queries, r2]
    row = row_prefix + hyperparms + meta

    return [row]


def grid_search_1(kernels, n_learners, n_initials, X_pool, y_pool, X_test, y_test, n_queries,
                  seed, filename, fields, row_prefix, details_func):
    """
    Grid search over (kernels, n_learners, n_initials).
    Appends results to file after each run combination. Experiment can be halted at any time without losing results.
    Uses CommitteeRegressor for active learner with GaussianProcessRegressors as members.

    :param kernels:
    :param n_learners:
    :param n_initials:
    :param X_pool:
    :param y_pool:
    :param X_test:
    :param y_test:
    :param n_queries:
    :param seed:
    :param filename:
    :param fields:
    :param row_prefix:
    :param details_func:
    :return:
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
    Grid search over (kernels, n_learners, n_initials).
    Appends results to file after each run combination. Experiment can be halted at any time without losing results.
    Uses Random Forest Regressor for active learner.

    :param kernels:
    :param n_learners:
    :param n_initials:
    :param X_pool:
    :param y_pool:
    :param X_test:
    :param y_test:
    :param n_queries:
    :param seed:
    :param filename:
    :param fields:
    :param row_prefix:
    :param details_func:
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
                row = _create_row_gs2(n_estimator,max_depth, n_initial, n_queries, row_prefix, r2)

                # append to file
                _append_results_to_file(filename, rows=row)

                # output to console for tracking progress
                print('RandomForestRegressor(n_estimators={},max_depth={})|{}|{}|{}'.format(n_estimator,max_depth, n_initial, n_queries, r2))
