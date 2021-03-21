"""
Helper functions to perform active learning with modAL learners.
"""
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from modAL.models import ActiveLearner, CommitteeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from modAL.disagreement import max_std_sampling
from modAL.uncertainty import margin_sampling


# provided wrapper class
class RfWrapper(RandomForestRegressor):  # superclass
    def predict(self, X, return_std=False):
        if return_std:
            ys = np.array([e.predict(X) for e in self.estimators_])
            return np.mean(ys, axis=0).ravel(), np.std(ys, axis=0).ravel()
        return super().predict(X).ravel()


def get_next_sample(learner, X, y):
    # call the query strategy defined in the learner to obtain a new sample
    query_idx, query_sample = learner.query(X)

    # modify indexing to interpret as collection of one element with d features
    query_sample_reshaped = query_sample.reshape(1, -1)

    # obtain the query label
    query_label = y[query_idx]

    # modify indexing to interpret as 1D array of one element
    query_label_reshaped = query_label.reshape(1, )

    return query_sample_reshaped, query_label_reshaped, query_idx


def run_active_learner_regression(learner, X_pool, y_pool, n_queries):
    # perform active learning
    for q in range(n_queries):
        # get sample
        X_sample, y_sample, query_idx = get_next_sample(learner, X_pool, y_pool)

        # use new sample to update the model
        learner.teach(X_sample, y_sample)

        # remove labeled instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)


def run_and_score_active_learner_regression(learner, X_pool, y_pool, X_test, y_test, n_queries):
    history = []

    # score before starting
    r2 = score_regression_model(learner, X_test, y_test)
    history.append(r2)

    # perform active learning
    for q in range(n_queries):
        # get sample
        X_sample, y_sample, query_idx = get_next_sample(learner, X_pool, y_pool)

        # use new sample to update the model
        learner.teach(X_sample, y_sample)

        # remove labeled instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)

        # score learner
        r2 = score_regression_model(learner, X_test, y_test)
        history.append(r2)

    return history


def build_committee(kernel, n_learner, n_initial, X_pool, y_pool, seed):
    # get initial training set for each learner
    initial_idx = []
    for i in range(n_learner):
        initial_idx.append(np.random.choice(len(X_pool), size=n_initial, replace=False))

    # initialize learners for Committee
    learner_list = [ActiveLearner(
        estimator=GaussianProcessRegressor(kernel, random_state=seed),
        X_training=X_pool[idx],
        y_training=y_pool[idx]) for idx in initial_idx]

    # create Committee
    committee = CommitteeRegressor(learner_list=learner_list, query_strategy=max_std_sampling)

    return committee


def score_regression_model(learner, X_test, y_test):
    y_pred = learner.predict(X_test, return_std=False)
    r2 = r2_score(y_test, y_pred)  # y_true, y_pred
    return r2


def build_random_forest_regressor(n_estimators, max_depth, n_initial, X_pool, y_pool, seed):
    initial_idx = np.random.choice(len(X_pool), size=n_initial, replace=False)

    # https://modal-python.readthedocs.io/en/latest/content/examples/active_regression.html
    def GP_regression_std(regressor, X):
        _, std = regressor.predict(X, return_std=True)
        query_idx = np.argmax(std)
        return query_idx, X[query_idx]

    regressor = ActiveLearner(
        estimator=RfWrapper(n_estimators=n_estimators, max_depth=max_depth, random_state=seed),
        query_strategy=GP_regression_std,
        X_training=X_pool[initial_idx],
        y_training=y_pool[initial_idx]
    )

    return regressor
