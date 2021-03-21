"""
Helper functions to perform active learning with modAL learners.
"""
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from modAL.models import ActiveLearner, CommitteeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from modAL.disagreement import max_std_sampling


class RfWrapper(RandomForestRegressor):  # superclass
    """
    Wrapper class for RandomForestRegressor which modifies predict() method to include second argument return_std.
    This argument is expected by
    modAL library for active learning regression. Provided by course instructors.
    """

    def predict(self, X, return_std=False):
        if return_std:
            ys = np.array([e.predict(X) for e in self.estimators_])
            return np.mean(ys, axis=0).ravel(), np.std(ys, axis=0).ravel()
        return super().predict(X).ravel()


def get_next_sample(learner, X, y):
    """
    Queries the pool X of data and selects a new sample using the query_strategy of the ActiveLearner.

    :param learner:the ActiveLearner within which a query_strategy is defined.
    :param X:the pool of data from which to select a sample. This is a numpy array of feature instances.
    :param y:the pool of labels corresponding to X instances. This is a numpy array of labels.
    :return: (X,y,idx) tuple of the selected sample, where idx is the index of the selected sample.
    """
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
    """
    Performs active learning using given ActiveLearner. Runs for
    the given number of queries. Each iteration draws from the pool of
    data using the learner's query_strategy, updates the model, then removes queried instance from the data pool.

    :param learner: the ActiveLearner
    :param X_pool:the pool of feature data from which to sample
    :param y_pool:the labels corresponding to the X_pool
    :param n_queries:the number of queries (iterations) to execute during active learning
    :return: None
    """
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
    """
    Performs active learning using given ActiveLearner. Runs for
    the given number of queries. Each iteration draws from the pool of
    data using the learner's query_strategy, updates the model, removes queried instance from the data pool,
    then scores the model against given test data.

    :param learner: the ActiveLearner
    :param X_pool:the pool of feature data from which to sample
    :param y_pool:the labels corresponding to the X_pool
    :param X_test:the collection of data with which to score the model
    :param y_test:the labels corresponding to the X_test
    :param n_queries:the number of queries (iterations) to execute during active learning
    :return: List of scores for each query.
    """
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
    """
    Constructs a CommitteeRegressor of ActiveLearner members based on provided parameters.
    Uses GaussianProcessRegressors as committee members.
    Defines initial training set of random instances for each learner in the committee.

    :param kernel: Kernel to be used in Gaussian Process regressors.
    :param n_learner: Number of members in the committee.
    :param n_initial: Number of initial training instances for each committee member.
    :param X_pool:the pool of feature data from which to sample
    :param y_pool:the labels corresponding to the X_pool
    :param seed: Random seed for reproducibility.
    :return: CommitteeRegressor
    """
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
    """
    Calculates R2 score for given learner.

    :param learner: ActiveLearner
    :param X_test: Test set
    :param y_test: Regression values for test set
    :return: Float, r2 score.
    """
    y_pred = learner.predict(X_test, return_std=False)
    r2 = r2_score(y_test, y_pred)  # y_true, y_pred
    return r2


def build_random_forest_regressor(n_estimators, max_depth, n_initial, X_pool, y_pool, seed):
    """
    Constructs RandomForestRegressor ActiveLearner with custom query_strategy.
    Defines initial training set of random instances for learner.

    :param n_estimators: Number of estimators in Random Forest.
    :param max_depth: Max depth of trees in Random Forest.
    :param n_initial: Number of initial training instances for each committee member.
    :param X_pool:the pool of feature data from which to sample
    :param y_pool:the labels corresponding to the X_pool
    :param seed: Random seed for reproducibility.
    :return: ActiveLearner
    """
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
