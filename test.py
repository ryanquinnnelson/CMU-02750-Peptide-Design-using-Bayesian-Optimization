import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RfWrapper(RandomForestRegressor):

    def __init__(self, regressor):
        self.regressor = regressor

    def predict(self, X, return_std=False):
        if return_std:
            ys = np.array([e.predict(X) for e in self.estimators_])
            return np.mean(ys, axis=0).ravel(), np.std(ys, axis=0).ravel()
        return super().predict(X).ravel()


X_training = np.array([1,2,3,4,5])
y_training = np.array([1,2,3,4,5])

a = RandomForestRegressor(n_estimators=20,max_depth=6,random_state=0)#.fit(X_training,y_training)
b = RfWrapper(a)
b.