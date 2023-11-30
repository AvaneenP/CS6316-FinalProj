import sklearn.ensemble

class Boosting():
    def __init__(self, dt, n_estimators=100):
        self.n_estimators = n_estimators
        self.boost = sklearn.ensemble.AdaBoostClassifier(estimator=dt, n_estimators=n_estimators)
    
    def train(self, x, y):
        self.boost.fit(x, y)