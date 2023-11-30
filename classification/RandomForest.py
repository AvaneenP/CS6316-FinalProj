import sklearn.ensemble

class RandomForest():
    def __init__(self, dt, n_estimators=100):
        self.n_estimators = n_estimators
        self.forest = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators)
        self.forest.estimator = dt
    
    def train(self, x, y):
        self.forest.fit(x, y)