import sklearn.linear_model

class LogisticRegression():
    def __init__(self, C=1.0):
        self.lr = sklearn.linear_model.LogisticRegression(C=C)
    
    def train(self, x, y):
        self.lr.fit(x, y)