import sklearn.tree

class DecisionTree():
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth)

    def fit(self, x, y):
        self.tree.fit(x, y)