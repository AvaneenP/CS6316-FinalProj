import sklearn.neighbors

class KNN():
    def __init__(self, n_neighbors=5, leaf_size=30):
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size)
    
    def train(self, x, y):
        self.knn.fit(x, y)