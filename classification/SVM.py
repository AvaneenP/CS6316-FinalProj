import sklearn.svm

class SVM():
    def __init__(self, kernel='rbf', C=1.0):
        self.kernel = kernel
        self.C = C
        self.svm = sklearn.svm.SVC(kernel=kernel, C=C)
    
    def train(self, x, y):
        self.svm.fit(x, y)