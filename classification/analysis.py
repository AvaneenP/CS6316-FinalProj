from datetime import datetime
import os
import numpy as np
import json
from Boosting import Boosting
from RandomForest import RandomForest
from DecisionTree import DecisionTree
from LogisticRegression import LogisticRegression
from SVM import SVM
from KNN import KNN
from sklearn.model_selection import cross_val_score

def compileClassifiers():
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    C = [0.1, 1, 10, 100]

    n_Neighbors = [3, 5, 10, 15, 20]
    leaf_Sizes = [10, 20, 30, 40, 50]

    depth = [None, 5, 10, 15, 20]
    num_Trees = [10, 50, 100, 200, 500]

    clfs = []
    # With different hyperparameters
    for d in depth:
        for n in num_Trees:
            dt = DecisionTree(max_depth=d)
            clfs.append(dt.tree)
            clfs.append(Boosting(dt.tree, n_estimators=n).boost)
            clfs.append(RandomForest(dt.tree, n_estimators=n).forest)
    for c in C:
        clfs.append(LogisticRegression(C=c).lr)
    for k in kernels:
        for c in C:
            clfs.append(SVM(kernel=k, C=c).svm)
    for n in n_Neighbors:
        for l in leaf_Sizes:
            clfs.append(KNN(n_neighbors=n, leaf_size=l).knn)
    return clfs

def mapClassifierToParams(name, cIdx):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    C = [0.1, 1, 10, 100]

    n_Neighbors = [3, 5, 10, 15, 20]
    leaf_Sizes = [10, 20, 30, 40, 50]

    depth = [None, 5, 10, 15, 20]
    num_Trees = [10, 50, 100, 200, 500]
    match(name):
        case "DecisionTreeClassifier":
            return f"max_depth={depth[cIdx // 15]}, n_estimators={num_Trees[cIdx // 3 % 5]}"
        case "AdaBoostClassifier":
            return f"max_depth={depth[cIdx // 15]}, n_estimators={num_Trees[cIdx // 3 % 5]}"
        case "RandomForestClassifier":
            return f"max_depth={depth[cIdx // 15]}, n_estimators={num_Trees[cIdx // 3 % 5]}"
        case "LogisticRegression":
            cIdx = cIdx - 75
            return f"C={C[cIdx]}"
        case "SVC":
            cIdx = cIdx - 79
            return f"kernel={kernels[cIdx // 4]}, C={C[cIdx % 4]}"
        case "KNeighborsClassifier":
            cIdx = cIdx - 104
            return f"n_neighbors={n_Neighbors[cIdx // 5]}, leaf_size={leaf_Sizes[cIdx % 5]}"

if __name__ == "__main__":
    dataset1 = np.loadtxt("project_dataset1.txt", delimiter="\t")
    x1 = dataset1[:, :-1]
    y1 = dataset1[:, -1]

    dataset2 = np.loadtxt("project_dataset2.txt", delimiter="\t", converters={4: lambda y: (1.0 if y == 'Absent' else 0.0)})
    x2 = dataset2[:, :-1]
    y2 = dataset2[:, -1]

    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    classifiers = compileClassifiers()

    bestScores = {
        'set1_accuracy': {},
        'set1_precision': {},
        'set1_recall': {},
        'set1_f1': {},
        'set1_roc_auc': {},
        'set2_accuracy': {},
        'set2_precision': {},
        'set2_recall': {},
        'set2_f1': {},
        'set2_roc_auc': {}
    }

    with open(f"classification/logs/{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.txt", "w+") as f:
        f.write("Dataset 1\n")
        for sc in scoring_metrics:
            for cIdx, classifier in enumerate(classifiers):
                s1 = cross_val_score(classifier, x1, y1, cv=10, scoring=sc)
                for i, s in enumerate(s1):
                    try:
                        if s > bestScores[f"set1_{sc}"][i][0]:
                            bestScores[f"set1_{sc}"][i] = (s, f"{classifier.__class__.__name__}: {mapClassifierToParams(classifier.__class__.__name__, cIdx)}")
                    except KeyError:
                        bestScores[f"set1_{sc}"][i] = (s, f"{classifier.__class__.__name__}: {mapClassifierToParams(classifier.__class__.__name__, cIdx)}")

                f.write(f"{classifier.__class__.__name__}, {sc}\n{s1}\n")
        
        f.write("\n" + 40 * "=" + "\n")
        f.write("Dataset 2\n")
        for sc in scoring_metrics:
            for cIdx, classifier in enumerate(classifiers):
                s2 = cross_val_score(classifier, x2, y2, cv=10, scoring=sc)

                for i, s in enumerate(s2):
                    try:
                        if s > bestScores[f"set2_{sc}"][i][0]:
                            bestScores[f"set2_{sc}"][i] = (s, f"{classifier.__class__.__name__}: {mapClassifierToParams(classifier.__class__.__name__, cIdx)}")
                    except KeyError:
                        bestScores[f"set2_{sc}"][i] = (s, f"{classifier.__class__.__name__}: {mapClassifierToParams(classifier.__class__.__name__, cIdx)}")

                f.write(f"{classifier.__class__.__name__}, {sc}\n{s2}\n")
        
        f.write("\n" + 40 * "=" + "\n")
        f.write(json.dumps(bestScores, indent=4))
