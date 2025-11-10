"""
Question 2:
K Nearest Neighbor Classification for Digits
"""
import numpy as np
from sklearn.datasets import load_digits
from collections import Counter
import random

# Set random seed
RNG_SEED = 1337
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

def split_per_class(X, Y):
    # Split each class: 50 for test, rest for train
    classes = np.unique(Y)
    train_idx = []
    test_idx = []

    # go through each class and split
    for c in classes:
        idx = np.where(Y == c)[0]
        np.random.shuffle(idx)
        test_idx += list(idx[:50])
        train_idx += list(idx[50:])

    # return train and test array
    return np.array(train_idx), np.array(test_idx)


def l2_distance(a, b):
    # Compute pairwise Euclidean distance
    n = a.shape[0]
    m = b.shape[0]
    d = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        diff = b - a[i]
        d[i] = np.sqrt(np.sum(diff * diff, axis=1))
    return d


def predict_knn(X_train, X_test, Y_train, k):
    D = l2_distance(X_test, X_train)
    y_pred = []

    for i in range(D.shape[0]):
        distances = D[i]
        sorted_idx = np.argsort(distances)
        nearest = sorted_idx[:k]
        labels = Y_train[nearest]
        counts = Counter(labels)
        best_label = max(counts, key=counts.get)
        y_pred.append(best_label)

    return np.array(y_pred)


def accuracy(y_true, y_pred):
    # returns accuracy of predictions 
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

def main():
    data = load_digits()
    X = data.images.reshape(len(data.images), -1).astype(np.float32)
    Y = data.target.astype(np.int64)

    # grab test and train data
    train_idx, test_idx = split_per_class(X, Y)
    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # check each k method as per problem description
    for k in [3, 5, 7]:
        Y_pred = predict_knn(X_train, X_test, Y_train, k)
        acc = accuracy(Y_test, Y_pred)
        print(f"k={k:>2} | accuracy={acc * 100:.2f}%")


if __name__ == "__main__":
    main()
