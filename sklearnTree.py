from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from DicisionTree import getdataset
import random
import graphviz
from IPython.display import Image
import pydotplus
def dicisionTree(data,label):
    random.shuffle(data)
    traindata = data[:425]
    testdata = data[425:]
    trainX = np.array(traindata)[..., :4]
    trainy = np.array(traindata)[..., 4]
    testX = np.array(testdata)[..., :4]
    testy = np.array(testdata)[..., 4]
    # 训练模型，限制树的最大深度4
    clf = DecisionTreeClassifier(max_depth=4, criterion='entropy')
    #拟合模型
    clf.fit(trainX, trainy)
    score = clf.score(testX, testy)
    with open("ID3.dot", 'w') as f:
        f = tree.export_graphviz(clf, feature_names=label, class_names=['survival','death'],out_file=f)
    print(score)

if __name__ == '__main__':
    data,label = getdataset('./Titanic_dataset.txt')
    dicisionTree(data,label)