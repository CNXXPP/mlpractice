from numpy import *
import operator

from pip._vendor.distlib.compat import raw_input


def createDataSet():
    group = array([[1.0, 1.1],
                   [1.0, 1.0],
                   [0, 0],
                   [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

group, labels = createDataSet()

#目标样本 数据集 数据集标签
def classfiy0(inX,dataSet,labels,k):
    m = dataSet.shape[0] #样本数m
    #print(m)
    diffMat = tile(inX, (m, 1)) - dataSet #待测值与样本值的差
    sqDiffMat = diffMat ** 2
    #print(sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    #print(sqDistances)
    distances = sqDistances ** 0.5
    sortedDistIdx = distances.argsort()
    #print(sortedDistIdx)
    classcount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIdx[i]]
        classcount[voteLabel] = classcount.get(voteLabel, 0) + 1
   # print(classcount)
    sortedClassCount = sorted(classcount.items(), key=operator.itemgetter(1), reverse=True)
    #print(sortedClassCount)
    return sortedClassCount[0][0]



#print(classfiy0([0,0],group,labels,3))
