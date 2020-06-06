from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
from pip._vendor.distlib.compat import raw_input

from knn import classfiy0

dict = {'didntLike':1, 'smallDoses':2, 'largeDoses': 3} #1代表不喜欢,2代表魅力一般,3代表极具魅力
def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = [] #返回的分类标签向量
    idx = 0
    for line in arrayOlines:
        line = line.strip()# 去除首尾空格
        listFromLine = line.split("\t")
        returnMat[idx, :] = listFromLine[0:3]
        classLabelVector.append(dict[listFromLine[3]])
        idx += 1
    return returnMat, classLabelVector

datingDataMat, datingLabels = file2matrix("datingTestSet.txt")

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],15.0*array(datingLabels),15.0*array(datingLabels))
#plt.show()

#归一化特征值 newVal = (old - min) / (max - min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet / tile(ranges,(m,1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    m = normDataSet.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classfiy0(normDataSet[i,:],normDataSet[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("分类器测试结果：%d,实际结果：%d"%(classifierResult,datingLabels[i]))
        if (classifierResult != datingLabels[i]):errorCount+=1.0
    print("测试集数目：%d,错误数目：%d,总误差率：%f"%(numTestVecs,errorCount,errorCount/float(numTestVecs)))
#print(autoNorm(datingDataMat))
#datingClassTest()


def classifyPersion():
    resultList=['没兴趣', '一点点兴趣', '非常感兴趣']
    percentTats = float(raw_input("玩视频游戏所消耗时间占比?"))
    ffmiles = float(raw_input("每年获得的飞行常客里程数?"))
    iceCream = float(raw_input("每周消费的冰激淋公升?"))
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    inArray = [percentTats,ffmiles,iceCream]
    classifierResult = classfiy0((inArray - minVals)/ranges, normDataSet[:, :], datingLabels, 3)
    print("喜欢这个人的可能性：",classifierResult,resultList[classifierResult-1])
#classifyPersion()

def img2Vector(filename):
    vector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            vector[0, 32*i+j] = int(lineStr[j])
    return vector

#vector = img2Vector("digits/testDigits/0_0.txt")
#print(vector[0, 0:32])

def handWritingClassTest():
    hwLabels = []
    trainFileList = listdir("digits/trainingDigits")
    m = len(trainFileList)
    trainMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        trainMat[i, :] = img2Vector("digits/trainingDigits/%s" % fileNameStr)
    testfileList = listdir("digits/testDigits")
    errorCount = 0.0
    mTest = len(testfileList)
    for i in range(mTest):
        fileNameStr = testfileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        vectTest = img2Vector("digits/trainingDigits/%s" % fileNameStr)
        classifierResult = classfiy0(vectTest, trainMat, hwLabels, 3)
        print("分类器测试结果：%d,实际结果：%d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("测试集数目：%d,错误数目：%d,总误差率：%f" % (mTest, errorCount, errorCount / float(mTest)))

handWritingClassTest()