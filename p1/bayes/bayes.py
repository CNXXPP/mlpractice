from numpy import *
#创建一个dataset

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

#返回数据集的所有词汇列表
def createVocabList(dataSet):
    vocabSet = set([])
    for doc in dataSet:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)

# vocabList所有词汇列表
# inputSet 某个文档
# 将文档转化为向量 长度为词汇表长度 0是未出现 1是已出现 [1 0 1 0 ······]
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:%s is not in my vocabLis!" % word)
    return returnVec

postingList, classVec = loadDataSet()
vocabList = createVocabList(postingList)

print(vocabList)
print(setOfWords2Vec(vocabList,postingList[0]))

#朴素贝叶斯分类器训练函数
def trainNBO(trainMatrix,trainGategory):
    numTrainDoc = len(trainMatrix) #样本数目
    numWords = len(trainMatrix[0]) #样本特征值数目
    pAbusive = sum(trainGategory) / float(numTrainDoc)  # 包含侮辱性词汇的样本概率
    p0Num = zeros(numWords); p1Num = zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDoc):
        if trainGategory[i] == 1:
            p1Num += trainMatrix[i]
