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
    #p0Num = zeros(numWords); p1Num = zeros(numWords) #不同分类下 单词表中每个单词出现的次数
    #p0Denom = 0.0; p1Denom = 0.0
    #由于p(w|c1)=p(w1|c1)*p(w2|c1)··· 有一个为0 则结果为0
    p0Num = ones(numWords); p1Num = ones(numWords) #不同分类下 单词表中每个单词出现的次数
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDoc): #遍历每个样本
        if trainGategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i]) #侮辱性词汇样本 单词表对应单词出现总次数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])  # 非侮辱性词汇样本 单词表对应单词出现总次数
    #p1Vec = p1Num / p1Denom
    #p0Vec = p0Num / p0Denom
    #结果太小 将其取对数
    p1Vec = log(p1Num / p1Denom)
    p0Vec = log(p0Num / p0Denom)
    return p0Vec, p1Vec, pAbusive

#p1Vec [0.1 0.2 ·····]
# 0.1代表单词表中第一个单词"posting" 在分类为1（包含侮辱性词汇）中出现的概率 即：P（w1|c1）
# p(w|c1)

trainMat = []

for post in postingList:
    trainMat.append(setOfWords2Vec(vocabList,post))

p0v,p1v,pAb = trainNBO(trainMat,classVec)
#print(p0v)
#print(p1v)
#print(pAb)

#朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0v,p1v,pclass1):
    #p(w|c1)=p(w1|c1)*p(w2|c1)···p(c1) 由于训练数据集时结果取对数
    #log(p(w|c1)) = log(p(w1|c1)) + log(p(w2|c1)) + ··· + log(p(c1))
    # p0 p1原本需要/p(w) 由于只需要比较p0 p1大小 可省略该步骤
    p1 = sum(vec2Classify * p1v) + log(pclass1)
    p0 = sum(vec2Classify * p0v) + log(1 - pclass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testNB():
    testEntry = ['love','my','dalmation']
    testVec = array(setOfWords2Vec(vocabList,testEntry))
    print(testEntry,'classified as: ',classifyNB(testVec,p0v,p1v,pAb))
    testEntry = ['stupid', 'garbage']
    testVec = array(setOfWords2Vec(vocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(testVec, p0v, p1v, pAb))

testNB()