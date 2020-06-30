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

#词袋模型 
def bagOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word:%s is not in my vocabLis!" % word)
    return returnVec

postingList, classVec = loadDataSet()
vocabList = createVocabList(postingList)

#print(vocabList)
#print(setOfWords2Vec(vocabList,postingList[0]))

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
    testVec = array(setOfWords2Vec(vocabList, testEntry))
    print(testEntry,'classified as: ', classifyNB(testVec,p0v,p1v,pAb))
    testEntry = ['stupid', 'garbage']
    testVec = array(setOfWords2Vec(vocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(testVec, p0v, p1v, pAb))

#testNB()
#垃圾邮件分类
#使用正则表达式切分文本为列表 \W 匹配字母数字下划线汉字
import re
regEx = re.compile(r'\b[\.,\s\n\r\n]+?\b')
mySent='this book is the best book on python or ML. I have ever laid eyes upon'
listOfTokens = regEx.split(mySent)
listOfTokens = [tok.lower() for tok in listOfTokens]

emailTest = open('/home/pi/mlpractice/machinelearninginaction/Ch04/email/ham/6.txt',encoding='ISO-8859-14').read()
listOfTokens = regEx.split(emailTest)
#print(listOfTokens)
#文本解析及完整的垃圾邮件测试函数
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\b[\.,\s\n\r\n]+?\b',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
#print(textParse(emailTest))
def spamTest():
    docList=[]; classList = []; fullText = []
    for i in range(1,26):
        #垃圾邮件
        wordList = textParse(open('/home/pi/mlpractice/machinelearninginaction/Ch04/email/spam/%d.txt' % i,encoding='ISO-8859-14').read())
        #print(docList)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        
        #正常邮件
        wordList = textParse(open('/home/pi/mlpractice/machinelearninginaction/Ch04/email/ham/%d.txt' % i,encoding='ISO-8859-14').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50));testSet = [] #训练集50个 随机选10个作为测试集
    for i in range(10):
        randIdx = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIdx])
        del(trainingSet[randIdx])
    trainMat=[];trainingClasses=[]
    for docIdx in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIdx]))
        trainingClasses.append(classList[docIdx])
    #print(trainMat)
   # print(array(trainMat))
    p0v,p1v,pSpam=trainNBO(array(trainMat),array(trainingClasses))
    errorCnt=0
    for docIdx in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIdx])
        if classifyNB(wordVector,p0v,p1v,pSpam) != classList[docIdx]:
            errorCnt+=1
            print('classify error',docList[docIdx])
    print('the error rate is :',float(errorCnt/len(testSet)))
    
#spamTest()
    
#从个人广告中获取区域倾向
import feedparser

#ny = feedparser.parse('https://www.nasa.gov/rss/dyn/image_of_the_day.rss')
#print(len(ny['entries']))
# rss源分类器及高频词去除函数
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed2):
    docList=[];classList=[];fullText=[]
    minLen = min(len(feed1['entries']),len(feed2['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        
        wordList = textParse(feed2['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    #top30Words = calcMostFreq(vocabList,fullText)
    #for pairW in top30Words:
        # pairW = {'word':3} 移除出现频率最高的30个词
    #    if pairW[0] in vocabList:vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen));testSet=[]
    for i in range(20):
        randIdx = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIdx])
        del(trainingSet[randIdx])
    trainMat=[];trainingClasses=[]
    for docIdx in trainingSet:
        trainMat.append(bagOfWords2Vec(vocabList,docList[docIdx]))
        trainingClasses.append(classList[docIdx])
    p0v,p1v,pSpam=trainNBO(array(trainMat),array(trainingClasses))
    errorCnt=0
    for docIdx in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIdx])
        if classifyNB(wordVector,p0v,p1v,pSpam) != classList[docIdx]:
            errorCnt+=1
            print('classify error',docList[docIdx])
    print('the error rate is :',float(errorCnt/len(testSet)))
    return vocabList,p0v,p1v

feed1 = feedparser.parse('https://www.nasa.gov/rss/dyn/image_of_the_day.rss')
feed2 = feedparser.parse('http://www.cppblog.com/kevinlynx/category/6337.html/rss')
print(len(feed1['entries']))
print(len(feed2['entries']))

localWords(feed1,feed2)