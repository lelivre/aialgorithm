from math import log
import operator
import plotTree

def calcShannonEnt(dataSet):
	numEntris = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntris
		shannonEnt -= prob * log(prob, 2)

	return shannonEnt

def createDataSet():
	dataSet = [[1, 1, 'yes'],
	[1, 1, 'yes'],
	[1, 0, 'no'],
	[0, 1, 'no'],
	[0, 1, 'no']
	]

	labels = ['no surfacing', 'flippers']

	return dataSet, labels

def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reduceFeatVec = featVec[:axis]
			reduceFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reduceFeatVec)
	return retDataSet

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0; bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob *calcShannonEnt(subDataSet)
		infoGain =baseEntropy - newEntropy

		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
        	classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree 

def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)

    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def main():
	myDat, labels = createDataSet()
	print 'create data:'
	print myDat
	print labels

	shan = calcShannonEnt(myDat)
	print 'calc shan:'
	print shan

	myDat[0][-1] = 'maybe'
	shan1 = calcShannonEnt(myDat)
	print 'change data and calc shan1:'
	print shan1

	subDate = splitDataSet(myDat, 0, 1)
	print 'the splitDataSet'
	print subDate

	axis = chooseBestFeatureToSplit(myDat)
	print 'the axis'
	print axis

	#myTree = createTree(myDat, labels)

	myTree = plotTree.retrieveTree(0)
	print 'myTree'
	print myTree

	myDat, labels = createDataSet()
	status1 = classify(myTree, labels, [1,0])
	print status1

	status2 = classify(myTree, labels, [1,1])
	print status2

if __name__ == "__main__":
	main()
