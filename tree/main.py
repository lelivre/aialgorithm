import trees

def main():
	myDat, labels = trees.createDataSet()
	print 'create data:'
	print myDat

	shan = trees.calcShannonEnt(myDat)
	print 'calc shan:'
	print shan

	myDat[0][-1] = 'maybe'
	shan1 = trees.calcShannonEnt(myDat)
	print 'change data and calc shan1:'
	print shan1

if __name__ == "__main__":
	main()
