import trees
import plotTree

def main():
	fr = open('lenses.txt')
	lenses = [inst.strip().split('\t') for inst in fr.readlines()]
	lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
	lensesTree = trees.createTree(lenses,lensesLabels)

	print lensesTree

	plotTree.createPlot(lensesTree)


if __name__ == "__main__":
	main()