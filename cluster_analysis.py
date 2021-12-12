import matplotlib.pyplot as pyplot
import pandas

dataset = pandas.read_csv("dataset_final.csv")

#visualizing the cluster patters for all the six combinations between each of the four factors:
pyplot.scatter(dataset['factor 1 score'], dataset['factor 2 score'])
pyplot.xlabel('factor 1 score')
pyplot.ylabel('factor 2 score')
pyplot.savefig("factors_1_and_2.png")
pyplot.close()

pyplot.scatter(dataset['factor 1 score'], dataset['factor 3 score'])
pyplot.xlabel('factor 1 score')
pyplot.ylabel('factor 3 score')
pyplot.savefig("factors_1_and_3.png")
pyplot.close()

pyplot.scatter(dataset['factor 1 score'], dataset['factor 4 score'])
pyplot.xlabel('factor 1 score')
pyplot.ylabel('factor 4 score')
pyplot.savefig("factors_1_and_4.png")
pyplot.close()

pyplot.scatter(dataset['factor 2 score'], dataset['factor 3 score'])
pyplot.xlabel('factor 2 score')
pyplot.ylabel('factor 3 score')
pyplot.savefig("factors_2_and_3.png")
pyplot.close()

pyplot.scatter(dataset['factor 2 score'], dataset['factor 4 score'])
pyplot.xlabel('factor 2 score')
pyplot.ylabel('factor 4 score')
pyplot.savefig("factors_2_and_4.png")
pyplot.close()

pyplot.scatter(dataset['factor 3 score'], dataset['factor 4 score'])
pyplot.xlabel('factor 3 score')
pyplot.ylabel('factor 4 score')
pyplot.savefig("factors_3_and_4.png")
pyplot.close()