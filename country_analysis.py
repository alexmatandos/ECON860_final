import matplotlib.pyplot as pyplot
import pandas
import numpy
import os

if not os.path.exists("country_plots"):
	os.mkdir("country_plots")

#let's check the patterns for Great Britain and the US
dataset = pandas.read_csv("result.csv")
dataset['united_states'] = numpy.where(dataset['country'] == 'US', 1, 0)
dataset == 0
usa_dataset = dataset[~(dataset == 0).any(axis = 1)]
dataset.drop(['united_states'], axis = 1, inplace = True)
dataset['gb'] = numpy.where(dataset['country'] == 'GB', 1, 0)
dataset == 0
gb_dataset = dataset[~(dataset == 0).any(axis = 1)]

pyplot.scatter(usa_dataset['factor 1 score'], usa_dataset['factor 2 score'])
pyplot.savefig("country_plots/usa_factors_1_2.png")
pyplot.close()

pyplot.scatter(usa_dataset['factor 1 score'], usa_dataset['factor 3 score'])
pyplot.savefig("country_plots/usa_factors_1_3.png")
pyplot.close()

pyplot.scatter(usa_dataset['factor 1 score'], usa_dataset['factor 4 score'])
pyplot.savefig("country_plots/usa_factors_1_4.png")
pyplot.close()

pyplot.scatter(usa_dataset['factor 2 score'], usa_dataset['factor 3 score'])
pyplot.savefig("country_plots/usa_factors_2_3.png")
pyplot.close()

pyplot.scatter(usa_dataset['factor 2 score'], usa_dataset['factor 4 score'])
pyplot.savefig("country_plots/usa_factors_2_4.png")
pyplot.close()

pyplot.scatter(usa_dataset['factor 3 score'], usa_dataset['factor 4 score'])
pyplot.savefig("country_plots/usa_factors_3_4.png")
pyplot.close()

pyplot.scatter(gb_dataset['factor 1 score'], gb_dataset['factor 2 score'])
pyplot.savefig("country_plots/gb_factors_1_2.png")
pyplot.close()

pyplot.scatter(gb_dataset['factor 1 score'], gb_dataset['factor 3 score'])
pyplot.savefig("country_plots/gb_factors_1_3.png")
pyplot.close()

pyplot.scatter(gb_dataset['factor 1 score'], gb_dataset['factor 4 score'])
pyplot.savefig("country_plots/gb_factors_1_4.png")
pyplot.close()

pyplot.scatter(gb_dataset['factor 2 score'], gb_dataset['factor 3 score'])
pyplot.savefig("country_plots/gb_factors_2_3.png")
pyplot.close()

pyplot.scatter(gb_dataset['factor 2 score'], gb_dataset['factor 4 score'])
pyplot.savefig("country_plots/gb_factors_2_4.png")
pyplot.close()

pyplot.scatter(gb_dataset['factor 3 score'], gb_dataset['factor 4 score'])
pyplot.savefig("country_plots/gb_factors_3_4.png")
pyplot.close()
#pretty similar distributions of dots around the plot for both GB and the US; let's try a different country, like Kenya

dataset.drop(['gb'], axis = 1, inplace = True)
dataset['kenya'] = numpy.where(dataset['country'] == 'KE', 1, 0)
dataset == 0
kenya_dataset = dataset[~(dataset == 0).any(axis =1)]

pyplot.scatter(kenya_dataset['factor 1 score'], kenya_dataset['factor 2 score'])
pyplot.savefig("country_plots/kenya_factors_1_2.png")
pyplot.close()

pyplot.scatter(kenya_dataset['factor 1 score'], kenya_dataset['factor 3 score'])
pyplot.savefig("country_plots/kenya_factors_1_3.png")
pyplot.close()

pyplot.scatter(kenya_dataset['factor 1 score'], kenya_dataset['factor 4 score'])
pyplot.savefig("country_plots/kenya_factors_1_4.png")
pyplot.close()

pyplot.scatter(kenya_dataset['factor 2 score'], kenya_dataset['factor 3 score'])
pyplot.savefig("country_plots/kenya_factors_2_3.png")
pyplot.close()

pyplot.scatter(kenya_dataset['factor 2 score'], kenya_dataset['factor 4 score'])
pyplot.savefig("country_plots/kenya_factors_2_4.png")
pyplot.close()

pyplot.scatter(kenya_dataset['factor 3 score'], kenya_dataset['factor 4 score'])
pyplot.savefig("country_plots/kenya_factors_3_4.png")
pyplot.close()

#while sparsely populated due to lack of observations, the same pattern of the two upper bound clusters are found