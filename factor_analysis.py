import os
import numpy
import pandas
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as pyplot

if not os.path.exists("data"):
	os.mkdir("data")

dataset = pandas.read_csv("dataset_final.csv")
countries = dataset['country']
dataset.drop(['Unnamed: 0', 'country'], axis = 1, inplace =  True)
dataset = dataset.values

machine = FactorAnalyzer(n_factors = 40, rotation =  None)
machine.fit(dataset)
ev, v = machine.get_eigenvalues()

pyplot.scatter(range(1, dataset.shape[1] + 1), ev)
pyplot.savefig("scatter_ev.png")
pyplot.close()

#from the analysis of the graph above it's possible to observe that the "kink" occurs for four factor; thus this is the optimal value for factors

machine = FactorAnalyzer(n_factors = 4, rotation = "varimax")
machine.fit(dataset)
output = machine.loadings_
#print(output.shape)
#print(output)
#print(machine.get_factor_variance())

#print("\nfactor loadings:\n")
#print(output)

dataset = dataset.values

result = numpy.dot(dataset, output)
#print(type(result))
#print(type(countries))

df = pandas.DataFrame(result, columns = ['factor 1 score', 'factor 2 score', 'factor 3 score', 'factor 4 score'])
df2 = pandas.DataFrame(countries)
df = df.join(df2)
df == 0
df = df[~(df == 0).any(axis = 1)]

print(df)

df.to_csv("result.csv")

#creating datasets for the 6 combinations of one-on-one factors
df_1_2 = df.drop(columns = ['factor 3 score', 'factor 4 score'])
df_1_3 = df.drop(columns = ['factor 2 score', 'factor 4 score'])
df_1_4 = df.drop(columns = ['factor 2 score', 'factor 3 score'])
df_2_3 = df.drop(columns = ['factor 1 score', 'factor 4 score'])
df_2_4 = df.drop(columns = ['factor 1 score', 'factor 3 score'])
df_3_4 = df.drop(columns = ['factor 1 score', 'factor 2 score'])

df_1_2.to_csv("data/result_factors_1_2.csv")
df_1_3.to_csv("data/result_factors_1_3.csv")
df_1_4.to_csv("data/result_factors_1_4.csv")
df_2_3.to_csv("data/result_factors_2_3.csv")
df_2_4.to_csv("data/result_factors_2_4.csv")
df_3_4.to_csv("data/result_factors_3_4.csv")