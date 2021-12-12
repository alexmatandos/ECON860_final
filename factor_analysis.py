import numpy
import pandas
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as pyplot

dataset = pandas.read_csv("dataset_final.csv")
countries = dataset['country']
dataset.drop(['Unnamed: 0', 'country'], axis = 1, inplace =  True)


machine = FactorAnalyzer(n_factors = 40, rotation =  None)
machine.fit(dataset)
ev, v = machine.get_eigenvalues()

pyplot.scatter(range(1, dataset.shape[1] + 1), ev)
pyplot.savefig("scatter_ev.png")
pyplot.close()

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
#print(df)

df.to_csv("result.csv")