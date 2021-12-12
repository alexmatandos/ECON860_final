import matplotlib.pyplot as pyplot
import pandas
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

dataset = pandas.read_csv("result.csv")

#visualizing for cluster patters in the scatter plots for all the six combinations between each of the four factors:

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

#let's compare KMeans with Gaussian Mixture and see the results

def run_kmeans(n, dataset):
	machine = KMeans(n_clusters = n)
	machine.fit(dataset)
	results = machine.predict(dataset)
	centroids = machine.cluster_centers_
	ssd = machine.inertia_
	#silhouette is set to zero since it does not make sense to calculate it when there's just one cluster (n=1)
	silhouette = 0
	if n > 1:
		silhouette = silhouette_score(dataset, machine.labels_, metric = 'euclidean')
	pyplot.scatter(dataset[:,0], dataset[:,1], c = results)
	pyplot.scatter(centroids[:,0], centroids[:,1], c = 'red', s = 200)
	pyplot.savefig("clusterplot_" + str(n) + "centroid")
	pyplot.close()
	return ssd, silhouette