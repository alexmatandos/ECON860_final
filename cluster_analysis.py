import matplotlib.pyplot as pyplot
import pandas
import os
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

if not os.path.exists("kmc_plots"):
		os.mkdir("kmc_plots")

if not os.path.exists("gmm_plots"):
	os.mkdir("gmm_plots")

dataset = pandas.read_csv("result.csv")
dataset.drop(['Unnamed: 0', 'country'], axis = 1, inplace = True)
dataset = dataset.values

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
	pyplot.scatter(dataset[:, 0], dataset[:, 1], c = results)
	pyplot.scatter(centroids[:, 0], centroids[:, 1], c = 'red', s = 200)
	pyplot.savefig("kmc_plots/clusterplot_factors_1_and_2_n=" + str(n) + "_centroid.png")
	pyplot.close()
	
	pyplot.scatter(dataset[:, 0], dataset[:, 2], c = results)
	pyplot.scatter(centroids[:, 0], centroids[:, 2], c = 'red', s = 200)
	pyplot.savefig("kmc_plots/clusterplot_factors_1_and_3_n=" + str(n) + "_centroid.png")
	pyplot.close()

	pyplot.scatter(dataset[:, 0], dataset[:, 3], c = results)
	pyplot.scatter(centroids[:, 0], centroids[:, 3], c = 'red', s = 200)
	pyplot.savefig("kmc_plots/clusterplot_factors_1_and_3_n=" + str(n) + "_centroid.png")
	pyplot.close()

	pyplot.scatter(dataset[:, 1], dataset[:, 2], c = results)
	pyplot.scatter(centroids[:, 1], centroids[:, 2], c = 'red', s = 200)
	pyplot.savefig("kmc_plots/clusterplot_factors_2_and_3_n=" + str(n) + "_centroid.png")
	pyplot.close()

	pyplot.scatter(dataset[:, 1], dataset[:, 3], c = results)
	pyplot.scatter(centroids[:, 1], centroids[:, 3], c = 'red', s = 200)
	pyplot.savefig("kmc_plots/clusterplot_factors_2_and_4_n=" + str(n) + "_centroid.png")
	pyplot.close()

	pyplot.scatter(dataset[:, 2], dataset[:, 3], c = results)
	pyplot.scatter(centroids[:, 2], centroids[:, 3], c = 'red', s = 200)
	pyplot.savefig("kmc_plots/clusterplot_factors_3_and_4_n=" + str(n) + "_centroid.png")
	pyplot.close()
	
	return ssd, silhouette

result = [run_kmeans(i + 1, dataset) for i in range(6)]

def run_gmm(n, dataset):
	gmm_machine = GaussianMixture(n_components = n)
	gmm = gmm_machine.fit_predict(dataset)
	silhouette = 0
	if n>1:
		silhouette = silhouette_score(dataset, gmm, metric = 'euclidean')
	#print(gmm)
	
	pyplot.scatter(dataset[:,0], dataset[:,1], c = gmm)
	pyplot.savefig("gmm_plots/scatter_gmm_factors_1_and_2_n=" + str(n) + ".png")
	pyplot.close()
	
	pyplot.scatter(dataset[:,0], dataset[:,2], c = gmm)
	pyplot.savefig("gmm_plots/scatter_gmm_factors_1_and_3_n=" + str(n) + ".png")
	pyplot.close()

	pyplot.scatter(dataset[:,0], dataset[:,3], c = gmm)
	pyplot.savefig("gmm_plots/scatter_gmm_factors_1_and_4_n=" + str(n) + ".png")
	pyplot.close()

	pyplot.scatter(dataset[:,1], dataset[:,2], c = gmm)
	pyplot.savefig("gmm_plots/scatter_gmm_factors_2_and_3_n=" + str(n) + ".png")
	pyplot.close()

	pyplot.scatter(dataset[:,1], dataset[:,3], c = gmm)
	pyplot.savefig("gmm_plots/scatter_gmm_factors_2_and_4_n=" + str(n) + ".png")
	pyplot.close()

	pyplot.scatter(dataset[:,2], dataset[:,3], c = gmm)
	pyplot.savefig("gmm_plots/scatter_gmm_factors_3_and_4_n=" + str(n) + ".png")
	pyplot.close()
	
	return silhouette

result_2 = [run_gmm(i + 1, dataset) for i in range(6)]