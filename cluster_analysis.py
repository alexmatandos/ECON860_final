import matplotlib.pyplot as pyplot
import pandas
import os
import glob
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

if not os.path.exists("kmc_plots"):
		os.mkdir("kmc_plots")

if not os.path.exists("gmm_plots"):
	os.mkdir("gmm_plots")

#creating data sets for the six combinations of all six factors

#let's compare KMeans with Gaussian Mixture and see the results

def run_kmeans(n, data):
	machine = KMeans(n_clusters = n)
	machine.fit(data)
	results = machine.predict(data)
	centroids = machine.cluster_centers_
	ssd = machine.inertia_
	#silhouette is set to zero since it does not make sense to calculate it when there's just one cluster (n=1)
	silhouette = 0
	if n > 1:
		silhouette = silhouette_score(data, machine.labels_, metric = 'euclidean')
	pyplot.scatter(data[:, 0], data[:, 1], c = results)
	pyplot.scatter(centroids[:, 0], centroids[:, 1], c = 'red', s = 200)
	pyplot.savefig("kmc_plots/clusterplot_n=" + str(n) + str(file_name.replace(".", "").replace("csv", "").replace("data", "").replace("\\", "")) + "_centroid.png")
	pyplot.close()
	
	return ssd, silhouette

def run_gmm(n, data):
	gmm_machine = GaussianMixture(n_components = n)
	gmm = gmm_machine.fit_predict(data)
	silhouette = 0
	if n>1:
		silhouette = silhouette_score(data, gmm, metric = 'euclidean')
	#print(gmm)
	pyplot.scatter(data[:,0], data[:,1], c = gmm)
	pyplot.savefig("gmm_plots/scatter_n="  + str(n) +  str(file_name.replace(".", "").replace("csv", "").replace("data", "").replace("\\", "")) + ".png")
	pyplot.close()
	
	return silhouette

for file_name in glob.glob("data/*.csv"):
	dataset = pandas.read_csv(file_name)
	dataset.drop(['Unnamed: 0', 'country'], axis = 1, inplace = True)
	dataset = dataset.values
	
	result_kmc = [run_kmeans(i + 1, dataset) for i in range(7)]
	result_gmm = [run_gmm(i + 1, dataset) for i in range(7)]
	
	silhouette_kmc = [i[1] for i in result_kmc][0:]
	silhouette_gmm = [i for i in result_gmm][0:]

	#below: the silhouette scores for KMeans and GMM (in this order) for factors 1 and 2, then factors 1 and 3, and so forth till factors 3 and 4:
	
	print(silhouette_kmc)
	print(silhouette_gmm)

	#thus, the optimal number of clusters shall be the one that maximizes the silhouette score: for all combinations, 2 clusters seems to be the optimal when using either KMeans or Gaussian Mixture

	print("\n optimal cluster KMeans " + str(file_name.replace(".", "").replace("data", "").replace("\\", "").replace("csv", "").replace("_", " ")) + ": \n", silhouette_kmc.index(max(silhouette_kmc))+1)
	print("\n optimal cluster GMM " + str(file_name.replace(".", "").replace("data", "").replace("\\", "").replace("csv", "").replace("_", " ")) + ": \n", silhouette_gmm.index(max(silhouette_gmm))+1)

	#I would argue that, while not better, the Gaussian Mixture model may be more appropriate than the KMeans method since the former is able to distinguish cluster in reasonably "linear" relationships that are overlapped by other dots due to the fact that the latter method is only concerned with the proximity to a given centroid.