from joblib import dump, load
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./Mall_Customers.csv')


X= dataset.iloc[151:202, [3,4]].values



kmean_clf = load('kmean.joblib') 



y_kmeans= kmean_clf.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
# plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 100, c = 'pink', label = 'Cluster 6')
plt.scatter(kmean_clf.cluster_centers_[:, 0], kmean_clf.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')


plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()















