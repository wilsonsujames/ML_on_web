import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Data Visualization 
import seaborn as sns  #Python library for Vidualization
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks
from joblib import dump, load


#https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python  資料集下載


dataset = pd.read_csv('./Mall_Customers.csv')
print(dataset.head(10))
print(dataset.shape)

X= dataset.iloc[:150, [3,4]].values

silhouette_avg = []



for i in range(2,11):
    try:
        kmeans = KMeans(n_clusters= i, init='k-means++').fit(X)
        silhouette_avg.append(silhouette_score(X, kmeans.labels_)) 
    except:
        pass

plt.plot(range(2,11), silhouette_avg)
plt.title('silhouette_avg')
plt.xlabel('i')
plt.ylabel('silhouette_avg')
plt.show()


indArr,peak_heightsDict=find_peaks(silhouette_avg)

print(silhouette_avg)
print(indArr)

# 在list中 index為三 最佳分群數為5
RowNumCategories=indArr[0]+2


kmeansmodel = KMeans(n_clusters= RowNumCategories, init='k-means++')
y_kmeans= kmeansmodel.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
# plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 100, c = 'pink', label = 'Cluster 6')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')


plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

dump(kmeansmodel, 'kmean.joblib') 



