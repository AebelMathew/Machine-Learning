from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
iris = load_iris()
print(iris)
data, y = iris.data, iris.target
iris_df = pd.DataFrame(data= iris.data, columns= iris.feature_names)
target_df = pd.DataFrame(data= iris.target, columns= ['species'])
data_scaled = normalize(iris_df)
data_scaled = pd.DataFrame(data_scaled, columns=iris.feature_names)
print(data_scaled)

 

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='single')  
p=cluster.fit_predict(data_scaled)
print(p)

 


#-------------SCATTER PLOT---------------

 

plt.scatter(data[p == 0, 0], data[p == 0, 1], s = 100, c = 'red', label = 'Cluster 1') # plotting cluster 0
plt.scatter(data[p == 1, 0], data[p == 1, 1], s = 100, c = 'blue', label = 'Cluster 2') # plotting cluster 1
plt.scatter(data[p == 2, 0], data[p == 2, 1], s = 100, c = 'green', label = 'Cluster 3') # plotting cluster 2

 

# plot title addition
plt.title('Clusters of iris Flower')
# labelling the x-axis
plt.xlabel('Sepal_Length')
# label of the y-axis
plt.ylabel('Sepal_Width')
# printing the legend
plt.legend()
plt.show()

 

#------------DENDROGRAM-------------

 

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Iris - HC using Complete Link")  
dend = shc.dendrogram(shc.linkage(data_scaled, method='complete'))
