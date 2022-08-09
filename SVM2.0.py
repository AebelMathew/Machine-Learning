from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

 

# Loading some example data
iris = datasets.load_wine()
X = iris.data[:, [0, 12]]
#print(X)
y = iris.target
#print("target", y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
weights=[]

 

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(gamma='scale', kernel='rbf', probability=True)

 

clf1 = clf1.fit(X_train, y_train)
p1=clf1.predict(X_test)
e1=accuracy_score(y_test, p1)
print("accuracy of Decision tree: ",e1)

 


clf2 = clf2.fit(X_train, y_train)
p2=clf2.predict(X_test)
e2=accuracy_score(y_test, p2)
print("\naccuracy of KNN: ",e2)

 

clf3 = clf3.fit(X_train, y_train)
p3=clf3.predict(X_test)
e3=accuracy_score(y_test, p3)
print("\naccuracy of SVM: ",e3)

 


weights.append(((e1/e1)*(e1/e1))+((e1/e2)*(e1/e2))+((e1/e3)*(e1/e3)))
weights.append(((e2/e1)*(e2/e1))+((e2/e2)*(e2/e2))+((e2/e3)*(e2/e3)))
weights.append(((e3/e1)*(e3/e1))+((e3/e2)*(e3/e2))+((e3/e3)*(e3/e3)))
print(weights)

 

eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[1,0.5,0.8],flatten_transform=True)

 

eclf = eclf.fit(X_train, y_train)
p4=eclf.predict(X_test)
print("p4",p4)
print("actual :",y_test)
print("accuracy of Ensemble Score :",accuracy_score(y_test, p4))
#average_precision = average_precision_score(y_test, p4)

 

#print('Average precision-recall score: {0:0.2f}'.format(average_precision))

 

# Plotting decision regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

 

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

 

for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [clf1, clf2, clf3, eclf],
                        ['Decision Tree (depth=4)', 'KNN (k=7)',
                         'Kernel SVM', 'Soft Voting']):

 

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

 

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

 

plt.show()
