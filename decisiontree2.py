import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


dataset=pd.read_table('datasets/fruit_data_with_colors.txt')
data=['mass','width','height','color_score']
target=['fruit_label']
X, y = dataset[data],dataset[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
clf = DecisionTreeClassifier(criterion='gini')
clf = clf.fit(X_train, y_train)
p=clf.predict(X_test)
c=multilabel_confusion_matrix(y_test,p)
print(c)
print('Accuracy: ',accuracy_score(y_test,p))
tree.plot_tree(clf)


train