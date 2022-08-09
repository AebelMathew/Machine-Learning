import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix,accuracy_score, recall_score,precision_score, mean_squared_error


dataset=pd.read_csv("D:\AEBEL MATHEW\STUDIES\SEMS\WINTER SEM 2021-22\C1-CSE4020-ML\LAB\DATASETS\car_evaluation.csv")

dataset.columns=['buying','maint','doors','persons','lugboot','safety','classes']
dataset.buying.replace(('vhigh','high','med','low'),(1,2,3,4), inplace=True)
dataset.maint.replace(('vhigh','high','med','low'),(1,2,3,4), inplace=True)
dataset.doors.replace(('2','3','4','5more'),(1,2,3,4), inplace=True)
dataset.persons.replace(('2','4','more'),(1,2,3), inplace=True)
dataset.lugboot.replace(('small','med','big'),(1,2,3), inplace=True)
dataset.safety.replace(('low','med','high'),(1,2,3), inplace=True)
dataset.classes.replace(('unacc','acc','good','vgood'),(1,2,3,4), inplace=True)


X=dataset[['buying', 'maint','doors','persons','lugboot','safety']]
y=dataset['classes']
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
clf=MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(4,2),random_state=1)
clf.fit(X,y)
p=clf.predict(X_test)
print(y_test,p)
print(accuracy_score(y_test,p))
print(mean_squared_error(y_test,p))
print(precision_score(y_test,p))
print(recall_score(y_test,p))