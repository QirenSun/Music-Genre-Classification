import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import svm
from sklearn import tree
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import StandardScaler


data= pd.read_csv('data.csv')
data.head()


# Dropping unneccesary columns
data = data.drop(['filename'],axis=1)

genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2)

clf1 = MLPClassifier(solver='lbfgs', 
                    alpha=1e-5,
                    hidden_layer_sizes=(95,50), 
                    random_state=0,
                    batch_size=128,
                    shuffle=True)
clf1.fit(X_train,y_train)
accuracy1=clf1.score(X_test,y_test)
print('MLP lbfgs Accuracy: ',accuracy1)

clf2 = MLPClassifier(solver='adam', 
                    alpha=1e-5,
                    hidden_layer_sizes=(150,20), 
                    random_state=1,
                    batch_size=128,
                    shuffle=True)
clf2.fit(X_train,y_train)
accuracy2=clf2.score(X_test,y_test)
print('MLP adam Accuracy2: ',accuracy2)



svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train,y_train)
#new_x = transform(np.asmatrix([6, 160]))
#predicted = svm_classifier.predict(new_x)
accuracy = svm_classifier.score(X_test, y_test)
print('SVM accuracy: ',accuracy)


clf3 = tree.DecisionTreeClassifier(criterion = 'entropy')
clf3.fit(X_train,y_train)
accuracy = clf3.score(X_test, y_test)
#clf = clf.fit(X_test,Y_test)
#accuracy = clf.score(X_train, Y_train)
print('Descision Tree accuracy: ',accuracy)


















