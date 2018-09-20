#!bin/usr/python
# Shelby Luttrell

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn import tree, preprocessing

from sklearn.externals.six import StringIO
import pydot
import pprint as pp

# Read in the csv of Q3 training data
q3train = pd.read_csv('q3_train.csv', sep= ',')

# set the categorical data to binary values 0 and 1
train1 = pd.get_dummies(q3train.loc[:, 'location':'media'])
train2 = pd.get_dummies(q3train.label)

# Read in the csv of Q3 test data
data = pd.read_csv('q3_test.csv', sep=',')

# set the test categorical data to binary values
test1 = pd.get_dummies(data.loc[:, 'location':'media'])

test2 = data.label

# set criterion to gini for C4.5 and entropy for ID3
gini_entropy = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=8, min_samples_leaf=4)
#gini_entropy = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=8, min_samples_leaf=4)
gini_entropy.fit(train1, train2)

# make the decision tree
predict2 = gini_entropy.predict(test1)
le = preprocessing.LabelEncoder()
le.fit(test2)
test2 = le.transform(test2)
print(type(test2))
predict2 = predict2[:, 1]

# prints classification_report to the screen as seen in my report
print(classification_report(test2, predict2))
dot_data = StringIO()

# prints the tree to a pdf
tree.export_graphviz(gini_entropy, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("t5_q1.pdf")

# end
