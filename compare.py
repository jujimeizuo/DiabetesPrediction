import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns


data = pd.read_csv('./dataset/train.csv')
X = data.drop(['ID', 'Diabetes_binary'], axis=1)
Y = data['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# K 近邻

from sklearn.neighbors import KNeighborsClassifier

training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 20)

for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label='training accuracy')
plt.plot(neighbors_settings, test_accuracy, label='test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()
plt.savefig('knn_compare_model')

knn = KNeighborsClassifier(n_neighbors=22)
knn.fit(X_train, y_train)

print('Accuary on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuary on test set: {:.2f}'.format(knn.score(X_test, y_test)))

# 逻辑回归

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print('Accuary on training set: {:.3f}'.format(log_reg.score(X_train, y_train)))
print('Accuary on test set: {:.3f}'.format(log_reg.score(X_test, y_test)))

# 决策树

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=6, random_state=0)
tree.fit(X_train, y_train)
print('Accuary on training set: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuary on test set: {:.3f}'.format(tree.score(X_test, y_test)))

# 随机森林

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
print('Accuary on training set: {:.3f}'.format(rf.score(X_train, y_train)))
print('Accuary on test set: {:.3f}'.format(rf.score(X_test, y_test)))

# 梯度提升

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=0)
gb.fit(X_train, y_train)
print('Accuary on training set: {:.3f}'.format(gb.score(X_train, y_train)))
print('Accuary on test set: {:.3f}'.format(gb.score(X_test, y_test)))

# 支持向量机

from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)
print('Accuary on training set: {:.3f}'.format(svc.score(X_train, y_train)))
print('Accuary on test set: {:.3f}'.format(svc.score(X_test, y_test)))

# 深度学习

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)
print('Accuary on training set: {:.3f}'.format(mlp.score(X_train_scaled, y_train)))
print('Accuary on test set: {:.3f}'.format(mlp.score(X_test_scaled, y_test)))

scores = []
scores.append(knn.score(X_test, y_test))
scores.append(log_reg.score(X_test, y_test))
scores.append(tree.score(X_test, y_test))
scores.append(rf.score(X_test, y_test))
scores.append(gb.score(X_test, y_test))
scores.append(svc.score(X_test, y_test))
scores.append(mlp.score(X_test, y_test))

#汇总数据
cvResDf=pd.DataFrame({'score': scores,
                     'algorithm':['knn','log_reg','tree',
                                  'rf','gb','svc','mlp']})

cvResDf

cvResFacet=sns.FacetGrid(cvResDf.sort_values(by='score',ascending=False),sharex=False,
            sharey=False,aspect=2)
cvResFacet.map(sns.barplot,'score','algorithm',
               palette='muted')
cvResFacet.set(xlim=(0.7,0.8))
cvResFacet.add_legend()