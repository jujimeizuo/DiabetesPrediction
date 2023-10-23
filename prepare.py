import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
train = pd.read_csv('./dataset/train.csv')
test = pd.read_csv('./dataset/test_withoutLable.csv')

# 输出训练集和测试集的大小
print("train.shape:", train.shape, "test.shape:", test.shape)

print("训练集的摘要信息")
train.info()
print("数据集的摘要信息")
test.info()


print(train.groupby(['HighBP'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='HighBP', hue='Diabetes_binary', data=train)
plt.title('HighBP & Diabetes_binary')

print(train.groupby(['HighChol'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='HighChol', hue='Diabetes_binary', data=train)
plt.title('HighChol & Diabetes_binary')

print(train.groupby(['CholCheck'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='CholCheck', hue='Diabetes_binary', data=train)
plt.title('CholCheck & Diabetes_binary')

print(train.groupby(['BMI'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='BMI', hue='Diabetes_binary', data=train)
plt.title('BMI & Diabetes_binary')

print(train.groupby(['Smoker'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='Smoker', hue='Diabetes_binary', data=train)
plt.title('Smoker & Diabetes_binary')

print(train.groupby(['Stroke'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='Stroke', hue='Diabetes_binary', data=train)
plt.title('Stroke & Diabetes_binary')

print(train.groupby(['HeartDiseaseorAttack'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='HeartDiseaseorAttack', hue='Diabetes_binary', data=train)
plt.title('HeartDiseaseorAttack & Diabetes_binary')

print(train.groupby(['PhysActivity'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='PhysActivity', hue='Diabetes_binary', data=train)
plt.title('PhysActivity & Diabetes_binary')

print(train.groupby(['Fruits'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='Fruits', hue='Diabetes_binary', data=train)
plt.title('Fruits & Diabetes_binary')

print(train.groupby(['Veggies'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='Veggies', hue='Diabetes_binary', data=train)
plt.title('Veggies & Diabetes_binary')

print(train.groupby(['HvyAlcoholConsump'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='HvyAlcoholConsump', hue='Diabetes_binary', data=train)
plt.title('HvyAlcoholConsump & Diabetes_binary')

print(train.groupby(['AnyHealthcare'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='AnyHealthcare', hue='Diabetes_binary', data=train)
plt.title('AnyHealthcare & Diabetes_binary')

print(train.groupby(['NoDocbcCost'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='NoDocbcCost', hue='Diabetes_binary', data=train)
plt.title('NoDocbcCost & Diabetes_binary')

print(train.groupby(['GenHlth'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='GenHlth', hue='Diabetes_binary', data=train)
plt.title('GenHlth & Diabetes_binary')

print(train.groupby(['MentHlth'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='MentHlth', hue='Diabetes_binary', data=train)
plt.title('MentHlth & Diabetes_binary')

print(train.groupby(['PhysHlth'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='PhysHlth', hue='Diabetes_binary', data=train)
plt.title('PhysHlth & Diabetes_binary')

print(train.groupby(['DiffWalk'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='DiffWalk', hue='Diabetes_binary', data=train)
plt.title('DiffWalk & Diabetes_binary')

print(train.groupby(['Sex'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='Sex', hue='Diabetes_binary', data=train)
plt.title('Sex & Diabetes_binary')

print(train.groupby(['Age'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='Age', hue='Diabetes_binary', data=train)
plt.title('Age & Diabetes_binary')

print(train.groupby(['Education'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='Education', hue='Diabetes_binary', data=train)
plt.title('Education & Diabetes_binary')

print(train.groupby(['Income'])['Diabetes_binary'].agg(['count', 'mean']))
plt.figure(figsize=(10, 5))
sns.countplot(x='Income', hue='Diabetes_binary', data=train)
plt.title('Income & Diabetes_binary')

plt.rcParams['axes.unicode_minus']=False

train = train.drop(['ID'], axis=1)
colormap = plt.cm.viridis
plt.figure(figsize=(22, 22))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(
    train.astype(float).corr(method='kendall'),
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=colormap,
    linecolor='white',
    annot=True
)