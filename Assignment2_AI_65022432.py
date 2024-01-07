from sklearn.tree import DecisionTreeClassifier, plot_tree#tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt #tree
import seaborn as sns #graph
import pandas as pd
import numpy as np

file_path = 'D:/data/'
file_name = 'Iris.xlsx'

df = pd.read_excel(file_path + file_name)

df.drop(columns=['Id'],inplace=True)

x = df.iloc[:,0:4]
y = df.iloc[:,4]

#train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,random_state=0)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train,y_train)

#validate
score_train = model.score(x_train, y_train)
print(f'Accuracy_train: {score_train:.2f}')

score_test = model.score(x_test, y_test)
print(f'Accuracy_test: {score_test:.2f}')

#tree
feature = x.columns.tolist()
Data_class = y.tolist()

plt.figure(figsize=(25,20))
_ = plot_tree(model,
              feature_names = feature,
              class_names=Data_class,
              label='all',
              impurity=True,
              precision=3,
              filled=True,
              rounded=True,
              fontsize=16
              )
plt.show()

#graph
feature_importances = model.feature_importances_
feature_names = ['SepalLenghtCm','SepalWidthCm','PetalLenghtCm','PetalWidthCm']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(x = feature_importances, y = feature_names)

