from sklearn.tree import DecisionTreeClassifier, plot_tree#tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt #tree
import seaborn as sns #graph
import pandas as pd
import numpy as np

File_path = 'D:/data/'
File_name = 'animals.xlsx'

df = pd.read_excel(File_path + File_name)

#set data 1
df.drop(columns=['Name'],inplace=True)

encoders = []

for i in range(0,len(df.columns)-1):
    enc = LabelEncoder()
    df.iloc[:,i] = enc.fit_transform(df.iloc[:,i])
    encoders.append(enc)
    
x = df.iloc[:,0:5]
y = df.iloc[:,5]

#train model
model = DecisionTreeClassifier(criterion='entropy')
model.fit(x,y)

#set data 2
x_pred = ['Warm','No','Yes','Sometimes','No']

for i in range(0,len(df.columns)-1):
    x_pred[i] = encoders[i].transform([x_pred[i]])
    
x_pred_adj = np.array(x_pred).reshape(-1, 5) # เปลี่ยนแนวตั้งเป็นนอน

#predict
y_pred = model.predict(x_pred_adj)
print(f'Prediction: {y_pred[0]}')

#validate
score = model.score(x, y)
print(f'Accuracy: {score:.2f}')

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
feature_names = ['Blood_Temperature','Give_Birth','Can_Fly','Live_In_Water','Have_Legs']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(x = feature_importances, y = feature_names)
print(feature_importances)
