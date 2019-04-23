import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

data = pd.read_csv('train_file.csv')
data = data[['ID', 'Title', 'Creator', 'Subjects', 'Publisher', 'MaterialType']]
df = data
mattype = {'BOOK':0, 'SOUNDDISC':1, 'SOUNDCASS':2, 'VIDEOCASS':3, 'MUSIC':4, 'MIXED':5, 'CR':6, 'VIDEODISC':7}
df["MaterialNo"]=df.MaterialType.apply(lambda x: mattype[x])
dt = pd.read_csv("test_file.csv")
dr = pd.read_csv("results_file.csv")
dt["MaterialType"]= dr.MaterialType
dt["MaterialNo"]=dt.MaterialType.apply(lambda x: mattype[x])

dfList=[]
dfList.append(df)
dfList.append(dt)
df = pd.concat(dfList,axis=0,sort=True)
df.loc[df.Subjects.isnull(), 'Subjects'] = ''

Xfeatures = df["Subjects"]
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)
y =df.MaterialNo
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.4, random_state=42, shuffle=False, stratify=None)
matno = { 0:'BOOK', 1:'SOUNDDISC', 2:'SOUNDCASS', 3:'VIDEOCASS', 4:'MUSIC', 5:'MIXED', 6:'CR', 7:'VIDEOCASS'}


model = SVC(C=1000, gamma='auto')
model.fit(X_train, y_train)
y_tes= model.predict(X_test)
y_tes= [matno[item] for item in y_tes]
dict={'ID':dr.ID, 'MaterialType':y_tes}
s1 = pd.DataFrame(dict)
s1.to_csv('file1.csv', index=False, header = True)
