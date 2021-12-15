import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

df = pd.read_csv("student_health_3.csv", encoding = "cp949")

#X = df[['수축기', '이완기', '키', '몸무게']]
#y = df['학년']

X = np.array(df[['수축기', '이완기', '키', '몸무게']])
y = np.array(df['학년'])

clf = LinearDiscriminantAnalysis()
clf.fit(X,y)

#clf.predict([[수축기, 이완기,키,몸무게]])
#clf.predict([[77, 58,125,27]])
