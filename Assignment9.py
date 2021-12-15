#기본적인 LDA(LinearDiscriminantAnalysis)구현
#필요한 라이브러리 import
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import pandas as pd

#데이터 파일 로드
df = pd.read_csv("student_health_3.csv", encoding = "cp949")
X = df[['수축기', '이완기', '키', '몸무게','학년']]

#데이터 정규 스케일링
df_scaled = StandardScaler().fit_transform(X)

#2개의 클래스로 구분하기 위한 LDA생성
#1~3학년을 저학년 (Class 1), 4~6학년으로 고학년(Class 2)로 분류
lda = LinearDiscriminantAnalysis(n_components=2)

#fit()호출 시 target값 입력
lda.fit(df_scaled, df.학년)
df_lda = lda.transform(df_scaled)

lda_columns=['Class1','Class2']
#dfDF_lda=pd.DataFrame(df_lda, columns=lda_columns)
#dfDF_lda['수축기']=df.수축

      
