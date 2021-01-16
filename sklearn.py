# check version of sklearn
import sklearn
print(sklearn.__version__)
# module loading
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# dataset loading
import pandas as pd
# iris dataset loading
iris = load_iris()
iris_data = iris.data
iris_label = iris.target
print('iris target값:', iris_label)
print('iris target명:',iris.target_names)
# into dataframe
iris_df = pd.DataFrame(data=iris_data,columns=iris.feature)
iris_df['label']=iris.target
iris_df.head(3)
# train data, test data
X_train, X_test, y_train, y_test = train_test_split(iris_data,iris_label,test_size=0.2,random_state=11)
# training
dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train,y_train)
# Predict using test dataset
pred = dt_clt.predict(X_test)
pred
# accuracy
from sklearn.metrics import accuracy_score
print('예측 정확도: {0:.4f'.format(accuracy_score(y_test,pred)))
# 사이킷런 내장 예제 데이터
from sklearn.datasets import load_iris
iris_data = load_iris()
print(type(iris_data))
keys = iris_data.keys()
print('붓꽃 데이터 세트의 키들:',keys)
print('\n feature_names 의 type:',type(iris_data.feature_names))
print(' feature_names 의 shape:',len(iris_data.feature_names))
print(iris_data.feature_names)

print('\n target_names 의 type:',type(iris_data.target_names))
print(' feature_names 의 shape:',len(iris_data.target_names))
print(iris_data.target_names)

print('\n data 의 type:',type(iris_data.data))
print(' data 의 shape:',iris_data.data.shape)
print(iris_data['data'])

print('\n target 의 type:',type(iris_data.target))
print(' target 의 shape:',iris_data.target.shape)
print(iris_data.target)

"""
model selection
train_test_split()
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
dt_clf = DecisionTreeClassifier()
train_data = iris.data
train_label = iris.target
dt_clf.fit(train_data,train_label)
# 학습 데이터 셋으로 예측 수행
pred = dt_clf.predict(train_data)
print('예측 정확도:',accuracy_score(train_label,pred))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris_data.data, iris_data.target,test_size=0.3,random_state=121)
dt_clf.fit(X_train,y_train)
pred = dt_clf.predict(X_test)
print('예측 정확도: {0:.4f'}.format(accuracy_score(y_test,pred)))
# 넘파이 ndarray뿐 아니라 판다스 DataFrame/Series도 가능
import pandas as pd
iris_df = pd.DataFrame(iris_data.data,columns=iris_data.feature_names)
iris_df['target'] = iris_data.target
iris_df.head()
ftr_df = iris_df.iloc[:,:-1]
tgt_df = iris_df.iloc[:,:-1]
X_train,X_test,y_train,y_test=train_test_split(ftr_df,tgt_df,test_size=0.3,random_state=121)
dt_clf = DecisionTreeClassifier( )
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))

"""
교차 검증
k-fold
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.traget
dt_clf=DecisionTreeClassifier(random_state=156)
# 5개의 폴드 세트로 분리하는 KFold 객체와 폴드 세트별 정확도를 담을 리스트 객체 생성
kfold = KFold(n_splits=5)
cv_accuracy = []
print('붓꽃 데이터 세트 크기:',features.shape[0])
n_iter = 0
# kfold객체의 split() 호출시 폴드별 학습용, 검증용 테스트 데이터 추출
for train_index, test_index in kfold.split(features):
    X_train,X_test,features[train_index],features[test_index]
    y_train,y_test=label[train_index],label[test_index]
    # 학습 데이터 에측
    dt_clf.fit(X_train,y_train)
    pred = dt_clf.predict(X_test)
    n_iter+=1
    # 반복시마다 정확도 측정
    accuracy = np.round(accuracy_score(y_test,pred),4)
    train_size = X_train.shape[0]
    test_size=X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'.format(n_iter, accuracy, train_size, test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)
# 개별 iteration별 정확도를 합하여 평균 정확도 계산
print('\n## 평균 검증 정확도:',np.mean(cv_accuracy))


# Strartied K폴드
import pandas as pd

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label']=iris.target
iris_df['label'].value_counts()

kfold = KFold(n_splits=3)
# kfold.split(X)는 폴드 세트를 3번 반복할 때마다 달라지는 학습/테스트 용 데이터 로우 인덱스 번호 반환. 
n_iter=0
for train_index,test_index in kfold.split(iris_df):
    n_iter+=1
    label_train=iris_df['label'].iloc[train_index]
    label_test=iris_df['label'].iloc[test_index]
    print('## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n', label_train.value_counts())
    print('검증 레이블 데이터 분포:\n', label_test.value_counts())

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
n_iter=0
for train_index,test_index in skf.split(iris_df,iris_df['label']):
    n_iter+=1
    label_train=iris_df['label'].iloc[train_index]
    label_test=iris_df['label'].iloc[test_index]
    print('## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n', label_train.value_counts())
    print('검증 레이블 데이터 분포:\n', label_test.value_counts())
    
