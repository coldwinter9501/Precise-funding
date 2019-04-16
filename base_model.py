
import numpy as np
import pandas as pd
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

#data
train = pd.read_csv('./train.csv', header=0, encoding='gbk')
test = pd.read_csv('./test.csv', header=0, encoding='gbk')
#print(test.head())
y_train = train.pop('money')
x_train = train.drop('id', axis=1)
ids = test.pop('id')
test = test.drop('money',axis=1)
print(x_train.shape)
print(test.shape)

model_rfc = RandomForestClassifier(n_estimators=500)
model_gbc = GradientBoostingClassifier(n_estimators=200)
meta_model = XGBClassifier()
models = [model_rfc, model_gbc, meta_model]  # 不同模型对象的集合

result = pd.DataFrame(columns=['id','model1','model2','model3'])
result['id'] = ids
for i,model in enumerate(models):
    model.fit(x_train, y_train)
    s = 'model' + str(i+1)
    result[s] = model.predict(test)

# Save results
print('RF model')
print('1000:'+str(len(result[result.model1==1000])))
print('1500:'+str(len(result[result.model1==1500])))
print('2000:'+str(len(result[result.model1==2000])))
print('GBDT model')
print('1000:'+str(len(result[result.model2==1000])))
print('1500:'+str(len(result[result.model2==1500])))
print('2000:'+str(len(result[result.model2==2000])))
print('XGBOOST model')
print('1000:'+str(len(result[result.model3==1000])))
print('1500:'+str(len(result[result.model3==1500])))
print('2000:'+str(len(result[result.model3==2000])))

result.to_csv("./result1.csv",index=False)
