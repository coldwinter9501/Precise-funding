#stack model
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator,ClassifierMixin,TransformerMixin,clone
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

#model
#model_gnb = GaussianNB()  # 建立朴素贝叶斯分类
#model_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 4)) # 建立MLP
model_rfc = RandomForestClassifier(n_estimators=500)
#model_svc = SVC()  # 建立支持向量机模型
model_gbc = GradientBoostingClassifier(n_estimators=200)  # 建立梯度增强分类模型对象
#base_model_names = ['GaussianNB', 'MLPClassifier', 'RandomForestClassifier', 'SVC', 'GradientBoostingClassifier']  # 不同模型的名称列表
base_model = [model_rfc, model_gbc]  # 不同回归模型对象的集合

meta_model = XGBClassifier()

class StackingAverageModels(BaseEstimator,ClassifierMixin,TransformerMixin):
    def __init__(self,base_models,meta_model,n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    #将原来的模型clone出来，并fit
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=328)

        #对每个模型，使用交叉验证来训练初级学习器，得到次级训练集
        out_of_fold_pre = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                instance.fit(X[train_index], y[train_index])
                self.base_models_[i].append(instance)
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_pre[holdout_index,i] = y_pred
        #使用次级训练集来训练次级学习器
        self.meta_model_.fit(out_of_fold_pre,y)
        return self
    #再fit方法中，已经保存了初级学习器和次级学习器，需利用predict
    def predict(self,X):
        meta_featues = np.column_stack([np.column_stack([model.predict(X) for model in base_models]).mean(axis =1) for base_models in self.base_models_])
        return self.meta_model_.predict(meta_featues)


STACK = StackingAverageModels(base_model, meta_model)

result = STACK.fit(x_train.values, y_train.values).predict(test.values)
print(result)

# Save results
test_result = pd.DataFrame(columns=["studentid","subsidy"])
test_result.studentid = ids.values
test_result.subsidy = result
test_result.subsidy = test_result.subsidy.apply(lambda x:int(x))

print('1000:'+str(len(test_result[test_result.subsidy==1000])))
print('1500:'+str(len(test_result[test_result.subsidy==1500])))
print('2000:'+str(len(test_result[test_result.subsidy==2000])))

test_result.to_csv("./result.csv",index=False)
