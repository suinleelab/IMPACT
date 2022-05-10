import xgboost
import shap
import matplotlib.pylab as pl
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse
from feature_selector import FeatureSelector
from sklearn.model_selection import GridSearchCV
import pickle

def label(permth, mortstat, month):
    if permth > month:
        return 0
    else:
        if mortstat == 1:
            return 1
        else:
            return 2

parser = argparse.ArgumentParser()
parser.add_argument('--year', dest="year", help="the year to create label")
parser.add_argument('--age', dest="age", help="age range", default='')

args = parser.parse_args()
year_num = int(args.year)
path = './model/NHANES'+str(year_num)+'_year'

if args.age != '':
    path += '_age_'+args.age
path += '/'

if not os.path.isdir(path):
    os.mkdir(path)
C_file = open(path+'score.txt', 'w')

X = pd.read_csv('./data/NHANES/NHANES.csv')

if str(year_num)+'_year_label' not in X.columns:
    X[str(year_num)+'_year_label'] = X.apply(lambda x: label(x['permth_int'], x['mortstat'], 12*int(year_num)), axis=1)
    
X = X[X[str(year_num)+'_year_label']!=2]
y = X[str(year_num)+'_year_label']

if int(year_num) not in [1,2,3,4,5]:
    X = X.drop([str(year_num)+'_year_label'], axis=1)

if args.age != '':
    age_range = args.age.split('_')
    print(age_range)
    y = y[(X['Demographics_Age']>=int(age_range[0])) & (X['Demographics_Age']<int(age_range[1]))]
    X = X[(X['Demographics_Age']>=int(age_range[0])) & (X['Demographics_Age']<int(age_range[1]))]
    
drop_list = ["mortstat", "permth_int", '1_year_label', '2_year_label', '3_year_label', '4_year_label', '5_year_label']
X = X.drop(drop_list, axis=1)

X = X.drop(['Demographics_ReleaseCycle'], axis=1)


fea_list = pd.read_csv('./data/NHANES/NHANES_feature_list.csv')
nominal_fea = fea_list[fea_list['Nominal']==1]['Type_Short_Name'].tolist()
nominal_fea = list(set(nominal_fea) & set(X.columns))
X = pd.get_dummies(X, columns=nominal_fea, drop_first=True)
print(X.columns)

print(X.columns)
print(X.shape)
print('# samples: ', X.shape[0])
print('# positive samples: ', sum(y==1))
print('# negative samples: ', sum(y==0))
print('# features: ', X.shape[1])
print('# samples: ', X.shape[0], file=C_file)
print('# positive samples: ', sum(y==1), file=C_file)
print('# negative samples: ', sum(y==0), file=C_file)
print('# features: ', X.shape[1], file=C_file)

# create a train/val/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

print('start training XGBoost')
###################
### Train Model ###
###################
other_params = {'learning_rate': 0.002, 'n_estimators': 10000, 'objective':'binary:logistic', 'gamma': 0, 'max_depth': 3, 'min_child_weight': 1,'colsample_bytree': 1, 'colsample_bylevel': 1, 'subsample': 0.5, 'reg_lambda': 1, 'reg_alpha': 0, 'random_state': 7}
C_file.write('parameters: '+str(other_params)+'\n')
xlf = xgboost.XGBClassifier(**other_params)
xlf.fit(X_train, y_train)
model_train = xlf
pickle.dump(model_train, open(path+"model.pickle.dat", "wb"))

######################
### Evaluate Model ###
######################
# see how well we can order people by survival
auc = roc_auc_score(y_test, model_train.predict_proba(X_test)[:, 1])
AP = average_precision_score(y_test, model_train.predict_proba(X_test)[:, 1])
print("ROC_AUC Score: {}".format(auc))
print("AP value: {}".format(AP))

C_file.write('ROC_AUC Score: '+str(auc)+'\n')
C_file.write('AP value: '+str(AP)+'\n')

C_file.close()

print('start training TreeExplainer')
if len(X_train)>=10000:
    back_data = X_train.sample(n=10000, random_state=428)
else:
    back_data = X_train
if len(X_test)>=5000:
    fore_data = X_test.sample(n=5000, random_state=528)
    fore_data_label = pd.DataFrame(y_test).sample(n=5000, random_state=528)
else:
    fore_data = X_test
    fore_data_label = pd.DataFrame(y_test)

    
fore_data.to_csv(path+'fore_data.csv', index = False)
fore_data_label.to_csv(path+'fore_data_label.csv', index = False)

explainer = shap.TreeExplainer(model_train, data=back_data)
shap_values = explainer.shap_values(fore_data, check_additivity=False)
np.save(path+'shap_values.npy', shap_values)

# ### Plotting ###
shap.summary_plot(shap_values, fore_data, show=False)
pl.savefig(path+'summary_plot.png', format='png', bbox_inches='tight')
pl.close()

print('start calculating SHAP interaction values')
# explainer = shap.TreeExplainer(model_train, data=back_data, feature_perturbation='tree_path_dependent')
shap_inter_values = shap.TreeExplainer(model_train, data=back_data, feature_perturbation='tree_path_dependent').shap_interaction_values(fore_data)
np.save(path+'shap_interaction_values.npy', shap_inter_values)