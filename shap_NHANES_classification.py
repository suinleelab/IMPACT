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
parser.add_argument('--time', dest="time", help="0-no collect time, 1-have collect time")
parser.add_argument('--feature', dest="feature", help="feature type, choose from Questionnaire, Demographics, Examination, Laboratory, Dietary", default='')
parser.add_argument('--onehot', dest="onehot", help="whether using onehot encoding for nominal variables", default='1')
parser.add_argument('--target_encoding', dest="target_encoding", help="whether using target encoding for nominal variables", default='0')
parser.add_argument('--age', dest="age", help="age range", default='')
parser.add_argument('--valid', dest="valid", help="whether use citizen of US as validation set or not", default='0')
parser.add_argument('--female', dest="female", help="only explain female samples", default='0')
parser.add_argument('--male', dest="male", help="only explain male samples", default='0')

args = parser.parse_args()
year_num = int(args.year)
if args.time != '1':
    if args.valid == '1':
        path = '../result/result_without_collection_time_noDietary_validation/XGB_imputed_classification_'+str(year_num)+'_year_feature_selection'
    else:
        path = '../result/result_without_collection_time_noDietary/XGB_imputed_classification_'+str(year_num)+'_year_feature_selection'
else:
    path = '../result/result_with_collection_time_noDietary/XGB_imputed_classification_'+str(year_num)+'_year_feature_selection_'
if args.feature!='':
    path += '_'+args.feature
if args.onehot == '1':
    path += '_onehot'
if args.target_encoding == '1':
    path += '_target_encoding'
if args.age != '':
    path += '_age_'+args.age
path += '_nomercury_10000/'
# print(path)

if not os.path.isdir(path):
    os.mkdir(path)
C_file = open(path+'score.txt', 'w')
# C_file = open(path+'temp.txt', 'w')

X = pd.read_csv('/projects/leelab2/wqiu/NHANES/data/data_460_classification_imputed_missforest_feature_selection.csv')

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
X = X.drop(['Questionnaire_AlcoholFreqYrToDate', 'Dietary_DietaryWeight', 'Laboratory_MercuryInorganic'], axis=1)
# X = X.drop(['Examination_ArmCircum', 'Examination_Height', 'Questionnaire_SelfReportedHeight', 'Questionnaire_SelfReportedWeight', 'Questionnaire_SelfReportedWeight1YrAgo',
#             'Questionnaire_SelfReportedGreatestWeight', 'Examination_Weight'], axis=1)
# X = X.drop(['Laboratory_BloodCadmium', 'Questionnaire_100Cigarettes'], axis=1)
# print('removed cotinine related')

if args.time != '1':
    if 'Demographics_ReleaseCycle' in X.columns:
        X = X.drop(['Demographics_ReleaseCycle'], axis=1)

# filter for dietary feature
feature_list = list(filter(lambda text: 'Dietary_' in text, X.columns))
X = X.drop(feature_list, axis=1)
    
if args.feature != '':
    feature_list = []
    feature = args.feature.split('_')
    for i in feature:
        feature_list += list(filter(lambda text: i+'_' in text, X.columns))
    X = X[feature_list]

print(X.columns)
if args.valid == '1':
    citizen = X['Demographics_Citizenship']
    X = X.drop(['Demographics_Citizenship'], axis=1)
    
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
if args.onehot == '1':
    fea_list = pd.read_csv('NHANES_feature_list.csv')
    nominal_fea = fea_list[fea_list['Nominal']==1]['Type_Short_Name'].tolist()
    nominal_fea = list(set(nominal_fea) & set(X.columns))
    X = pd.get_dummies(X, columns=nominal_fea, drop_first=True)
    print(X.columns)
    print('After encoding', X.shape)
    print('# features after encoding: ', X.shape[1], file=C_file)

if args.valid == '1':
    valid_X = X[citizen==2]
    valid_y = y[citizen==2]
    X = X[citizen==1]
    y = y[citizen==1]
    print('# valid samples: ', valid_X.shape[0])
    print('# valid samples: ', valid_X.shape[0], file=C_file)
    print('# valid positive samples: ', sum(valid_y==1), file=C_file)
    print('# valid negative samples: ', sum(valid_y==0), file=C_file)
    print('# learning samples: ', X.shape[0])
    print('# learning samples: ', X.shape[0], file=C_file)
    print('# learning positive samples: ', sum(y==1), file=C_file)
    print('# learning negative samples: ', sum(y==0), file=C_file)
# create a train/val/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

if args.target_encoding == '1':
    fea_list = pd.read_csv('NHANES_feature_list.csv')
    nominal_fea = fea_list[fea_list['Nominal']==1]['Type_Short_Name'].tolist()
    nominal_fea = list(set(nominal_fea) & set(X.columns))
    X_train['target'] = y_train
    for fea in nominal_fea:
        mean_encode = X_train.groupby(fea)['target'].mean()
        X_train.loc[:, fea] = X_train[fea].map(mean_encode)
        X_test.loc[:, fea] = X_test[fea].map(mean_encode)
    X_train = X_train.drop(['target'], axis=1)
    print('After encoding', X.shape)
    print('# features after encoding: ', X.shape[1], file=C_file)
    
xgb_train = xgboost.DMatrix(X_train, label=y_train)
# xgb_val = xgboost.DMatrix(X_val, label=y_val)
xgb_test = xgboost.DMatrix(X_test, label=y_test)
y_train = np.array(y_train); y_test = np.array(y_test)

print('start training XGBoost')
###################
### Train Model ###
###################
# other_params = {'learning_rate': 0.01, 'n_estimators': 1000, 'objective':'binary:logistic', 'random_state': 7}
other_params = {'learning_rate': 0.002, 'n_estimators': 10000, 'objective':'binary:logistic', 'gamma': 0, 'max_depth': 3, 'min_child_weight': 1,'colsample_bytree': 1, 'colsample_bylevel': 1, 'subsample': 0.5, 'reg_lambda': 1, 'reg_alpha': 0, 'random_state': 7}
C_file.write('parameters: '+str(other_params)+'\n')
# xlf = xgboost.XGBClassifier(**other_params)
# xlf.fit(X_train, y_train)
# model_train = xlf
# pickle.dump(model_train, open(path+"model.pickle.dat", "wb"))
model_train = pickle.load(open(path+"model.pickle.dat", "rb"))
X_train = X_train[model_train.get_booster().feature_names]
X_test = X_test[model_train.get_booster().feature_names]
X_train.to_csv(path+'X_train.csv', index=False)
X_test.to_csv(path+'X_test.csv', index=False)
pd.DataFrame(y_train).to_csv(path+'y_train.csv', index=False)
pd.DataFrame(y_test).to_csv(path+'y_test.csv', index=False)
exit()

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

if args.valid == '1':
    auc_valid = roc_auc_score(valid_y, model_train.predict_proba(valid_X)[:, 1])
    AP_valid = average_precision_score(valid_y, model_train.predict_proba(valid_X)[:, 1])
    print("validation ROC_AUC Score: {}".format(auc_valid))
    print("validation AP value: {}".format(AP_valid))

    C_file.write('validation ROC_AUC Score: '+str(auc_valid)+'\n')
    C_file.write('validation AP value: '+str(AP_valid)+'\n')
    
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

if args.female == '1':
    columns = pd.read_csv(path+'/fore_data.csv').columns
    back_data = back_data[columns]
    fore_data = fore_data[columns]
    back_data = back_data[back_data['Demographics_Gender_2.0']==1]
    fore_data = fore_data[fore_data['Demographics_Gender_2.0']==1]
    path += 'female/'
    if not os.path.isdir(path):
        os.mkdir(path)
elif args.male == '1':
    columns = pd.read_csv(path+'/fore_data.csv').columns
    back_data = back_data[columns]
    fore_data = fore_data[columns]
    back_data = back_data[back_data['Demographics_Gender_2.0']==0]
    fore_data = fore_data[fore_data['Demographics_Gender_2.0']==0]
    path += 'male/'
    if not os.path.isdir(path):
        os.mkdir(path)
    
fore_data.to_csv(path+'fore_data.csv', index = False)
fore_data_label.to_csv(path+'fore_data_label.csv', index = False)

# for i in range(0, len(back_data), 500):
#     shap_explainer = shap.TreeExplainer(model_train, data=back_data.iloc[i:i+500])
#     shap_values = shap_explainer.shap_values(fore_data, check_additivity=False)
#     if i == 0:
#         shap_values_all = np.reshape(shap_values, [1]+list(shap_values.shape))
#     else:
#         shap_values = np.reshape(shap_values, [1]+list(shap_values.shape))
#         shap_values_all = np.concatenate((shap_values_all, shap_values))
# np.save(path+'shap_values_all.npy', shap_values_all)
# shap_values = shap_values_all.mean(0)

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

######################
### Evaluate Model (Probability) ###
######################
# print('start training TreeExplainer (Probability)')
# explainer = shap.TreeExplainer(model_train, data=back_data, model_output="probability")
# shap_values = explainer.shap_values(fore_data, check_additivity=False)
# np.save(path+'shap_values_probability.npy', shap_values)

# # ### Plotting ###
# shap.summary_plot(shap_values, fore_data, show=False)
# pl.savefig(path+'summary_plot_probability.png', format='png', bbox_inches='tight')
# pl.close()

if args.valid == '1':
#     fore_data_valid = valid_X
#     fore_data_label_valid = pd.DataFrame(valid_y)
    if len(valid_X)>=5000:
        fore_data_valid = valid_X.sample(n=5000, random_state=528)
        fore_data_label_valid = pd.DataFrame(valid_y).sample(n=5000, random_state=528)
    else:
        fore_data_valid = valid_X
        fore_data_label_valid = pd.DataFrame(valid_y)
    fore_data_valid.to_csv(path+'fore_data_valid.csv', index = False)
    fore_data_label_valid.to_csv(path+'fore_data_label_valid.csv', index = False)
    shap_values_valid = explainer.shap_values(fore_data_valid, check_additivity=False)
    np.save(path+'shap_values_valid.npy', shap_values_valid)
    print('start calculating SHAP interaction values')
    # explainer = shap.TreeExplainer(model_train, data=back_data, feature_perturbation='tree_path_dependent')
    shap_inter_values_valid = shap.TreeExplainer(model_train, data=back_data, feature_perturbation='tree_path_dependent').shap_interaction_values(fore_data_valid)
    np.save(path+'shap_interaction_values_valid.npy', shap_inter_values_valid)
    # ### Plotting ###
    shap.summary_plot(shap_values_valid, fore_data_valid, show=False)
    pl.savefig(path+'summary_plot_valid.png', format='png', bbox_inches='tight')
    pl.close()

    X_display_valid = fore_data_valid.copy()
    # we pass "Age" instead of an index because dependence_plot() will find it in X's column names for us
    # Sex was automatically chosen for coloring based on a potential interaction
    # plot_feature = ["Demographics_SDDSRVYR", "Demographics_RIDAGEYR", "Demographics_RIAGENDR", "Examination_BMXBMI", "Demographics_RIDRETH1", "Demographics_INDFMPIR"]
    plot_feature_valid = valid_X.columns[np.argsort(-np.sum(np.abs(shap_values_valid), axis=0))][0:5]
    for f in plot_feature_valid:
        if f not in valid_X.columns:
            continue
        shap.dependence_plot(f, shap_values_valid, fore_data_valid, display_features=X_display_valid, show=False)
        pl.savefig(path+f+'_valid.png', format='png', bbox_inches='tight')
        pl.close()
