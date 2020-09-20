import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.estimators import H2OPrincipalComponentAnalysisEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols

train_base_df = pd.read_csv("E:/Subhabrata_backup/Data_sceince/INSOFE_Lecture_Documents/PHD/CodeFrom_Avinash/code/code/data/train/Train.csv",na_values=['?'])
train_claim_df = pd.read_csv("E:/Subhabrata_backup/Data_sceince/INSOFE_Lecture_Documents/PHD/CodeFrom_Avinash/code/code/data/train/Train_Claim.csv",na_values=['?',"MISSEDDATA","MISSINGVALUE",-5])
train_demographics_df = pd.read_csv("E:/Subhabrata_backup/Data_sceince/INSOFE_Lecture_Documents/PHD/CodeFrom_Avinash/code/code/data/train/Train_Demographics.csv",na_values=['?','NA',' '])
train_policy_df = pd.read_csv("E:/Subhabrata_backup/Data_sceince/INSOFE_Lecture_Documents/PHD/CodeFrom_Avinash/code/code/data/train/Train_Policy.csv",na_values=['?',"MISSINGVAL"])
train_vehicle_df = pd.read_csv("E:/Subhabrata_backup/Data_sceince/INSOFE_Lecture_Documents/PHD/CodeFrom_Avinash/code/code/data/train/Train_Vehicle.csv",na_values=['???'])


test_base_df = pd.read_csv("E:/Subhabrata_backup/Data_sceince/INSOFE_Lecture_Documents/PHD/CodeFrom_Avinash/code/code/data/test/Test.csv",na_values=['?'])
test_claim_df = pd.read_csv("E:/Subhabrata_backup/Data_sceince/INSOFE_Lecture_Documents/PHD/CodeFrom_Avinash/code/code/data/test/Test_Claim.csv",na_values=['?',"MISSEDDATA","MISSINGVALUE",-5])
test_demographics_df = pd.read_csv("E:/Subhabrata_backup/Data_sceince/INSOFE_Lecture_Documents/PHD/CodeFrom_Avinash/code/code/data/test/Test_Demographics.csv",na_values=['?','NA',' '])
test_policy_df = pd.read_csv("E:/Subhabrata_backup/Data_sceince/INSOFE_Lecture_Documents/PHD/CodeFrom_Avinash/code/code/data/test/Test_Policy.csv",na_values=['?',"MISSINGVAL"])
test_vehicle_df = pd.read_csv("E:/Subhabrata_backup/Data_sceince/INSOFE_Lecture_Documents/PHD/CodeFrom_Avinash/code/code/data/test/Test_Vehicle.csv",na_values=['???'])

   
# Merging the Dataframes for Claim, Demographics, Policy for train",
train_temp_1 = pd.merge(train_base_df, train_claim_df, on='CustomerID', how='inner')
train_temp_2 = pd.merge(train_temp_1, train_demographics_df, on='CustomerID', how='inner')
train_temp_3 = pd.merge(train_temp_2,train_policy_df,on = 'CustomerID', how='inner')

# Merging the Dataframes for Claim, Demographics, Policy for test\n",
test_temp_1 = pd.merge(test_base_df, test_claim_df, on='CustomerID', how='inner')
test_temp_2 = pd.merge(test_temp_1, test_demographics_df, on='CustomerID', how='inner')
test_temp_3 = pd.merge(test_temp_2,test_policy_df,on = 'CustomerID', how='inner')

#Splitting the Vehicle Dataframe into Vehicle_ID, Vehicle_Make, Vehicle_Model, Vehicle_YOM for Train\n",

train_vehicle_id_df = train_vehicle_df[train_vehicle_df['VehicleAttribute'] == 'VehicleID']
train_vehicle_make_df = train_vehicle_df[train_vehicle_df['VehicleAttribute'] == 'VehicleMake']
train_vehicle_model_df = train_vehicle_df[train_vehicle_df['VehicleAttribute'] == 'VehicleModel']
train_vehicle_yom_df = train_vehicle_df[train_vehicle_df['VehicleAttribute'] == 'VehicleYOM']

# Splitting the Vehicle Dataframe into Vehicle_ID, Vehicle_Make, Vehicle_Model, Vehicle_YOM for Test\n",

test_vehicle_id_df = test_vehicle_df[test_vehicle_df['VehicleAttribute'] == 'VehicleID']
test_vehicle_make_df = test_vehicle_df[test_vehicle_df['VehicleAttribute'] == 'VehicleMake']
test_vehicle_model_df =  test_vehicle_df[test_vehicle_df['VehicleAttribute'] == 'VehicleModel']
test_vehicle_yom_df =  test_vehicle_df[test_vehicle_df['VehicleAttribute'] == 'VehicleYOM']
 
# Dropping the Vehicle Attribute for all the Vehicle Sub Dataframes\n",
train_vehicle_id_df.drop(['VehicleAttribute'],inplace=True,axis=1)
train_vehicle_make_df.drop(['VehicleAttribute'],inplace=True,axis=1)
train_vehicle_model_df.drop(['VehicleAttribute'],inplace=True,axis=1)
train_vehicle_yom_df.drop(['VehicleAttribute'],inplace=True,axis=1)
    
# Assinging the columns to the specific Sub Vehicle Dataframes\n",
train_vehicle_id_df.columns = ['CustomerID', 'VehicleID']
train_vehicle_make_df.columns = ['CustomerID', 'VehicleMake']
train_vehicle_model_df.columns = ['CustomerID', 'VehicleModel']
train_vehicle_yom_df.columns = ['CustomerID', 'VehicleYOM']
  
# Dropping the Vehicle Attribute for all the Vehicle Sub Dataframes\n",
test_vehicle_id_df.drop(['VehicleAttribute'],inplace=True,axis=1)
test_vehicle_make_df.drop(['VehicleAttribute'],inplace=True,axis=1)
test_vehicle_model_df.drop(['VehicleAttribute'],inplace=True,axis=1)
test_vehicle_yom_df.drop(['VehicleAttribute'],inplace=True,axis=1)


# Assinging the columns to the specific Sub Vehicle Dataframes\n",
test_vehicle_id_df.columns = ['CustomerID', 'VehicleID']
test_vehicle_make_df.columns = ['CustomerID', 'VehicleMake']
test_vehicle_model_df.columns = ['CustomerID', 'VehicleModel']
test_vehicle_yom_df.columns = ['CustomerID', 'VehicleYOM']

# Merging train_temp_3 with train_vehicle Subdatasets\n",
train_temp_4 = pd.merge(train_temp_3,train_vehicle_id_df,on = 'CustomerID', how='inner')
train_temp_5 = pd.merge(train_temp_4,train_vehicle_make_df,on = 'CustomerID', how='inner')
train_temp_6 = pd.merge(train_temp_5,train_vehicle_model_df,on = 'CustomerID', how='inner')
train_data = pd.merge(train_temp_6,train_vehicle_yom_df,on = 'CustomerID', how='inner')
  
# Merging test_temp_3 with test_vehicle Subdatasets\n",
test_temp_4 = pd.merge(test_temp_3,test_vehicle_id_df,on = 'CustomerID', how='inner')
test_temp_5 = pd.merge(test_temp_4,test_vehicle_make_df,on = 'CustomerID', how='inner')
test_temp_6 = pd.merge(test_temp_5,test_vehicle_model_df,on = 'CustomerID', how='inner')
test_data = pd.merge(test_temp_6,test_vehicle_yom_df,on = 'CustomerID', how='inner')

len(train_data.columns),len(test_data.columns)

# Merging all the both train and test dataframes to form one single frame"

target = 'ReportedFraud'
target_value = train_data[target]
temp_train_data = train_data.drop('ReportedFraud',axis=1)
temp_train_data['dataset'] = 'train'
temp_test_data = test_data.copy()
temp_test_data['dataset'] = 'test'
final_data = temp_train_data.append(temp_test_data)

# Function to calculate the percentage of the NA values present"
def percentage_na(df,count):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent*100], axis=1, keys=['Total', 'Percent','mode_val'])
    print(missing_data.head(count))
    percentage_na(final_data,15)

# Listing the Categorical, Numerical and Date Variables"
cat_variables = ['Policy_CombinedSingleLimit',
    'IncidentState',
    'Witnesses',
    'VehicleMake',
    'NumberOfVehicles',
    'AuthoritiesContacted',
    'IncidentCity',
    'InsuredRelationship',
    'SeverityOfIncident',
    'InsuredEducationLevel',
    'InsuredOccupation',
    'IncidentTime',
    'InsurancePolicyState',
    'TypeOfCollission',
    'TypeOfIncident',
    'InsuredGender',
    'VehicleYOM',
    'IncidentAddress',
    'BodilyInjuries',
    'InsuredHobbies',
    'InsuredZipCode',
    'Policy_Deductible',
    ]
num_variables = ['AmountOfTotalClaim',
                 'InsuredAge',
                 'CapitalGains',
                 'CapitalLoss',
                 'CustomerLoyaltyPeriod',
                 'PolicyAnnualPremium',
                 'UmbrellaLimit',
                 'AmountOfInjuryClaim',
                 'AmountOfPropertyClaim',
                 'AmountOfVehicleDamage'
                ]
date_variables = [
    'DateOfPolicyCoverage',
    'DateOfIncident',
    ]

# Converting the datatypes of category, float and Datetime\n"
for var in cat_variables:
    final_data[var] = final_data[var].astype('category')
    
    for var in num_variables:
        final_data[var] = final_data[var].astype('float')

    for var in date_variables:
        final_data[var] =  pd.to_datetime(final_data[var], format='%Y-%m-%d')

import datetime as dt
    
final_data['RemainingDays'] = final_data['DateOfIncident'] - final_data['DateOfPolicyCoverage']
final_data['RemainingDays'] = final_data['RemainingDays'].astype(dt.timedelta).map(lambda x:x.days)

# Creating a new feature Financial Status by taking the adding of CapitalGains and CapitalLoss"
final_data['FinancialStatus'] = final_data['CapitalGains'] + final_data['CapitalLoss']
final_data['FinancialStatus'] = final_data['FinancialStatus'].astype('float')

# Feature AmountOfTotalClaim by adding of AmountOfInjuryClaim + AmountOfPropertyClaim + AmountOfVehicleDamage"
final_data['AmountOfTotalClaim'] = final_data['AmountOfInjuryClaim'] + final_data['AmountOfPropertyClaim'] + final_data['AmountOfVehicleDamage']
final_data['AmountOfTotalClaim'] = final_data['AmountOfTotalClaim'].astype('int')

# Transforming the Umbrealla Limit to Positive Number"
final_data['UmbrellaLimit'] = final_data['UmbrellaLimit'].apply(lambda x: x if x > 0 else 0)

# Creating a new feature DistanceClaim by finding the difference of IncidentState  and InsurancePolicyState"

def state_assignment(x):
    state_map = {'State1': 0 ,  'State2' : 1,  'State3':2,'State4' :3 , 'State5' :4,  'State6':5,  'State7':6, 'State8':7, 'State9':8}
    return(state_map[x])
    
final_data['IncidentState_Num'] = final_data['IncidentState'].apply(lambda x: state_assignment(x))
final_data['IncidentState_Num'] = final_data['IncidentState_Num'].astype('int')
final_data['InsurancePolicyState_Num'] = final_data['InsurancePolicyState'].apply(lambda x: state_assignment(x))
final_data['InsurancePolicyState_Num'] = final_data['InsurancePolicyState_Num'].astype('int')
final_data['DistanceClaim'] = np.absolute(final_data['InsurancePolicyState_Num'] - final_data['IncidentState_Num'])

#final_data['DistanceClaim'] = final_data['DistanceClaim'].astype('category')

#Convert the InsuredAge feature to Categorical with levels ( Youngster, MiddleAge, SeniorCitizen)"

#def convertInsuredAge(x):

def convertInsuredAge(x):
    if (x < 35):
        return 'Youngster',
    elif (x >= 35 and x < 55):
        return 'MiddleAge'
    else:
        return 'SeniorCitizen'

final_data['InsuredAgeGroup'] = final_data['InsuredAge'].apply(lambda x : convertInsuredAge(x))
final_data['InsuredAgeGroup'] = final_data['InsuredAgeGroup'].astype('category')

# Splitting the Final Merged Dataset to Train and Test"
train_final_data = final_data[final_data['dataset'] =='train']
train_final_data.drop('dataset',axis=1,inplace=True)
test_final_data = final_data[final_data['dataset'] =='test']
test_final_data.drop('dataset',axis=1,inplace=True)
train_final_data[target] = target_value

# Function to delete the columns from the Dataframe\n"
def drop_columns(df,cols):
    for col in cols:
        df.drop([col],inplace=True,axis=1)
                                              
# Getting the Numerical and Categorical Variables"
numerical_variables = list(train_final_data._get_numeric_data().columns)
catergorical_variables = list(set(train_final_data.columns) - set(numerical_variables))
catergorical_variables.remove(target)
temp_num_df = train_final_data[numerical_variables]

# Creating the Correlation Plots\n",
plt.subplots(figsize=(10,10))
sns.heatmap(temp_num_df.corr(), annot=True)

# Checking the Imbalance of the target variable"
target_tab = pd.crosstab(index = train_final_data[target], columns="count")
target_tab.plot.bar()
plt.show()

# Understanding the Distribution fo the Numerical Variable"

def get_skew_kurtosis(df):
    cols = df._get_numeric_data().columns
    for i in cols:
        print("For the var %s"%(i))
        print("Skewness:" ,df[i].skew())
        print("Kurtosis:" % df[i].kurt())
        sns.distplot(df[i], fit=norm);
        fig = plt.figure()
        res = stats.probplot(df[i], plot=plt)
        plt.show()

   # except:
    #    pass
    
get_skew_kurtosis(train_final_data)
   
# InsurancePolicyState Bar Plot\n",
## State 3 where Policy were registered have maximum fraud"
sns.countplot(x="InsurancePolicyState", hue=target, data=train_final_data)
# State 3 where Policy were registered have maximum fraud"
# InsurancePolicyState Bar Plot\n",
## State 7 where Incidents happen were registered have maximum fraud"
sns.countplot(x="IncidentState", hue=target, data=train_final_data)
        
# State 7 where Incidents happen were registered have maximum fraud"
# SeverityOfIncident Bar Plot\n",
## # Major Damage has the maximum fraud"

sns.countplot(x="SeverityOfIncident", hue=target, data=train_final_data)
        
# Major Damage has the maximum fraud"
sns.countplot(x="NumberOfVehicles", hue=target, data=train_final_data)
# People with 3 Number of Vehicles have found to be making the Fraud considerable"
# AuthoritiesContacted Bar Plot\n",
## Shockingly after contacting Police Authorities we are seeing the Fraud "
sns.countplot(x="AuthoritiesContacted", hue=target, data=train_final_data)

#Shockingly after contacting Police Authorities we are seeing the Fraud "
# Car with the Year of Make 1999, 2000 and 2007 are being involved in the Fraud"
plt.subplots(figsize=(10,10))
sns.countplot(x="VehicleYOM", hue=target, data=train_final_data)
        
# Car with the Year of Make 1999, 2000 and 2007 are being involved in the Fraud"
sns.countplot(x="TypeOfCollission", hue=target, data=train_final_data)

sns.countplot(x="BodilyInjuries", hue=target, data=train_final_data)
 
sns.countplot(x="InsuredEducationLevel", hue=target, data=train_final_data)
 
plt.subplots(figsize=(10,10))
    
sns.countplot(x="VehicleMake", hue=target, data=train_final_data)
   
  
  
   # City 1 and City 2 are involved in the Fraud"
plt.subplots(figsize=(10,10))
sns.countplot(x="IncidentCity", hue=target, data=train_final_data)
   
plt.subplots(figsize=(10,10))
sns.countplot(x="InsuredGender", hue=target, data=train_final_data)
plt.subplots(figsize=(8,8))
sns.countplot(x="TypeOfIncident", hue=target, data=train_final_data)
plt.subplots(figsize=(10,10))
sns.countplot(x="IncidentTime", hue=target, data=train_final_data)
plt.subplots(figsize=(10,10))
sns.countplot(x="InsuredRelationship", hue=target, data=train_final_data)
plt.subplots(figsize=(17,17))
sns.countplot(x="InsuredOccupation", hue=target, data=train_final_data)
plt.subplots(figsize=(17,17))
sns.countplot(x="InsuredHobbies", hue=target, data=train_final_data)
sns.countplot(x="Witnesses", hue=target, data=train_final_data)
test_cat_variables = catergorical_variables + ['ReportedFraud']

def anova_one_way(frame,numerical_variable):
    anv = pd.DataFrame()
    anv['feature'] = test_cat_variables
    pvals = []
    for c in test_cat_variables:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls][numerical_variable].values
            samples.append(s)
            pval = stats.f_oneway(*samples)[1]
            pvals.append(pval)
            anv['pval'] = pvals
        return anv.sort_values('pval')
    
    
a = anova_one_way(train_final_data,'InsuredAge')
a['disparity'] = np.log(1./a['pval'].values)
sns.barplot(data=a, x='feature', y='disparity')
x=plt.xticks(rotation=90)

from scipy.stats import chi2_contingency
def chisq_of_df_cols(df, c1, c2):
    groupsizes = df.groupby([c1, c2]).size()
    ctsum = groupsizes.unstack(c1)
    # fillna(0) is necessary to remove any NAs which will cause exceptions\n",
    return(chi2_contingency(ctsum.fillna(0)))
p_value = chisq_of_df_cols(final_data,'TypeOfCollission','TypeOfIncident')
print("P Value is:",p_value)
train_final_data.to_csv('data/preprocessed/final_train.csv',header=True,index=False)
test_final_data.to_csv('data/preprocessed/unseen.csv',header=True,index=False)
   # Removing the Columns\n",
columns_drop = [
        # Unique values columns\n",
    'CustomerID',
    'VehicleID',
    'InsurancePolicyNumber',
    'Country',
      # Created variables\n",
    #'IncidentState_Num',
    #'InsurancePolicyState_Num',
        
        # Date Variables\n",
    #'DateOfPolicyCoverage',
    #'DateOfIncident',
    ]
  
drop_columns(train_final_data,columns_drop)
drop_columns(test_final_data,columns_drop)
high_na_columns = ['PropertyDamage','PoliceReport']
drop_columns(train_final_data,high_na_columns)
drop_columns(test_final_data,high_na_columns)

# H2O Part Starts from Here"
target = "ReportedFraud"
    
preprocessed_csv_path = 'data/preprocessed/final_train.csv'
    
unseen_csv_path = 'data/preprocessed/unseen.csv'
    
prediction_folder = 'data/prediction/'
    
max_seconds = 300
    
seed = 12345

    # Option specifies the scheme to use for cross-validation fold assignment.\n",
    # Values : \"Auto\", \"Random\" , \"Modulo\" , \"Stratified\"\n",
fold_assignment="Modulo",
    
nfolds = 10 # Cross Fold Validation\n",
    
keep_cross_validation_predictions=True
    
    # select the values for lambda_ to grid over\n",
glm_hyper_params = {
                   'lambda' : list(np.arange(0,1,0.01)),
                   'alpha' : [0,0.25,0.5,0.75,1],
                    }
gbm_hyper_params = {
        
           'ntrees':list(np.arange(100,2000,100)),
           'max_depth':list(np.arange(1,20)),
           'min_rows':[1,5,10,20,50,100],
           'learn_rate':list(np.arange(0.001,0.01,0.001)),
           'sample_rate':list(np.arange(0.3,1,0.05)),
           'col_sample_rate' :list(np.arange(0.3,1,0.05))    
    
    }
balance_classes = True # Is Applicable only if the Target Response Variable is imbalanced\n",

search_criteria = {
       "strategy" : "RandomDiscrete",
       "max_runtime_secs": max_seconds,
       "max_models": 500,
       "stopping_metric": "AUTO",
       "stopping_tolerance": 0.00001,
       "stopping_rounds": 5,
       "seed": 123456,
    }
  # Import the preprocessed dataset"
   
preprocessed_df = h2o.import_file(path=preprocessed_csv_path,header=1)
#preprocessed_df = h2o.import_file(path=preprocessed_csv_path,header=1)
cat_variables = [
        'VehicleModel',
        'IncidentAddress',
        'InsuredHobbies',
        'InsuredZipCode',
        'SeverityOfIncident',
        'InsuredEducationLevel',
        'InsuredOccupation',
        'IncidentTime',
        'VehicleMake',
    
        
 # Commenting all the variables as these are not that significant      \n",
    
        'InsuredAgeGroup',
        'TypeOfIncident',
        'AuthoritiesContacted',
        'InsuredGender',
    
        'IncidentCity',
        'Policy_CombinedSingleLimit',
        'VehicleYOM',
        'TypeOfCollission',
        'Witnesses',
        'NumberOfVehicles',
        'BodilyInjuries',
        'IncidentState',
        'DistanceClaim',
        'Policy_Deductible',
        'InsuredAge',
        'InsuredRelationship',
        'InsurancePolicyState'
    ]
    
num_variables = [
            'CapitalLoss',

     # Commenting all the variables as these are not that significant   \n",
            'AmountOfTotalClaim',
            'InsuredAge',
            'AmountOfInjuryClaim',
            'AmountOfPropertyClaim',
            'AmountOfVehicleDamage',
            'CapitalGains',
            'CustomerLoyaltyPeriod',
            'PolicyAnnualPremium',
            'UmbrellaLimit',
            'RemainingDays',
            'FinancialStatus'
    
    ]
   
  
for var in cat_variables:
    preprocessed_df[var] = preprocessed_df[var].asfactor()
    
for var in num_variables:
    preprocessed_df[var] = preprocessed_df[var].asnumeric(),

preprocessed_df[target] = preprocessed_df[target].asfactor()   
predictors = num_variables + cat_variables
    
    
train,valid,test = preprocessed_df.split_frame(ratios=[.7,.15],seed=12345)
import time
ts = time.time()
ts = int(ts)

# Generate a GLM model using the training dataset\n",
glm_classifier = H2OGeneralizedLinearEstimator(family="binomial", nfolds=nfolds,
                                                keep_cross_validation_predictions = True,
                                                balance_classes=balance_classes,
                                                standardize=True,
                                                seed=seed)
    # glm_classifier.train(x = predictors, y = target, training_frame = train, validation_frame = valid)\n",
glm_grid = H2OGridSearch(model = glm_classifier,
                     hyper_params = glm_hyper_params,
                     search_criteria = search_criteria,grid_id = "glm_grid_1_"+str(ts))
    # train using the grid\n",
glm_grid.train(x = predictors, y = target, training_frame = train, validation_frame = valid)
    
glm_grid_table = glm_grid.get_grid(sort_by = 'auc', decreasing = True)
print(glm_grid_table)
glm_classifier._model_json['output']['validation_metrics']
    
glm_final_model = glm_grid_table.models[0]
    
   
train_performance = glm_final_model.model_performance(train)
train_performance.plot()
    
    
valid_performance = glm_final_model.model_performance(valid)
valid_performance.plot()
    
    
test_performance = glm_final_model.model_performance(test)
test_performance.plot()
import time
ts = time.time()
ts = int(ts)
    # Hyper Parameters"
rf_hyper_params = {
    'ntrees':list(np.arange(100,2000,100)),
    'mtries':list(np.arange(1,len(predictors)-1,1)),
    'max_depth':list(np.arange(5,25,2))
    }
rf_classifier = H2ORandomForestEstimator(distribution="bernoulli", nfolds=nfolds,
                                     fold_assignment=fold_assignment,
                                     keep_cross_validation_predictions=keep_cross_validation_predictions,
                                     balance_classes=balance_classes,
                                     seed=seed)
    # build grid search with previously made GLM and hyperparameters\n",
rf_grid = H2OGridSearch(model = rf_classifier,
                     hyper_params = rf_hyper_params,
                     search_criteria = search_criteria,grid_id = "rf_grid_1_"+str(ts))
    
    # train using the grid
rf_grid.train(x = predictors, y = target, training_frame = train, validation_frame = valid)
    
    
rf_grid_table = rf_grid.get_grid(sort_by = 'auc', decreasing = True)
print(rf_grid_table)
        
rf_final_model = rf_grid_table.models[0]
train_performance = rf_final_model.model_performance(train)
train_performance.plot()
valid_performance = rf_final_model.model_performance(valid)
valid_performance.plot()
test_performance = rf_final_model.model_performance(test)
test_performance.plot()
 # Displaying the Variable Importance Graph"
plt.rcdefaults()
fig, ax = plt.subplots()
variables = rf_final_model._model_json['output']['variable_importances']['variable']
y_pos = np.arange(len(variables))
scaled_importance = rf_final_model._model_json['output']['variable_importances']['scaled_importance']
ax.barh(y_pos, scaled_importance, align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.invert_yaxis()
ax.set_xlabel('Scaled Importance')
ax.set_title('Variable Importance for RF')
plt.show()
import time
ts = time.time()
ts = int(ts)
    
    
gbm_classifier = H2OGradientBoostingEstimator(distribution="bernoulli",
                                      nfolds=nfolds,
                                      fold_assignment=fold_assignment,
                                      keep_cross_validation_predictions=keep_cross_validation_predictions,
                                      balance_classes=balance_classes,
                                      seed=seed)
    
    # build grid search with previously made GLM and hyperparameters
gbm_grid = H2OGridSearch(model = gbm_classifier,
                     hyper_params = gbm_hyper_params,
                     search_criteria = search_criteria,grid_id = "gbm_grid_1_"+str(ts))
    
# train using the grid
gbm_grid.train(x = predictors, y = target, training_frame = train, validation_frame = valid)
gbm_grid_table = gbm_grid.get_grid(sort_by = 'auc', decreasing = True)
    
print(gbm_grid_table)
   
gbm_final_model = gbm_grid_table.models[0]
  
train_performance = gbm_final_model.model_performance(train)
train_performance.plot()
valid_performance = gbm_final_model.model_performance(valid)
valid_performance.plot()
test_performance = gbm_final_model.model_performance(test)
test_performance.plot()
  
import time
ts = time.time()
ts = int(ts)
gbm_model = gbm_final_model
rf_model = rf_final_model
ensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_binomial_"+str(ts),
                                       base_models=[gbm_model, rf_model])
  
ensemble.train(x=predictors, y=target, training_frame=train,validation_frame=valid)
  
train_performance = ensemble.model_performance(train)
train_performance.plot()
valid_performance = ensemble.model_performance(valid)
valid_performance.plot()
test_performance = ensemble.model_performance(test)
test_performance.plot()
 
    # Eval ensemble performance on the test data\n",
perf_stack_test = ensemble.model_performance(test)
   
    # Compare to base learner performance on the test set
perf_gbm_test = gbm_model.model_performance(test)
perf_rf_test = rf_model.model_performance(test)
baselearner_best_auc_test = max(perf_gbm_test.auc(), perf_rf_test.auc())
stack_auc_test = perf_stack_test.auc()
print("Best Base-learner Test AUC:  {0}".format(baselearner_best_auc_test))
print("Ensemble Test AUC:  {0}".format(stack_auc_test))
 
import time
ts = time.time()
ts = int(ts)
   
    # Specify GBM hyperparameters for the grid
ensemble_hyper_params = {"learn_rate": [0.01, 0.03],
                         "max_depth": [3, 4, 5, 6, 9],
                         "sample_rate": [0.7, 0.8, 0.9, 1.0],
                         "col_sample_rate": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                         "ntrees" : [30,40,50,60,70,80,90,100],
                             }
ensemble_search_criteria = {"strategy": "RandomDiscrete", "max_models": 200, "seed": 12345,"max_runtime_secs": max_seconds}
    # Train the grid\n",
grid = H2OGridSearch(model=H2OGradientBoostingEstimator(
                                                        seed=12345,
                                                        nfolds=nfolds,
                                                        fold_assignment="Modulo",
                                                        keep_cross_validation_predictions=True
                                                        ))
hyper_params=gbm_hyper_params
search_criteria=ensemble_search_criteria
grid_id= ("gbm_grid_binomial_"+str(ts))
grid.train(x=predictors, y=target, training_frame=train,validation_frame=valid)

    # Train a stacked ensemble using the GBM grid
ensemble_stack = H2OStackedEnsembleEstimator(model_id="my_ensemble_gbm_grid_binomial_"+str(ts),
                                             base_models=grid.model_ids)
ensemble_stack.train(x=predictors, y=target, training_frame=train,validation_frame=valid)
    
# Eval ensemble performance on the test data
perf_stack_test = ensemble_stack.model_performance(test)
    
# Compare to base learner performance on the test set\n",
baselearner_best_auc_test = max([h2o.get_model(model).model_performance(test_data=test).auc() for model in grid.model_ids])
stack_auc_test = perf_stack_test.auc()
print("Best Base-learner Test AUC:  {0}".format(baselearner_best_auc_test))
print("Ensemble GBM Test AUC:  {0}".format(stack_auc_test))
    
# Generate predictions on a test set (if neccessary)\n",
pred = ensemble_stack.predict(test) 
train_performance = ensemble_stack.model_performance(train)
train_performance.plot()
 
valid_performance = ensemble_stack.model_performance(valid)
valid_performance.plot()
   
    
test_performance = ensemble_stack.model_performance(test)
test_performance.plot()
import time
ts = time.time()
ts = int(ts)
    
    # Specify RF hyperparameters for the grid\n",
    
ensemble_search_criteria = {"strategy": "RandomDiscrete", "max_models": 200, "seed": 12345,"max_runtime_secs": max_seconds}
rf_classifier = H2ORandomForestEstimator(distribution="bernoulli", nfolds=nfolds,
                                 fold_assignment=fold_assignment,
                                 keep_cross_validation_predictions=keep_cross_validation_predictions,
                                 balance_classes=balance_classes,
                                 seed=seed)
  
    # Train the grid\n",
grid = H2OGridSearch(model=H2ORandomForestEstimator(seed=seed,
                                                    nfolds=nfolds,
                                                    fold_assignment="Modulo",
                                                    keep_cross_validation_predictions=True),
                                                    hyper_params=rf_hyper_params,
                                                    search_criteria=ensemble_search_criteria,
                                                    grid_id="rf_grid_binomial_"+str(ts))
grid.train(x=predictors, y=target, training_frame=train,validation_frame=valid)

    # Train a stacked ensemble using the GBM grid\n",
ensemble_rf_stack = H2OStackedEnsembleEstimator(model_id="my_ensemble_rf_grid_binomial_"+str(ts),
                                       base_models=grid.model_ids)
ensemble_rf_stack.train(x=predictors, y=target, training_frame=train,validation_frame=valid)
    
# Eval ensemble performance on the test data\n",
perf_rf_stack_test = ensemble_rf_stack.model_performance(test)
    
# Compare to base learner performance on the test set\n",
baselearner_best_auc_test = max([h2o.get_model(model).model_performance(test_data=test).auc() for model in grid.model_ids])
stack_auc_test = perf_rf_stack_test.auc()
print("Best Base-learner Test AUC:  {0}".format(baselearner_best_auc_test))
print("Ensemble RF Test AUC:  {0}".format(stack_auc_test))
    
# Generate predictions on a test set (if neccessary)\n",
pred = ensemble_rf_stack.predict(test)
train_performance = ensemble_rf_stack.model_performance(train)
train_performance.plot()
valid_performance = ensemble_rf_stack.model_performance(valid)
valid_performance.plot()
test_performance = ensemble_rf_stack.model_performance(test)
test_performance.plot()
 
from h2o.automl import H2OAutoML
# Run AutoML for 30 seconds\n",
aml = H2OAutoML(nfolds=10,max_runtime_secs = max_seconds)
aml.train(x = predictors, y = target,
         training_frame = train,
         leaderboard_frame = valid)

    # View the AutoML Leaderboard\n",
lb = aml.leaderboard
  
unseen_df = h2o.import_file(path=unseen_csv_path,header=1)

# Converting the Datatype of the Variable"
  
for var in cat_variables:
    unseen_df[var] = unseen_df[var].asfactor()
   
for var in num_variables:
    unseen_df[var] = unseen_df[var].asnumeric()
 
nseen_pandas_df = unseen_df.as_data_frame()
unseen_df = unseen_df.drop('CustomerID')
  
    # Predict using the GLM model and the testing dataset\n",
    # predict_glm = best_logistic_model.predict(unseen_df)\n",
    
predict_glm = glm_final_model.predict(unseen_df)

    # Predict using the RF model and the testing dataset
predict_rf = rf_final_model.predict(unseen_df)
    
    # Predict using the GBM model and the testing dataset\n",
predict_gbm = gbm_final_model.predict(unseen_df)

    # Predict using the Ensemble of GBM + RF model and the testing dataset\n",
predict_ensemble = ensemble.predict(unseen_df)

    # Predict using the Ensemble GBM\n",

predict_ensemble_stack = ensemble_stack.predict(unseen_df)

    # Predict using the Ensemble RF\n",
predict_ensemble_rf_stack = ensemble_rf_stack.predict(unseen_df)

    # Predict using the Auto ML\n",
predict_auto_ml = aml.leader.predict(unseen_df)
 

prediction_glm_df = predict_glm.as_data_frame()
type(prediction_glm_df)
prediction_glm_df['CustomerID'] = unseen_pandas_df['CustomerID']
prediction_glm_df[target] = prediction_glm_df['predict']
    
    ###############################################\n",

prediction_gbm_df = predict_gbm.as_data_frame()
type(prediction_gbm_df)

prediction_gbm_df['CustomerID'] = unseen_pandas_df['CustomerID']
prediction_gbm_df[target] = prediction_gbm_df['predict']
   
prediction_rf_df = predict_rf.as_data_frame()
type(prediction_rf_df)
    
prediction_rf_df['CustomerID'] = unseen_pandas_df['CustomerID']
prediction_rf_df[target] = prediction_rf_df['predict']
    
    ###############################################\n",
   
prediction_ensemble_df = predict_ensemble.as_data_frame()
type(prediction_glm_df)
    
prediction_ensemble_df['CustomerID'] = unseen_pandas_df['CustomerID']
prediction_ensemble_df[target] = prediction_ensemble_df['predict']
 
    ###############################################\n",
   
prediction_automl_df = predict_auto_ml.as_data_frame()
type(prediction_automl_df)
    
prediction_automl_df['CustomerID'] = unseen_pandas_df['CustomerID']
prediction_automl_df[target] = prediction_automl_df['predict']
   
    #########################################################\n",

prediction_ensemble_stack_df = predict_ensemble_stack.as_data_frame()

prediction_ensemble_stack_df['CustomerID'] = unseen_pandas_df['CustomerID']
prediction_ensemble_stack_df[target] = prediction_ensemble_stack_df['predict']

    #################################################################\n",
   
prediction_ensemble_rf_stack_df = predict_ensemble_rf_stack.as_data_frame()
 
prediction_ensemble_rf_stack_df['CustomerID'] = unseen_pandas_df['CustomerID']
prediction_ensemble_rf_stack_df[target] = prediction_ensemble_rf_stack_df['predict']
   
import time
ts = time.time()
ts = int(ts)
  
prediction_glm_df[['CustomerID',target]].to_csv('data/prediction/glm/prediction_glm_'+str(ts)+'.csv',header=True,index=False)
   
 # Exporting the results of GBM model on the Unseen Dataset\n",
    
prediction_gbm_df[['CustomerID',target]].to_csv('data/prediction/gbm/prediction_gbm_'+str(ts)+'.csv',header=True,index=False)
    
    # Exporting the results of RF model on the Unseen Dataset\n",
    
prediction_rf_df[['CustomerID',target]].to_csv('data/prediction/rf/prediction_rf_'+str(ts)+'.csv',header=True,index=False)

    # Exporting the results of GBM + RF Ensemble model on the Unseen Dataset\n",
    
prediction_ensemble_df[['CustomerID',target]].to_csv('data/prediction/ensemble/prediction_ensemble_'+str(ts)+'.csv',header=True,index=False)

    # Exporting the results of AutoML model on the Unseen Dataset\n",

prediction_automl_df[['CustomerID',target]].to_csv('data/prediction/automl/prediction_automl_'+str(ts)+'.csv',header=True,index=False)

    # Exporting the results of Ensemble_GBM model on the Unseen Dataset\n",
    
prediction_ensemble_stack_df[['CustomerID',target]].to_csv('data/prediction/gbm_stack/prediction_gbm_stack_'+str(ts)+'.csv',header=True,index=False)    # Exporting the results of Ensemble_RF model on the Unseen Dataset\n"
 
prediction_ensemble_rf_stack_df[['CustomerID',target]].to_csv('data/prediction/rf_stack/prediction_rf_stack_'+str(ts)+'.csv',header=True,index=False)
   

