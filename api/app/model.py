import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

import lightgbm as lgbm

salary_data = pd.read_csv('data/ds_salaries.csv', index_col=0)
# print(salary_data)

x = salary_data.drop(labels=['salary_in_usd'], axis=1)
y = salary_data['salary_in_usd']


params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
}

# TODO: Encode columns invdidually, so that the encoding will be saved
# TODO: Move encoding to a separate function
columns_to_encode = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']

ct = ColumnTransformer([
        ('encoder', OrdinalEncoder(), columns_to_encode),
        # ('employment_type_encoder', OrdinalEncoder(), 'employment_type'),
        # ('job_title_encoder', OrdinalEncoder(), 'job_title'),
        # ('employee_residence_encoder', OrdinalEncoder(), 'employee_residence'),
        # ('company_location_encoder', OrdinalEncoder(), 'company_location'),
        # ('company_size_encoder', OrdinalEncoder(), 'company_size'),
    ], remainder='passthrough')
    
transformed_data = ct.fit_transform(x)
feature_names = ct.get_feature_names_out()
ct_params = ct.get_params()
print(feature_names)
print(ct_params)
# print(x)
print(transformed_data)
print(type(transformed_data))

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# lgbm_train = lgbm.Dataset(x_train, y_train)
# lgbm_test = lgbm.Dataset(x_test, y_test, reference=lgbm_train)

# model = lgbm.train(params, train_set=lgbm_train, valid_sets=lgbm_test, early_stopping_rounds=30)

# y_pred = model.predict(x_test)

# mse = mean_squared_error(y_test, y_pred)
# rmse = mse ** 0.5

# print(f"MSE = {mse:.3f}")
# print(f"RMSE = {rmse:.3f}")