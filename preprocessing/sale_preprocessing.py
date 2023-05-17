import pandas as pd
import pickle
import numpy as np

num_columns = ['log_Followers',
               'log_Avg Price',
               'log_First Day Revenue',
               'log_Brand Appearance',
               'Avg Discount',
               'Conversion']

log_columns = ['Followers', 'Avg Price', 'First Day Revenue', 'Brand Appearance']

def adding_log(train: pd.DataFrame, columns_to_log: list = log_columns):
    """ 
    Adding columns using log of provided information.
    """
    new_train = train.__deepcopy__()
    for column_name in columns_to_log:
        new_position = train.columns.get_loc(str(column_name)) + 1
        new_name = str ('log_' + str(column_name))
        new_train.insert(new_position, new_name, np.log(new_train[str(column_name)] + 1))
        new_train = new_train.drop(str(column_name), axis='columns')
        
    return (new_train)

def scale(df, scaler, numerical_columns: list = num_columns):
    """
    Returns dataframe with the given columns scaled.
    """
    new_train = df.__deepcopy__()
    num_train_data = new_train[numerical_columns]
    new_train[numerical_columns] = scaler.transform(num_train_data)

    return (new_train)

def sales_pre_pocessing(all_sales, cols_to_log, scaler, cols_to_scale):

    dummies_category = pd.get_dummies(all_sales['Category'])
    dummies_badges = all_sales['Badges'].str.strip('{}').str.replace('"', '').str.get_dummies(',')
    dropped_cols_sales = all_sales.drop(['Category','Badges'], axis = 1)

    all_sales_dummies = pd.concat([dropped_cols_sales, dummies_category, dummies_badges], axis = 1)

    all_sales_logged = adding_log(all_sales_dummies, cols_to_log)

    all_sales_scaled = scale(all_sales_logged, scaler, cols_to_scale,)
    
    return all_sales_scaled
