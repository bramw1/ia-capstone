import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def filter_year(df):
	df['Date'] = pd.to_datetime(df['Date'])
	df['Order.Year'] = df['Date'].dt.year
	df['Order.Month'] = df['Date'].dt.month
	df['Order.Day'] = df['Date'].dt.dayofyear

	year = [2012, 2013]
	mask = df['Order.Year'].isin(year)

	df1 = df[mask].copy()

	return(df1)

	

def feature_eng(df, df2):
	df['Profit'] = df['State Bottle Retail'] - df['State Bottle Cost']
	df['Bottle Volume (L)'] = df['Bottle Volume (ml)'] / 1000

	ageinfo = df2[['Item Number', 'Age']]
	proofinfo = df2[['Item Number', 'Proof']]

	df['Item Number'] = df['Item Number'].apply(str)
	df2['Item Number'] = df2['Item Number'].apply(str)

	df['Item Number'] = df['Item Number'].str.replace(r'\D', '')
	df2['Item Number'] = df2['Item Number'].str.replace(r'\D', '')

	info_dict = dict(ageinfo.values)
	df['Age'] = df['Item Number'].map(info_dict)

	info_dict = dict(proofinfo.values)
	df['Proof'] = df['Item Number'].map(info_dict)

	return(df)


def clean_data(df):
	li = ['Profit', 'Age', 'Proof', 'Vendor Number', 'Category', 'County Number']
	
	for i in li:
		df = df[df[i].notnull()]

	return(df)




def vendor_lasso(df):
	train2 = df[['Order.Month', 'Order.Day', 'Order.Year',
        'Vendor Number', 'Pack', 'Bottle Volume (L)', 
        'Volume Sold (Liters)', 'Profit']].copy()

	sale_data = train2.loc[:, train2.columns != 'Profit']
	sale_target = train2['Profit']
	
	X_train, X_test, y_train, y_test = train_test_split(sale_data, sale_target, test_size=0.2, random_state=3)

	train_cols = X_train[['Order.Day', 'Order.Year', 'Order.Month', 'Bottle Volume (L)', 'Pack', 'Volume Sold (Liters)']]

	train_nomcols = X_train[['Vendor Number']]

	enc = OneHotEncoder(drop = 'first', sparse = False)

	encodecols = enc.fit_transform(train_nomcols)
	feature_names = enc.get_feature_names(['Vendor Number'])

	pd.DataFrame(encodecols, columns = feature_names)

	X_train = pd.concat([train_cols.reset_index(drop = True), pd.DataFrame(encodecols, columns = feature_names).astype(int).reset_index(drop = True)], axis = 1)

	lasso = Lasso()
	lasso.set_params(alpha=1, normalize=True)
	lasso.fit(X_train, y_train)
	print('The intercept is %.4f' %(lasso.intercept_))
	lassoCoef = pd.Series(lasso.coef_, index=X_train.columns).sort_values(ascending = False)
	print('The slopes are \n%s' %(lassoCoef))


def ageproof_lasso(df):
	train2 = df[['Order.Month', 'Order.Day', 'Order.Year',
        'Age', 'Proof', 'Pack', 'Bottle Volume (L)', 
        'Volume Sold (Liters)', 'Profit']].copy()

	sale_data = train2.loc[:, train2.columns != 'Profit']
	sale_target = train2['Profit']
	
	X_train, X_test, y_train, y_test = train_test_split(sale_data, sale_target, test_size=0.2, random_state=3)

	train_cols = X_train[['Age', 'Order.Day', 'Order.Year', 'Order.Month', 'Bottle Volume (L)', 'Pack', 'Volume Sold (Liters)']]

	train_nomcols = X_train[['Proof']]

	enc = OneHotEncoder(drop = 'first', sparse = False)

	encodecols = enc.fit_transform(train_nomcols)
	feature_names = enc.get_feature_names(['Proof'])

	pd.DataFrame(encodecols, columns = feature_names)

	X_train = pd.concat([train_cols.reset_index(drop = True), pd.DataFrame(encodecols, columns = feature_names).astype(int).reset_index(drop = True)], axis = 1)

	lasso = Lasso()
	lasso.set_params(alpha=1, normalize=True)
	lasso.fit(X_train, y_train)
	print('The intercept is %.4f' %(lasso.intercept_))
	lassoCoef = pd.Series(lasso.coef_, index=X_train.columns).sort_values(ascending = False)
	print('The slopes are \n%s' %(lassoCoef))


def category_county_lasso(df):
	train2 = df[['Order.Month', 'Order.Day', 'Order.Year',
	 'County Number', 'Category', 'Pack', 'Bottle Volume (L)', 
	 'Volume Sold (Liters)', 'Profit']].copy()

	sale_data = train2.loc[:, train2.columns != 'Profit']
	sale_target = train2['Profit']
	
	X_train, X_test, y_train, y_test = train_test_split(sale_data, sale_target, test_size=0.2, random_state=3)

	train_cols = X_train[['Order.Day', 'Order.Year', 'Order.Month', 'Bottle Volume (L)', 'Pack', 'Volume Sold (Liters)']]

	train_nomcols = X_train[['County Number', 'Category']]

	enc = OneHotEncoder(drop = 'first', sparse = False)

	encodecols = enc.fit_transform(train_nomcols)
	feature_names = enc.get_feature_names(['County Number', 'Category'])

	pd.DataFrame(encodecols, columns = feature_names)

	X_train = pd.concat([train_cols.reset_index(drop = True), pd.DataFrame(encodecols, columns = feature_names).astype(int).reset_index(drop = True)], axis = 1)

	lasso = Lasso()
	lasso.set_params(alpha=1, normalize=True)
	lasso.fit(X_train, y_train)
	print('The intercept is %.4f' %(lasso.intercept_))
	lassoCoef = pd.Series(lasso.coef_, index=X_train.columns).sort_values(ascending = False)
	print('The slopes are \n%s' %(lassoCoef))
