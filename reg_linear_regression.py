import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def filter_year(df):
	'''
	Subsetting the full dataset by year for OneHotEncoder/linear regression
	'''
	df['Date'] = pd.to_datetime(df['Date'])
	df['Order.Year'] = df['Date'].dt.year
	df['Order.Month'] = df['Date'].dt.month
	df['Order.Day'] = df['Date'].dt.dayofyear

	year = [2012, 2013]
	mask = df['Order.Year'].isin(year)

	df1 = df[mask].copy()

	return(df1)

	

def feature_eng(df, df2):
	'''
	Feature engineering and merging a supplementary dataset (df2) onto the primary dataset (df) for additional information 
	(Two new columns from df2, age and proof)
	'''

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
	'''
	Removing null data for columns being analyzed
	'''
	li = ['Profit', 'Age', 'Proof', 'Vendor Number', 'Category', 'County Number']
	
	for i in li:
		df = df[df[i].notnull()]

	return(df)




def vendor_lasso(df):
	'''
	Creating a subsetted version of the full dataset ('new_train') and running lasso regression on data with OneHotEncoded vendor column 
	'''
	new_train = df[['Order.Month', 'Order.Day', 'Order.Year',
        'Vendor Number', 'Pack', 'Bottle Volume (L)', 
        'Volume Sold (Liters)', 'Profit']].copy()

	sale_data = new_train.loc[:, new_train.columns != 'Profit']
	sale_target = new_train['Profit']
	
	X_train, X_test, y_train, y_test = train_test_split(sale_data, sale_target, test_size=0.2, random_state=3)

	new_train_cols = X_train[['Order.Day', 'Order.Year', 'Order.Month', 'Bottle Volume (L)', 'Pack', 'Volume Sold (Liters)']]

	new_train_nomcols = X_train[['Vendor Number']]

	enc = OneHotEncoder(drop = 'first', sparse = False)

	encodecols = enc.fit_transform(new_train_nomcols)
	feature_names = enc.get_feature_names(['Vendor Number'])

	pd.DataFrame(encodecols, columns = feature_names)

	X_train = pd.concat([new_train_cols.reset_index(drop = True), pd.DataFrame(encodecols, columns = feature_names).astype(int).reset_index(drop = True)], axis = 1)

	'''lasso = Lasso()
	lasso.set_params(alpha=1, normalize=True)
	lasso.fit(X_train, y_train)
	print('The intercept is %.4f' %(lasso.intercept_))
	lassoCoef = pd.Series(lasso.coef_, index=X_train.columns).sort_values(ascending = False)
	print('The slopes are \n%s' %(lassoCoef))'''

	lasso = Lasso()
	coefs = []

	alphaRange = np.linspace(1e-3,20,20)
	for alpha in alphaRange:
	    lasso.set_params(alpha=alpha, normalize = True)  
	    lasso.fit(X_train, y_train)
	    coefs.append(lasso.coef_)


	coefs = pd.DataFrame(np.array(coefs), columns=X_train.columns)

	for name in coefs.columns:
   		plt.plot(alphaRange, coefs[name])
	plt.xlabel('Alpha')
	plt.ylabel("Coefficients")
	plt.title('Change of Lasso Slopes Varying Alpha')


def ageproof_lasso(df):
	'''
	Creating a subsetted version of the full dataset ('new_train') and running lasso regression on data with OneHotEncoded proof column 

	'''
	new_train = df[['Order.Month', 'Order.Day', 'Order.Year',
        'Age', 'Proof', 'Pack', 'Bottle Volume (L)', 
        'Volume Sold (Liters)', 'Profit']].copy()

	sale_data = new_train.loc[:, new_train.columns != 'Profit']
	sale_target = new_train['Profit']
	
	X_train, X_test, y_train, y_test = train_test_split(sale_data, sale_target, test_size=0.2, random_state=3)

	new_train_cols = X_train[['Age', 'Order.Day', 'Order.Year', 'Order.Month', 'Bottle Volume (L)', 'Pack', 'Volume Sold (Liters)']]

	new_train_nomcols = X_train[['Proof']]

	enc = OneHotEncoder(drop = 'first', sparse = False)

	encodecols = enc.fit_transform(new_train_nomcols)
	feature_names = enc.get_feature_names(['Proof'])

	pd.DataFrame(encodecols, columns = feature_names)

	X_train = pd.concat([new_train_cols.reset_index(drop = True), pd.DataFrame(encodecols, columns = feature_names).astype(int).reset_index(drop = True)], axis = 1)

	lasso = Lasso()
	coefs = []

	alphaRange = np.linspace(1e-3,20,20)
	for alpha in alphaRange:
	    lasso.set_params(alpha=alpha, normalize = True)  
	    lasso.fit(X_train, y_train)
	    coefs.append(lasso.coef_)


	coefs = pd.DataFrame(np.array(coefs), columns=X_train.columns)

	for name in coefs.columns:
   		plt.plot(alphaRange, coefs[name])
	plt.xlabel('Alpha')
	plt.ylabel("Coefficients")
	plt.title('Change of Lasso Slopes Varying Alpha')


def category_lasso(df):
	'''
	Creating a subsetted version of the full dataset ('new_train') and running lasso regression on data with OneHotEncoded
	product category column 
	'''

	new_train = df[['Order.Month', 'Order.Day', 'Order.Year', 
	'Category', 'Pack', 'Bottle Volume (L)', 
	 'Volume Sold (Liters)', 'Profit']].copy()

	sale_data = new_train.loc[:, new_train.columns != 'Profit']
	sale_target = new_train['Profit']
	
	X_train, X_test, y_train, y_test = train_test_split(sale_data, sale_target, test_size=0.2, random_state=3)

	new_train_cols = X_train[['Order.Day', 'Order.Year', 'Order.Month', 'Bottle Volume (L)', 'Pack', 'Volume Sold (Liters)']]

	new_train_nomcols = X_train[['Category']]

	enc = OneHotEncoder(drop = 'first', sparse = False)

	encodecols = enc.fit_transform(new_train_nomcols)
	feature_names = enc.get_feature_names(['Category'])

	pd.DataFrame(encodecols, columns = feature_names)

	X_train = pd.concat([new_train_cols.reset_index(drop = True), pd.DataFrame(encodecols, columns = feature_names).astype(int).reset_index(drop = True)], axis = 1)

	lasso = Lasso()
	coefs = []

	alphaRange = np.linspace(1e-3,20,20)
	for alpha in alphaRange:
	    lasso.set_params(alpha=alpha, normalize = True)  
	    lasso.fit(X_train, y_train)
	    coefs.append(lasso.coef_)


	coefs = pd.DataFrame(np.array(coefs), columns=X_train.columns)

	for name in coefs.columns:
   		plt.plot(alphaRange, coefs[name])
	plt.xlabel('Alpha')
	plt.ylabel("Coefficients")
	plt.title('Change of Lasso Slopes Varying Alpha')

def county_lasso(df):
	'''
	Creating a subsetted version of the full dataset ('new_train') and running lasso regression on data with OneHotEncoded county column 
	'''

	new_train = df[['Order.Month', 'Order.Day', 'Order.Year',
	 'County Number', 'Pack', 'Bottle Volume (L)', 
	 'Volume Sold (Liters)', 'Profit']].copy()

	sale_data = new_train.loc[:, new_train.columns != 'Profit']
	sale_target = new_train['Profit']
	
	X_train, X_test, y_train, y_test = train_test_split(sale_data, sale_target, test_size=0.2, random_state=3)

	new_train_cols = X_train[['Order.Day', 'Order.Year', 'Order.Month', 'Bottle Volume (L)', 'Pack', 'Volume Sold (Liters)']]

	new_train_nomcols = X_train[['County Number']]

	enc = OneHotEncoder(drop = 'first', sparse = False)

	encodecols = enc.fit_transform(new_train_nomcols)
	feature_names = enc.get_feature_names(['County Number'])

	pd.DataFrame(encodecols, columns = feature_names)

	X_train = pd.concat([new_train_cols.reset_index(drop = True), pd.DataFrame(encodecols, columns = feature_names).astype(int).reset_index(drop = True)], axis = 1)

	lasso = Lasso()
	coefs = []

	alphaRange = np.linspace(1e-3,20,20)
	for alpha in alphaRange:
	    lasso.set_params(alpha=alpha, normalize = True)  
	    lasso.fit(X_train, y_train)
	    coefs.append(lasso.coef_)


	coefs = pd.DataFrame(np.array(coefs), columns=X_train.columns)

	for name in coefs.columns:
   		plt.plot(alphaRange, coefs[name])
	plt.xlabel('Alpha')
	plt.ylabel("Coefficients")
	plt.title('Change of Lasso Slopes Varying Alpha')