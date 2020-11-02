import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

def year_mba(df, year1, year2):
	'''
	Make the Iowa liquor sales and distribution dataset more manageable by filtering data to a two-year range 
	in order to perform a market basket analysis 
	'''
	df['Date'] = pd.to_datetime(df['Date'])
	df['Order.Year'] = df['Date'].dt.year

	year = [year1, year2]
	mask = df['Order.Year'].isin(year)

	new_df = df[mask].copy()

	new_df['Item Description'] = new_df['Item Description'].str.title()
	new_df = new_df[~new_df['Bottles Sold'] < 0]

	basketframe = new_df.groupby(['Store Number', 'Date', 'Item Description'])['Bottles Sold'].sum().unstack().reset_index().fillna(0).set_index(['Store Number', 'Date'])

	basket_sets = basketframe.applymap(encode_units)
	basket_sets

	frequent_itemsets = fpgrowth(basket_sets, min_support = 0.05, use_colnames = True)
	rules = association_rules(frequent_itemsets, metric = 'lift', min_threshold = 1)

	return rules



