import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

df = pd.read_csv("bakery.csv")

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())


df = df.dropna(subset=['TransactionNo', 'Items'])

df['Items'] = df['Items'].astype(str)
df['Items'] = df['Items'].str.strip()
df = df.drop_duplicates()

print("\nCleaned Data:")
print(df.head())


basket = (df.groupby(['TransactionNo', 'Items'])['Items'].count().unstack().fillna(0))
basket = basket.map(lambda x: 1 if x > 0 else 0)
print("\nBasket Matrix:")
print(basket.head())

#apriori algorithm

frequent_itemsets = apriori(basket,min_support=0.02,use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by='support',ascending=False)
print("\nFrequent Itemsets:")
print(frequent_itemsets.head(10))

#association rules

rules = association_rules(frequent_itemsets,metric='lift',min_threshold=1)
rules = rules.sort_values(by='lift',ascending=False)
print("\nAssociation Rules:")
print(rules.head(10))

strong_rules = rules[(rules['confidence'] >= 0.5) & (rules['lift'] >= 1.2)]
print("\nStrong Rules:")
print(strong_rules[['antecedents','consequents','support','confidence','lift']])

top_items = df['Items'].value_counts().head(10)
print("\nTop Selling Items:")
print(top_items)

plt.figure(figsize=(10,6))
top_items.plot(kind='bar', color='skyblue')
plt.title("Top 10 Selling Items")
plt.xlabel("Items")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10,6))
sns.scatterplot(data=rules,x='support',y='confidence',size='lift',hue='lift',palette='viridis',sizes=(20, 200))
plt.title("Support vs Confidence")
plt.show()

#morning transactions

morning_df = df[df['Daypart'] == 'Morning']
morning_basket = (morning_df.groupby(['TransactionNo', 'Items'])['Items'].count().unstack().fillna(0))
morning_basket = morning_basket.map(lambda x: 1 if x > 0 else 0)

morning_itemsets = apriori(morning_basket,min_support=0.02,use_colnames=True)
morning_rules = association_rules(morning_itemsets,metric='lift',min_threshold=1)
print("\nMorning Rules:")
print(morning_rules.head())

#Weekend transactions

weekend_df = df[df['DayType'] == 'Weekend']
weekend_basket = (weekend_df.groupby(['TransactionNo', 'Items'])['Items'].count().unstack().fillna(0))

weekend_basket = weekend_basket.map(lambda x: 1 if x > 0 else 0)
weekend_itemsets = apriori(weekend_basket,min_support=0.02,use_colnames=True)
weekend_rules = association_rules(weekend_itemsets,metric='lift',min_threshold=1)
print("\nWeekend Rules:")
print(weekend_rules.head())

#fp-growth

fp_itemsets = fpgrowth(basket,min_support=0.02,use_colnames=True)

fp_rules = association_rules(fp_itemsets,metric='lift',min_threshold=1)
print("\nFP-Growth Rules:")
print(fp_rules.head())

print(f"\nTotal Transactions : {df['TransactionNo'].nunique()}")
print(f"Total Unique Items : {df['Items'].nunique()}")
print(f"Total Rules Found  : {len(rules)}")
print(f"Strong Rules Found : {len(strong_rules)}")
