import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

df = pd.read_csv("bakery.csv")

st.title("Market Basket Analysis")

st.write(df.head())

basket = (df.groupby(['TransactionNo', 'Items'])['Items'].count().unstack().fillna(0))

basket = basket.map(lambda x: 1 if x > 0 else 0)
frequent_items = apriori(basket,min_support=0.02,use_colnames=True)

rules = association_rules(frequent_items,metric='lift',min_threshold=1)
st.write(rules)