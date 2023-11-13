import streamlit as st
import csv
import mlxtend.frequent_patterns 
import mlxtend.preprocessing
import pandas as pd
import plotly.express as px


st.sidebar.warning("Please remember that Market Basket Analysis primarily studies the correlation between items in someone's cart and is not a robust means of understanding causality in purchase behavior.")

st.header(":wave: Welcome to Ben's Market Basket Analysis App!")
st.info("Real World Adaptation: This tool is a modified form of a tool I created in my work at Summersalt, a women's swimwear/apparel start-up. I've adapted it to use a publicly available grocery shopping dataset to ensure the privacy of their data.")
st.write("""You can use this tool to find relationships in our customers' purchasing behavior. For example, assuming someone buys milk, how likely are they also to buy eggs?""")
## The Grocery data is a row-wise collection of transactions, where each item has its own column
## In order to use the data, we want to read each row as a list
with open('groceries.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

## Then transform it into a one-hot encoded dataframe

encode_=mlxtend.preprocessing.TransactionEncoder()
encode_arr=encode_.fit_transform(data)

encode_df=pd.DataFrame(encode_arr, columns=encode_.columns_)

md_minsup=mlxtend.frequent_patterns.apriori(encode_df,
                                           min_support=0.01, 
                                           max_len = 2,
                                            use_colnames=True)

rules=mlxtend.frequent_patterns.association_rules(
md_minsup, metric="confidence",min_threshold=0.06,support_only=False)

rules['ant_string'] = rules['antecedents'].apply(lambda x: list(x)[0]).astype("unicode")
rules['con_string'] = rules['consequents'].apply(lambda x: list(x)[0]).astype("unicode")
rules['rule'] = rules['ant_string']+" -> "+rules['con_string']

item_filter = st.sidebar.selectbox("Please choose an Item to Analyze:",rules['ant_string'].unique())
item_df = rules[rules['ant_string']==item_filter]

st.sidebar.markdown("""### Some Important Terms
**Confidence:** The probability of a customer Purchasing Y, given they purchased X. E.G. If the confidence of the rule `Milk -> Bread` is 45% then we would expect 45% of customers buying milk to also buy bread.

**Lift:** An expression of the strength of a rule. The closer a rule's lift is to 1, the more likely it is to be a result of random chance. The greater a rule's lift is over one, the the more likely it is to be a meaningful rule.

**Conviction:** A similar measure to lift, but more demanding. A high conviction value (>1) for the rule `X -> Y` means that we believe Y to be highly dependent on X.""")

st.subheader("Top 25 Rules (by Conviction)")
st.write("The best overall rules look at all carts and find the rules which are most likely to be real associations (rather than random chance). For simplicity, we're only considering rules with one antecedent and one consequent, so only rules like `X -> Y` and not like `[X,Y] -> Z` or `Z -> [X,Y]`.")
rules.sort_values(by = ['conviction'], ascending=False, inplace=True)

st.write(rules[rules['lift']>1][['rule','confidence','lift','conviction']].head(25))

fig = px.bar(rules.head(10).sort_values(by ='confidence',ascending=False), x='rule', y='confidence', 
            text_auto=".1%", title="Confidence of Top 10 Rules <br><sup>Confidence: When someone buys X, what percent of the time do they buy Y with it? (rule: X -> Y)</sup>")
st.plotly_chart(fig)

st.subheader("Single Item Analysis")
st.write("Use the filter in the side-bar to choose a product to analyze.")

item_df.sort_values(by = 'confidence',ascending=False, inplace = True )
item_df.reset_index(inplace = True)

c1, c2, c3 = st.columns(3)

st.write(item_df[item_df['lift']>1][['rule','confidence','lift','conviction']])

c1.metric(label= item_df['con_string'][0], value=str(round(item_df['confidence'][0]*100,2))+"%")
c1.caption("<b> "+str(round(item_df['confidence'][0]*100,2))+"%</b>"+" of customers who bought "+item_df['ant_string'][0]+" also bought <b>"+item_df['con_string'][0], unsafe_allow_html=True)

c2.metric(label= item_df['con_string'][1], value=str(round(item_df['confidence'][1]*100,2))+"%")
c2.caption("<b> "+str(round(item_df['confidence'][1]*100,2))+"%</b>"+" of customers who bought "+item_df['ant_string'][1]+" also bought <b>"+item_df['con_string'][1], unsafe_allow_html=True)

c3.metric(label= item_df['con_string'][2], value=str(round(item_df['confidence'][2]*100,2))+"%")
c3.caption("<b> "+str(round(item_df['confidence'][2]*100,2))+"%</b>"+" of customers who bought "+item_df['ant_string'][2]+" also bought <b>"+item_df['con_string'][2], unsafe_allow_html=True)