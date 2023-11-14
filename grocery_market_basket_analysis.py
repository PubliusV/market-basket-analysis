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

## Run the apriori algorithm
md_minsup=mlxtend.frequent_patterns.apriori(encode_df,
                                           min_support=0.01, 
                                           max_len = 2,
                                            use_colnames=True)

rules=mlxtend.frequent_patterns.association_rules(
md_minsup, metric="confidence",min_threshold=0.06,support_only=False)

## Extract the rules as strings for user legibility -- MLXtend outputs the antecedent and consequent as frozenset() types
rules['ant_string'] = rules['antecedents'].apply(lambda x: list(x)[0]).astype("unicode")
rules['con_string'] = rules['consequents'].apply(lambda x: list(x)[0]).astype("unicode")
rules['rule'] = rules['ant_string']+" -> "+rules['con_string']

## Use the antecedent strings as a filter for the product-level analysis later
item_filter = st.sidebar.selectbox("Please choose an Item to Analyze:",rules['ant_string'].unique())
item_df = rules[rules['ant_string']==item_filter]

## Make some notes in the sidebar
st.sidebar.markdown("""### Some Important Terms
**Confidence:** The probability of a customer Purchasing Y, given they purchased X. E.G. If the confidence of the rule `Milk -> Bread` is 45% then we would expect 45% of customers buying milk to also buy bread.

**Lift:** An expression of the strength of a rule. The closer a rule's lift is to 1, the more likely it is to be a result of random chance. The greater a rule's lift is over one, the the more likely it is to be a meaningful rule.

**Conviction:** A similar measure to lift, but more demanding. A high conviction value (>1) for the rule `X -> Y` means that we believe Y to be highly dependent on X.""")

#############################
## Rules Overview Section  ##
#############################
st.subheader("Top 25 Rules (by Conviction)")
st.write("The best overall rules look at all carts and find the rules which are most likely to be real associations (rather than random chance). For simplicity, we're only considering rules with one antecedent and one consequent, so only rules like `X -> Y` and not like `[X,Y] -> Z` or `Z -> [X,Y]`.")

# Get organized
rules.sort_values(by = ['conviction'], ascending=False, inplace=True)

# Exclude bad rules (Lift <=1)
st.write(rules[rules['lift']>1][['rule','confidence','lift','conviction']].head(25))

# Ooh pretty picture
fig = px.bar(rules.head(10).sort_values(by ='confidence',ascending=False), x='rule', y='confidence', 
            text_auto=".1%", title="Confidence of Top 10 Rules <br><sup>Confidence: When someone buys X, what percent of the time do they buy Y with it? (rule: X -> Y)</sup>")
st.plotly_chart(fig)

#########################
## Single Item Section ##
#########################

st.subheader("Single Item Analysis")
st.write("Use the filter in the side-bar to choose a product to analyze.")

# Get organized (& reset the index so our references work)
item_df.sort_values(by = 'confidence',ascending=False, inplace = True )
item_df.reset_index(inplace = True)



# We need three columns for metrics.
# We want these to show up before the table, so we create the columns now
# and fill them after we write out the table to the page.
c1, c2, c3 = st.columns(3)

# Exclude bad rules (Lift <=1) & write to a table
st.write(item_df[item_df['lift']>1][['rule','confidence','lift','conviction']])


# Create a metric for the top 3 rules

# Build a metric in column 1:
# Put the column objects in a list for the loop
columns = [c1, c2, c3]
n_iter = 0

# initiate a loop through the columns. Use an iterator so we can reference the index of the item dataframe
for column in columns:
    column.metric(label= item_df['con_string'][n_iter], # by pulling the complement string of the top rule into the metric label
            value=str(round(item_df['confidence'][n_iter]*100,2))+"%" # And passing the formatted confidence score
            )
    # Add a caption to it, using the confidence score, the antecedent string, and the consequent string
    column.caption("<b> "+str(round(item_df['confidence'][n_iter]*100,2))+
            "%</b>"+" of customers who bought "+item_df['ant_string'][n_iter]+
            " also bought <b>"+item_df['con_string'][n_iter], unsafe_allow_html=True
            )
    n_iter+=1
