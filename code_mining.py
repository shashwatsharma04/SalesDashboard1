#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, accuracy_score, precision_score, recall_score, f1_score, pairwise_distances_argmin_min
from scipy.stats import zscore, shapiro, ttest_ind, probplot
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
import statsmodels.api as sm
from wordcloud import WordCloud
import plotly.express as px
from memory_profiler import profile
import time
from collections import Counter
import networkx as nx
from sklearn.tree import DecisionTreeRegressor
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score
from pyECLAT import ECLAT
from pymining import seqmining
from prefixspan import PrefixSpan
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

# In[2]:


df = pd.read_csv(r"C:\Users\hp\Desktop\sem5\DM\proj\supermarket_sales - Sheet1.csv")

# In[3]:


viz_df = df.copy()
ap_df = df.copy()
df5 = df.copy()
clustering_df = df.copy()

#             1) DATA CLEANING, PREPROCESSING AND EDA
#         
#                     1.1) EDA

# In[4]:


df.head()

# In[5]:


df.columns

# In[6]:


df.info()       # Some general information about our data.

# In[7]:


df.describe()

# In[8]:


print(df.isnull().sum())        # Checking NaN values

# In[9]:


print(df.duplicated().sum())        # Checking redundancy in the dataset

# In[10]:


numeric_columns = df.select_dtypes(include=['number']).columns
z_scores_df = df[numeric_columns].apply(zscore)
threshold = 3
outliers_df = df[(abs(z_scores_df) > threshold).any(axis=1)]

print("Outliers:")
outliers_df

# There are no outliers.

# In[11]:


# Visualizing the distribution of numerical features:
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols].hist(bins=20, figsize=(8, 8))
plt.suptitle('Histograms of Numerical Features')
plt.show()

# In[12]:


# Visualize the correlation matrix
corr_matrix = df[num_cols].corr()
corr_matrix

# In[13]:


# Customer gender distribution across product lines:
plt.figure(figsize=(12, 2))  
sns.countplot(x='Product line', hue='Gender', data=df)
plt.title('Customer Gender Distribution across Product Lines')
plt.xlabel('Product Line')
plt.ylabel('Count')
plt.show()

# In[14]:


# Time-based analysis:
df['Date'] = pd.to_datetime(df['Date'])
time_analysis = df.groupby('Date').agg({'Total': 'sum', 'Quantity': 'sum'}).reset_index()
fig = px.line(time_analysis, x='Date', y=['Total', 'Quantity'], title='Time-based Analysis', labels={'value': 'Amount'})
fig.update_layout(width=800, height=300)
fig.show()

# In[15]:


# Payment method distribution:
plt.figure(figsize=(6, 3))  
df['Payment'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Payment Method Distribution')
plt.show()

# In[16]:


branch_analysis = df.groupby('Branch').agg({'Total': 'sum', 'Rating': 'mean'}).reset_index()
fig, ax1 = plt.subplots()

ax1.bar(branch_analysis['Branch'], branch_analysis['Total'], color='b', alpha=0.7, label='Total Sales')
ax1.set_xlabel('Branch')
ax1.set_ylabel('Total Sales', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(branch_analysis['Branch'], branch_analysis['Rating'], color='r', label='Average Rating')
ax2.set_ylabel('Average Rating', color='r')
ax2.tick_params('y', colors='r')

plt.title('Branch-wise Analysis')
plt.show()

# In[17]:


# Interactive Visualizations:
fig = px.scatter(df, x='Quantity', y='Total', color='Branch', hover_data=['Product line'], title='Scatter Plot')
fig.update_layout(width=500, height=450)
fig.show()

# In[18]:


# Customer Type Analysis
plt.figure(figsize=(5, 2))
sns.boxplot(x='Customer type', y='Total', data=df)
plt.title('Customer Type Analysis')
plt.xlabel('Customer Type')
plt.ylabel('Total Sales')
plt.show()

# In[19]:


# Word Cloud for Product Line Names:
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Product line']))
plt.figure(figsize=(8, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#                     1.2) Data Cleaning

# In[20]:


df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Hour'] = df['Time'].dt.hour

# In[21]:


df = df.drop(['Invoice ID', 'Date', 'Year', 'Time'], axis=1)

# Since there are no NULL values and no duplicates, as seen in the EDA, we can skip those steps.

# In[22]:


apriori_df = df.copy()
df.head()

# In[23]:


df[['Month','Day','Hour']].head()

#                     1.3) Encoding

# In[24]:


print('DISTINCT VALUES BY COLUMN:\n--------------------------')
for column in df.columns:
    unique_values = df[column].nunique()
    print(f"{column}: {unique_values}")

# In[25]:


distinct_value_counts = [df[column].nunique() for column in df.columns]

plt.figure(figsize=(15, 5))
bars = plt.bar(df.columns, distinct_value_counts, color='skyblue')
plt.title('Distinct Values Count for Each Column')
plt.xlabel('Columns')
plt.ylabel('Distinct Values Count')
plt.xticks(rotation=45, ha='right') 

# Adding text annotations on each bar
for bar, count in zip(bars, distinct_value_counts):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 0.1, str(count), ha='center', color='black')

plt.tight_layout()

plt.show()

# As we can see, 'gross market percentage' feature has a single value throughout the dataset. Hence, it is of no use to us, so I'll drop it:

# In[26]:


df = df.drop(['gross margin percentage'], axis=1)

# In[27]:


num_cols = df.select_dtypes(include=['float64', 'int64']).columns  # Numeric Cols
num_cols_with_rating = num_cols.union(pd.Index(['Rating', 'Day', 'Month', 'Hour']))

corr_matrix = df[num_cols_with_rating].corr().loc[['Total'], num_cols_with_rating]
corr_matrix

# In[28]:


df.head()

# -> Using One-Hot Encoding Encoding for the columns 'Customer Type', 'Day', 'Month' and 'Hour'.                                                                                    
# -> Using Label Encoding for the columns 'Day', 'Month' and 'Hour'

# In[29]:


ordinal_columns = ['Customer type', 'Day', 'Month', 'Hour']  # Add other ordinal columns here
nominal_columns = ['Branch', 'City', 'Gender', 'Product line', 'Payment']  # Add other nominal columns here

# Create a copy of the original DataFrame
encoded_df = df.copy()

# Convert ordinal columns using Label Encoding for both DataFrames
label_encoder = LabelEncoder()
for col in ordinal_columns:
    encoded_df[col] = label_encoder.fit_transform(encoded_df[col])

# Convert nominal columns using One-Hot Encoding for the new DataFrame
encoded_df = pd.get_dummies(encoded_df, columns=nominal_columns, drop_first=True)

# Convert boolean columns to 0s and 1s
encoded_df = encoded_df.astype(int)

# Display the encoded DataFrame
encoded_df.head()

# In[30]:


outlier_df = encoded_df.copy()
df2_1 = outlier_df.copy()

#                     1.4) Standardization

# In[31]:


sns.histplot(df, kde=True)
plt.show()

# In[32]:


sm.qqplot(encoded_df, line='s')
plt.show()

# In[33]:


stat, p_value = shapiro(encoded_df)

# Check the p-value
print(f"Shapiro-Wilk Test: Statistic={stat}, p-value={p_value}")
if p_value > 0.05:
    print("The data appears to be normally distributed.")
else:
    print("The data does not appear to be normally distributed.")

# In[34]:


print("N:", len(encoded_df))

# In[35]:


df_standardized = encoded_df.copy()
cols_to_standardize = df_standardized.columns.difference(['Total'])
scaler = StandardScaler()
df_standardized[cols_to_standardize] = scaler.fit_transform(df_standardized[cols_to_standardize])

df_standardized.head()

# In[36]:


scaler.mean_

# In[37]:


scaler.scale_

# In[38]:


np.random.seed(42)
examples_df = df_standardized.sample(n=5, random_state=42)
standardized_df = df_standardized.drop(examples_df.index)

# In[39]:


df_standardized2 = df_standardized.copy()
df_classif = df_standardized.copy()

#                     1.5) Feature Selection

# In[40]:


X = df_standardized2.drop(['Total'], axis=1)  
y = df_standardized2['Total'] 
pca = PCA(n_components='mle', svd_solver='full')
X_pca = pca.fit_transform(X)

# Scree plot:
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.8)
plt.title('Explained Variance Ratio')
plt.xlabel('Principal Component')
plt.ylabel('Ratio')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-', color='b')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance')
plt.grid(True)

plt.tight_layout()
plt.show()

# In[41]:


eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

plt.figure(figsize=(10, 3))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.xlabel('Principal Component Index')
plt.ylabel('Eigenvalue')
plt.title('Scree Plot for Eigenvalues')
plt.show()

# In[42]:


n_top_components = 15  # Choose the number of top components to visualize

top_components = pd.DataFrame(eigenvectors[:n_top_components, :], columns=X.columns)
plt.figure(figsize=(20, 5))
sns.heatmap(top_components, annot=True, cmap='coolwarm', fmt='.2f')
plt.xlabel('Original Features')
plt.ylabel('Principal Components')
plt.title(f'Top {n_top_components} Principal Components Visualization')
plt.show()

# In[43]:


feature_importance = np.abs(eigenvectors[:n_top_components, :]).sum(axis=0)

plt.figure(figsize=(15, 3))
plt.bar(X.columns, feature_importance)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Feature Importance Plot based on Top Principal Components')
plt.xticks(rotation=45, ha='right')
plt.show()

#                     1.6) Splitting into Training and Testing Sets

# In[44]:


X = df_standardized.drop('Total', axis=1)
y = df_standardized['Total']

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the resulting sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#             2) VISUALIZATION

# In[45]:


df2 = viz_df.copy()

# In[46]:


num_cols = df2.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df2[num_cols].corr()
corr_matrix = corr_matrix.drop('gross margin percentage', axis=1)
corr_matrix = corr_matrix.drop('gross margin percentage', axis=0)
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

# In[47]:


# Box Plots
plt.figure(figsize=(6, 4))
sns.boxplot(x="Branch", y="Total", data=df2)
plt.show()

# In[48]:


# Time Series Plots
df2['Date'] = pd.to_datetime(df2['Date'])
time_series_df = df2.groupby('Date')['Total'].sum().reset_index()
plt.figure(figsize=(8, 3))
plt.plot(time_series_df['Date'], time_series_df['Total'])
plt.title('Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()

# In[49]:


# Scatter Plots
plt.figure(figsize=(10, 4))
sns.scatterplot(x="Unit price", y="Total", data=df2)
plt.show()

# In[50]:


features = df2[['Total', 'Quantity', 'Rating']]

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
df2['Cluster'] = kmeans.fit_predict(features_standardized)

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

categories = features.columns
num_vars = len(categories)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

values = features.mean().tolist()
values += values[:1]
angles += angles[:1]

ax.plot(angles, values, linewidth=2, linestyle='solid', label='Average')
ax.fill(angles, values, alpha=0.4)

for cluster in range(3):  # Assuming 3 clusters
    values_cluster = features[df2['Cluster'] == cluster].mean().tolist()
    values_cluster += values_cluster[:1]
    ax.plot(angles, values_cluster, linewidth=2, linestyle='solid', label=f'Cluster {cluster}')
    ax.fill(angles, values_cluster, alpha=0.4)

ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.show()

# In[51]:


G = nx.Graph()
pairs = zip(df2['Product line'][:-1], df2['Product line'][1:])
G.add_edges_from(pairs)

plt.figure(figsize=(10, 6))
nx.draw(G, with_labels=True, font_size=10, node_size=800, font_color='black', font_weight='bold')
plt.title('Product Affinity Network Graph')
plt.show()

# In[52]:


sns.set(style="whitegrid")
numerical_vars = df2.select_dtypes(include=['float64', 'int64']).columns
df2[numerical_vars].hist(figsize=(8, 8), bins=20)
plt.show()

# In[53]:


categorical_vars = df2.select_dtypes(include='object').columns
categorical_vars = categorical_vars.drop('Invoice ID')
categorical_vars = categorical_vars.drop('Time')
plt.figure(figsize=(15, 6))

for i, var in enumerate(categorical_vars, 1):
    plt.subplot(2, 3, i)
    sns.countplot(x=var, data=df2)

plt.tight_layout()
plt.show()

#             3) ASSOCIATION RULE MINING

# Preparing data to apply Apriori

# In[54]:


apriori_df = apriori_df.drop(['gross margin percentage'], axis=1)
apriori_df.head()

# In[55]:


exploded_list = apriori_df['Product line'].str.split(' and ').explode().tolist()
print(exploded_list)

# In[56]:


# Check item frequency
item_counts = Counter(apriori_df["Product line"])
print("\nItem Frequencies:")
for item, count in item_counts.items():
    print(f"{item}: {count}")

# In[57]:


apriori_df = pd.DataFrame(exploded_list, columns=['Product'])
apriori_df['Count'] = apriori_df.groupby('Product')['Product'].transform('count')
apriori_df = apriori_df.drop_duplicates().reset_index(drop=True)
apriori_df

# In[58]:


transactions_list = apriori_df.groupby('Count')['Product'].agg(list).reset_index(name='Transactions')['Transactions'].tolist()
transactions_list

# Using Transaction Encoder

# In[59]:


tx = TransactionEncoder()
encoded_array = tx.fit(transactions_list).transform(transactions_list)
encoded_array

# In[60]:


encoded_df = pd.DataFrame(encoded_array, columns=tx.columns_, dtype=int)
encoded_df

#             Applying Apriori using MinSup 1%.

# In[61]:


frequent_itemsets = apriori(encoded_df, min_support=0.01, use_colnames=True)
frequent_itemsets

# If we were to Apply Apriori directly:

# In[62]:


dataset = ap_df.groupby("Invoice ID")["Product line"].apply(list).tolist()
te = TransactionEncoder()
onehot = te.fit_transform(dataset)
df_encoded = pd.DataFrame(onehot, columns=te.columns_)

frq_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

print("Frequent Itemsets:")
frq_itemsets

#             Forming Association Rules:

# In[63]:


rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.01)
print("\nAssociation Rules:")
rules

#             4) Comparative Analysis with FP Growth Algorithm

# In[64]:


frequent_itemsets_fp = fpgrowth(encoded_df, min_support=0.01, use_colnames=True)
frequent_itemsets_fp

# In[65]:


association_rules_fp = association_rules(frequent_itemsets_fp, metric='confidence', min_threshold=0.1)
print("Association Rules (FP-Growth):")
association_rules_fp

#             Comparitive Analysis:

# Apriori:

# In[66]:


start_time_apriori = time.time()
frequent_itemsets_apriori = apriori(encoded_df, min_support=0.01, use_colnames=True)
rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.01)
end_time_apriori = time.time()
apriori1 = end_time_apriori - start_time_apriori
print(apriori1)

# In[67]:


start_time_apriori = time.time()
frequent_itemsets_apriori = apriori(encoded_df, min_support=0.01, use_colnames=True)
end_time_apriori = time.time()
apriori2 = end_time_apriori - start_time_apriori
print(apriori2)

# In[68]:


start_time_apriori = time.time()
rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.01)
end_time_apriori = time.time()
apriori3 = end_time_apriori - start_time_apriori
print(apriori3)

# FP Growth:

# In[69]:


start_time_fp = time.time()
frequent_itemsets_fp = fpgrowth(encoded_df, min_support=0.01, use_colnames=True)
rules_fp = association_rules(frequent_itemsets_fp, metric='confidence', min_threshold=0.1)
end_time_fp = time.time()
fp1 = end_time_fp - start_time_fp
print(fp1)

# In[70]:


start_time_fp = time.time()
frequent_itemsets_fp = fpgrowth(encoded_df, min_support=0.01, use_colnames=True)
end_time_fp = time.time()
fp2 = end_time_fp - start_time_fp
print(fp2)

# In[71]:


start_time_fp = time.time()
rules_fp = association_rules(frequent_itemsets_fp, metric='confidence', min_threshold=0.1)
end_time_fp = time.time()
fp3 = end_time_fp - start_time_fp
print(fp3)

# In[72]:


execution_times_df = pd.DataFrame(columns=['Algorithm', 'Total', 'Itemsets', 'Rules'])
apriori_df = pd.DataFrame({'Algorithm': ['Apriori'], 'Total': [apriori1], 'Itemsets': [apriori2], 'Rules': [apriori3]})
fp_growth_df = pd.DataFrame({'Algorithm': ['FP Growth'], 'Total': [fp1], 'Itemsets': [fp2], 'Rules': [fp3]})
execution_times_df = pd.concat([execution_times_df, apriori_df, fp_growth_df], ignore_index=True)
execution_times_df

# In[73]:


# Extract data from the DataFrame
algorithms = execution_times_df['Algorithm']
total_execution_times = execution_times_df['Total']
itemsets_times = execution_times_df['Itemsets']
rules_times = execution_times_df['Rules']

# Set up bar positions
bar_width = 0.2
index = np.arange(len(algorithms))

# Create grouped bar chart
plt.bar(index, total_execution_times, width=bar_width, label='Total Time', color='blue')
plt.bar(index + bar_width, itemsets_times, width=bar_width, label='Itemsets Time', color='green')
plt.bar(index + 2 * bar_width, rules_times, width=bar_width, label='Rules Time', color='orange')

# Set up labels and title
plt.xlabel('Algorithm')
plt.ylabel('Time (seconds)')
plt.title('Time Comparison for Apriori and FP-Growth')
plt.xticks(index + bar_width, algorithms)
plt.legend()

# Show the plot
plt.show()


# The following block of code does not run here, I used a .py script to execute it and get the requried info.

# In[74]:


def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Time'].dt.hour

    df = df.drop(['Invoice ID', 'Date', 'Year', 'Time'], axis=1)

    return df


def prepare_apriori_data(df):
    apriori_df = df.copy()
    apriori_df = apriori_df.drop(['gross margin percentage'], axis=1)
    exploded_list = apriori_df['Product line'].str.split(
        ' and ').explode().tolist()
    apriori_df = pd.DataFrame(exploded_list, columns=['Product'])
    apriori_df['Count'] = apriori_df.groupby(
        'Product')['Product'].transform('count')
    apriori_df = apriori_df.drop_duplicates().reset_index(drop=True)
    transactions_list = apriori_df.groupby('Count')['Product'].agg(
        list).reset_index(name='Transactions')['Transactions'].tolist()
    tx = TransactionEncoder()
    encoded_array = tx.fit(transactions_list).transform(transactions_list)
    encoded_df = pd.DataFrame(encoded_array, columns=tx.columns_, dtype=int)
    return encoded_df


@profile
def apriori_algorithm(encoded_df):
    start_mem_apriori = time.time()
    frequent_itemsets_apriori = apriori(
        encoded_df, min_support=0.01, use_colnames=True)
    rules_apriori = association_rules(
        frequent_itemsets_apriori, metric="confidence", min_threshold=0.01)
    end_mem_apriori = time.time()
    apriori1_mem = end_mem_apriori - start_mem_apriori
    print(apriori1_mem)


@profile
def fp_growth_algorithm(encoded_df):
    start_time_fp_mem = time.time()
    frequent_itemsets_fp = fpgrowth(
        encoded_df, min_support=0.01, use_colnames=True)
    rules_fp = association_rules(
        frequent_itemsets_fp, metric='confidence', min_threshold=0.1)
    end_time_fp_mem = time.time()
    fp1_mem = end_time_fp_mem - start_time_fp_mem
    print(fp1_mem)


if __name__ == "__main__":
    df = pd.read_csv(
        r"C:\Users\hp\Desktop\sem5\DM\proj\supermarket_sales - Sheet1.csv")
    df = preprocess_data(df)

    apriori_df = prepare_apriori_data(df)

    apriori_algorithm(apriori_df)
    fp_growth_algorithm(apriori_df)

#             5) Tracking Patterns and Customer Behavior Analysis

# In[75]:


df5.head()

# Creating Time Series Data:

# In[76]:


df5['Date'] = pd.to_datetime(df5['Date'])
df5['Time'] = pd.to_datetime(df5['Time'], format='%H:%M')

df5['Year'] = df5['Date'].dt.year
df5['Month'] = df5['Date'].dt.month
df5['Day'] = df5['Date'].dt.day
df5['Hour'] = df5['Time'].dt.hour
df5['Minute'] = df5['Time'].dt.minute

df5 = df5.drop(['Date', 'Time', 'gross margin percentage'], axis=1)
df5.head()

# In[77]:


df5['Timestamp'] = pd.to_datetime(df5[['Year', 'Month', 'Day', 'Hour', 'Minute']])
df5.set_index('Timestamp')
df5.head()

# In[78]:


df5['Timestamp'].head()

# In[79]:


monthly_stats = df5.groupby(['Year', 'Month']).agg({'Quantity': 'mean', 'Total': 'sum', 'Rating': 'mean'})
plt.figure(figsize=(5, 3))
monthly_stats['Total'].plot(kind='bar', color='blue')
plt.title('Total Sales Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Total Sales')
plt.show()

# In[80]:


plt.figure(figsize=(12, 6))
plt.plot(df5.index, df5['Total'], marker='o')
plt.title('Total Sales Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Total Sales')
plt.show()

# In[81]:


# e. Customer Segmentation Over Time:
# Standardize the numerical columns
scaler = StandardScaler()
numerical_cols = ['Unit price', 'Quantity', 'Tax 5%', 'Total', 'cogs', 'gross income', 'Rating']
df5[numerical_cols] = scaler.fit_transform(df5[numerical_cols])

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
df5[['PCA1', 'PCA2']] = pca.fit_transform(df5[numerical_cols])

# Cluster customers based on purchasing behavior
kmeans = KMeans(n_clusters=3, random_state=42)
df5['Cluster'] = kmeans.fit_predict(df5[['PCA1', 'PCA2']])

# Plot customer segmentation over time
plt.figure(figsize=(12, 6))
for cluster in df5['Cluster'].unique():
    plt.scatter(df5[df5['Cluster'] == cluster].index, df5[df5['Cluster'] == cluster]['Total'], label=f'Cluster {cluster}')

plt.title('Customer Segmentation Over Time')
plt.xlabel('Time')
plt.ylabel('Total Sales')
plt.legend()
plt.show()

# In[82]:


# Seasonal Decomposition
sns.set(style="whitegrid")
result = seasonal_decompose(df5['Total'], model='additive', period=12)
plt.figure(figsize=(15, 3))
result.plot()
plt.show()

# Autocorrelation
plot_acf(df5['Total'])
plt.show()


# In[83]:


# Fit ARIMA model
model = ARIMA(df5['Total'], order=(1, 1, 1))
result = model.fit()

# Generate forecast for the next 5 time points
forecast_steps = 5
forecast = result.get_forecast(steps=forecast_steps).predicted_mean

# Visualize the results
plt.figure(figsize=(8, 4))
plt.plot(df5.index, df5['Total'], label='Actual', marker='o')
plt.plot(forecast.index, forecast, label='Forecast', linestyle='dashed', color='red', marker='o')
plt.title('ARIMA Forecasting')
plt.xlabel('Timestamp')
plt.ylabel('Total Sales')
plt.legend()
plt.show()

# In[84]:


df5.head()

# Feature Engineering to calculate and visualize Rolling Mean: calculation of the mean over a specified window size for the 'Total' sales column.

# In[85]:


# Rolling Statistics
df5['Total_Rolling_Mean'] = df5['Total'].rolling(window=3).mean()

plt.figure(figsize=(8, 4))
plt.plot(df5.index, df5['Total'], label='Original Total', marker='o')
plt.plot(df5.index, df5['Total_Rolling_Mean'], label='Total Rolling Mean', linestyle='dashed', marker='o')
plt.title('Original Total and Rolling Mean')
plt.xlabel('Timestamp')
plt.ylabel('Total Sales')
plt.legend()
plt.show()

#             6) SEQUENTIAL PATTERN MINING

# In[86]:


transactions_list

#             Prefix Span

# In[87]:


patterns = PrefixSpan(transactions_list).frequent(0.1, closed=True)

for pattern in patterns:
    print(pattern)

#             GSP: Generalized Sequential Patterns

# In[88]:


freq_seqs = seqmining.freq_seq_enum(transactions_list, 0.1)
for seq in freq_seqs:
    print(seq)

#             Eclat Algorithm

# In[89]:


# Assuming transactions_list is a list of lists
df_transactions = pd.DataFrame(transactions_list)

eclat_instance = ECLAT(df_transactions, verbose=True)
frequent_itemsets = eclat_instance.fit(min_support=0.1)
for itemset in frequent_itemsets:
    print(itemset)

#             7) CLUSTERING       

# In[90]:


clustering_df.head()

# In[91]:


# Assuming clustering_df is your DataFrame
features = ['Unit price', 'Quantity', 'Tax 5%', 'Total', 'cogs', 'gross income', 'Rating']

# Select relevant features for clustering
data_for_clustering = clustering_df[features]

#             K-Means

# In[92]:


scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_for_clustering)

kmeans = KMeans(n_clusters=3)
clustering_df['KMeans_Cluster'] = kmeans.fit_predict(scaled_features)

inertia = kmeans.inertia_
print("Inertia (K-Means):", inertia)
silhouette_avg = silhouette_score(scaled_features, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)
db_index = davies_bouldin_score(scaled_features, kmeans.labels_)
print("Davies-Bouldin Index:", db_index)
ch_index = calinski_harabasz_score(scaled_features, kmeans.labels_)
print("Calinski-Harabasz Index:", ch_index)

# In[93]:


kmeans = KMeans(n_clusters=2)
kmeans_labels = kmeans.fit_predict(scaled_features)

plt.figure(figsize=(4, 2))
sns.scatterplot(x='Total', y='Rating', hue=kmeans_labels, data=df, legend=False)
plt.title('K-Means Clustering')
plt.show()

# In[94]:


linkage_matrix = linkage(scaled_features, method='ward')
plt.figure(figsize=(8, 6))
dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
plt.title('K-Means Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

#             K-Median

# In[98]:


# Assuming scaled_features is your standardized data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_for_clustering)

# Perform K-Median clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(scaled_features)

# Find medians of each cluster
medians_idx = pairwise_distances_argmin_min(kmeans.cluster_centers_, scaled_features)[0]
medians = scaled_features[medians_idx]

# Apply PCA for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)

# Visualize the clusters
plt.figure(figsize=(10, 3))

for cluster_id, median_point in enumerate(medians):
    cluster_points = reduced_features[kmeans.labels_ == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}', s=50)

    # Plot median as an 'X'
    plt.scatter(median_point[0], median_point[1], marker='x', s=100, c='black')

plt.title('K-Median Clustering (PCA Visualization)')
plt.legend()
plt.show()


#             8) CLASSIFICATION

# I decided to choose the column 'Total' as the target variable.

# In[103]:


df_classif.head()

#                     8.1) Binning

# In[104]:


# Square Root Rule:

num_bins1 = int(np.sqrt(len(df_classif)))
num_bins1

# In[105]:


# Sturges' Formula:

num_bins2 = int(np.ceil(1 + np.log2(len(df_classif))))
num_bins2

# In[106]:


# Scott's Rule:

bin_width = 3.5 * df_classif['Total'].std() / len(df_classif['Total'])**(1/3)
num_bins3 = int((df_classif['Total'].max() - df_classif['Total'].min()) / bin_width)
num_bins3

# In[107]:


# Freedman-Diaconis Rule:

df_c = df_classif.copy()
df_c['Total'] = pd.to_numeric(df_c['Total'], errors='coerce')
iqr = np.percentile(df_c['Total'], 75) - np.percentile(df_c['Total'], 25)
bin_width = 2 * iqr / len(df_c['Total'])**(1/3)
num_bins4 = int((df_c['Total'].max() - df_c['Total'].min()) / bin_width)
num_bins4

# In[108]:


# Doane's Formula:

from scipy.stats import skew
num_bins5 = int(1 + np.log2(len(df_classif['Total'])) + np.log2(1 + abs(skew(df_classif['Total']) / (6/np.sqrt(len(df_classif['Total']))))))
num_bins5

# In[109]:


bins_data = {
    'Method': ['Square Root Rule', 'Sturges\' Formula', 'Scott\'s Rule', 'Freedman-Diaconis Rule', 'Doane\'s Formula'],
    'Number of Bins': [num_bins1, num_bins2, num_bins3, num_bins4, num_bins5]
}

bins_df = pd.DataFrame(bins_data)
bins_df

# In[110]:


selected_num_bins = [3, 5, 7, 9, 11, 13]

plt.figure(figsize=(20, 3))

for i, num_bins in enumerate(selected_num_bins, start=1):
    plt.subplot(1, len(selected_num_bins), i)
    plt.hist(df_classif['Rating'], bins=num_bins, edgecolor='black')
    plt.title(f'Number of Bins: {num_bins}')
    plt.xlabel('Total')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# In[111]:


column_name = 'Total'
value_counts = df_classif[column_name].value_counts()
variance = df_classif[column_name].var()

print(f"Distinct values and their frequencies in {column_name}:\n{value_counts}\n")
print(f"Variance of {column_name}: {variance}")

# In[112]:


# Assuming df_classif is your DataFrame
sns.set(style="whitegrid")

# Plot histogram
plt.figure(figsize=(8, 2))
sns.histplot(df_classif['Total'], kde=True, bins=30, color='blue')
plt.title('Histogram of Total')
plt.xlabel('Total Sales')
plt.ylabel('Frequency')
plt.show()

# In[113]:


# Q-Q plot
plt.figure(figsize=(6, 4))
probplot(df_classif['Total'], plot=plt)
plt.title('Q-Q Plot of Total')
plt.show()

# In[114]:


column_name = 'Total'
percentile_33 = df_classif[column_name].quantile(0.33)
percentile_66 = df_classif[column_name].quantile(0.66)

# Define bin edges
bin_edges = [df_classif[column_name].min(), percentile_33, percentile_66, df_classif[column_name].max()]
print("Bin Edges:")
print(bin_edges)

# In[115]:


# Assuming df_classif is your DataFrame and 'Total' is the name of the column
column_name = 'Total'

# Calculate bin edges based on percentiles
percentile_33 = df_classif[column_name].quantile(0.33)
percentile_66 = df_classif[column_name].quantile(0.66)

# Define bin edges
bin_edges = [df_classif[column_name].min(), percentile_33, percentile_66, df_classif[column_name].max()]

# Define bin labels (optional)
bin_labels = ['Low', 'Medium', 'High']

# Apply binning using cut
df_classif['Total_bins'] = pd.cut(df_classif[column_name], bins=bin_edges, labels=bin_labels, include_lowest=True)
df_classif['Total'] = df_classif['Total_bins']
df_classif.drop('Total_bins', axis=1, inplace=True)
df_classif['Total']


# In[116]:


# Assuming df_classif is your DataFrame
sns.set(style="whitegrid")

# Plot histogram
plt.figure(figsize=(8, 4))
sns.histplot(df_classif['Total'], kde=True, bins=30, color='blue')
plt.title('Histogram of Total')
plt.xlabel('Total Sales')
plt.ylabel('Frequency')
plt.show()

# In[117]:


# Assuming 'Total' is the target column
new_rating_counts = df_classif['Total'].value_counts().sort_index()

# Display the counts for each discrete value
new_rating_counts

# In[118]:


new_rating_counts = df_classif['Total'].value_counts().sort_index()

plt.figure(figsize=(4,4))
plt.pie(new_rating_counts, labels=new_rating_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen', 'lightcoral'])
plt.title('Distribution of Binned Total Values')
plt.show()

# In[119]:


df_anova = df_classif.copy()

#                     8.2) Splitting into Training and Testing Sets

# In[120]:


Xc = df_classif.drop('Total', axis=1)
yc = df_classif['Total']
classif_X_train, classif_X_test, classif_y_train, classif_y_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)

print("Training set shape:", classif_X_train.shape, classif_y_train.shape)
print("Testing set shape:", classif_X_test.shape, classif_y_test.shape)

#                     8.3) Model Training and Testing

# In[121]:


classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(random_state=42)
}

evaluation_results = []

pca = PCA(n_components=15)
X_train_pca = pca.fit_transform(classif_X_train)
X_test_pca = pca.transform(classif_X_test)

for model_name, classifier in classifiers.items():
    classifier.fit(X_train_pca, classif_y_train)
    classif_y_pred = classifier.predict(X_test_pca)
    accuracy = accuracy_score(classif_y_test, classif_y_pred)
    precision = precision_score(classif_y_test, classif_y_pred, average='weighted')
    recall = recall_score(classif_y_test, classif_y_pred, average='weighted')
    f1 = f1_score(classif_y_test, classif_y_pred, average='weighted')

    evaluation_results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

evaluation_df = pd.DataFrame(evaluation_results)
evaluation_df

#             9) REGRESSION

# In[122]:


# Applying PCA with 15 components:
pca = PCA(n_components=15)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

results_df = pd.DataFrame(columns=['Model', 'MSE', 'RMSE', 'MAE', 'R2', 'Adjusted R2', 'Explained Variance'])

# Function to calculate adjusted R-squared:
def adjusted_r2_score(y_true, y_pred, n, p):
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2

# List of regression models to apply:
models = [LinearRegression(), Ridge(), ElasticNet(), RandomForestRegressor(), DecisionTreeRegressor()]

# Initialize results DataFrame
results_df = pd.DataFrame(columns=['Model', 'MSE', 'RMSE', 'MAE', 'R2', 'Adjusted R2', 'Explained Variance'])

# Loop through each model
for model in models:
    # Train the model
    model.fit(X_train_pca, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test_pca)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = adjusted_r2_score(y_test, y_pred, len(y_test), X_test_pca.shape[1])
    ev = explained_variance_score(y_test, y_pred)

    # Create a DataFrame with the results for the current model
    model_results = pd.DataFrame({
        'Model': [model.__class__.__name__],
        'MSE': [mse],
        'RMSE': [rmse],
        'MAE': [mae],
        'R2': [r2],
        'Adjusted R2': [adj_r2],
        'Explained Variance': [ev]
    })

    # Concatenate the current model's results to the overall results DataFrame
    results_df = pd.concat([results_df, model_results], ignore_index=True)

# Display the results
results_df

#             10) Outlier Detection & Statistical Validation

# I'm going to use encoded data, so that I'm able to apply the methods to all columns (since they are now numeric).

# In[123]:


outlier_df.head()

# In[124]:


outlier_df.info()

# In[125]:


distinct_counts = outlier_df.nunique()
print('DISTINCT COUNT:')
distinct_counts

# Analyzing the 'Total' column for Outliers:

# In[126]:


columns_of_interest = [
    'Total', 'Customer type', 'Unit price', 'Quantity', 'Tax 5%', 'Total',
    'cogs', 'gross income', 'Rating', 'Month', 'Day', 'Hour'
]

outlier_counts_df = pd.DataFrame(columns=['Threshold'] + columns_of_interest)

def detect_outliers(data, threshold):
    outliers = []
    mean = np.mean(data)
    std = np.std(data)

    for i in data:
        z_score = (i - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return len(outliers)

threshold_values = np.arange(1, 4, 0.5)  # Varying threshold from 1 to 3 in steps of 0.5

dfs_to_concat = []

for threshold in threshold_values:
    outlier_counts = [threshold]
    
    for column in columns_of_interest:
        count_outliers = detect_outliers(outlier_df[column], threshold)
        outlier_counts.append(count_outliers)

    df_temp = pd.DataFrame([outlier_counts], columns=outlier_counts_df.columns)
    dfs_to_concat.append(df_temp)

outlier_counts_df = pd.concat(dfs_to_concat, ignore_index=True)

print('Outlier Counts for Different Thresholds:')
outlier_counts_df

# In[127]:


plt.figure(figsize=(4, 3))
sns.boxplot(x=outlier_df['Total'])
plt.title('Boxplot of Total')
plt.show()

z_scores_rating = zscore(outlier_df['Total'])
threshold = 2.5
rating_outlier_indices = (z_scores_rating > threshold) | (z_scores_rating < -threshold)
print('Rows with Total outliers:')
outlier_df[rating_outlier_indices]

# In[128]:


# Change the name of 'Total' column in df_anova to 'Total_Category'
df_anova.rename(columns={'Total': 'Total_Category'}, inplace=True)

# Copy 'Total' column from outlier_df to df_anova
df_anova['Total_Category'] = outlier_df['Total']

# Now you can use df_anova in the previous ANOVA code
from scipy.stats import f_oneway

# Assuming 'df_anova' is your dataset
total_high = df_anova[df_anova['Total_Category'] == 'High']['Total_Category']
total_medium = df_anova[df_anova['Total_Category'] == 'Medium']['Total_Category']
total_low = df_anova[df_anova['Total_Category'] == 'Low']['Total_Category']

# Perform ANOVA test
f_stat, p_value = f_oneway(total_high, total_medium, total_low)

# Print the results
print(f'F-statistic: {f_stat}\nP-value: {p_value}')

# Check if the p-value is below the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The difference in Total sales among High, Medium, and Low categories is statistically significant.")
else:
    print("There is no statistically significant difference in Total sales among High, Medium, and Low categories.")

# In[129]:


# Assuming 'outlier_df' is your dataset
customer_type_normal = outlier_df[outlier_df['Customer type'] == 'Normal']['Total']
customer_type_member = outlier_df[outlier_df['Customer type'] == 'Member']['Total']

# Perform independent t-test
t_stat, p_value = ttest_ind(customer_type_normal, customer_type_member, equal_var=False)

# Print the results
print(f'T-statistic: {t_stat}\nP-value: {p_value}')

# Check if the p-value is below the significance level (e.g., 0.05)
if p_value < 0.05:
    print("The difference in Total sales between Normal and Member customers is statistically significant.")
else:
    print("There is no statistically significant difference in Total sales between Normal and Member customers.")


# In[130]:


# Visualize outliers using box plots
plt.figure(figsize=(8, 6))
sns.boxplot(data=outlier_df[['Unit price', 'Quantity', 'Tax 5%', 'Total', 'cogs', 'gross income', 'Rating']])
plt.title('Boxplot of Numeric Features')
plt.show()

# Using InterQuartile Range Method for the 'Total' column:

# In[131]:


rating_column = outlier_df['Total']

# Calculate the IQR (Interquartile Range)
Q1 = rating_column.quantile(0.25)
Q3 = rating_column.quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_df_iqr = outlier_df[(rating_column < lower_bound) | (rating_column > upper_bound)]
print('Rows with Rating outliers using IQR method:')
outliers_df_iqr

# In[132]:


# Visualize the distribution of 'Total' values with boxplot
plt.figure(figsize=(4, 2))
sns.boxplot(x=outlier_df['Total'])
plt.title('Distribution of Total Sales')
plt.xlabel('Total Sales')
plt.show()

# Visualize the identified outliers using scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=outlier_df.index, y=outlier_df['Total'], label='Normal Transactions', color='blue', legend=False)
sns.scatterplot(x=outliers_df_iqr.index, y=outliers_df_iqr['Total'], label='Outliers', color='red', legend=False)
plt.title('Outliers Detected using IQR Method')
plt.xlabel('Transaction Index')
plt.ylabel('Total Sales')
#plt.legend()
plt.show()


# In[133]:


# Separate 'Total' values for normal transactions and outliers
normal_total = outlier_df.loc[~outlier_df.index.isin(outliers_df_iqr.index), 'Total']
outlier_total = outliers_df_iqr['Total']

# Perform a two-sample t-test
t_stat, p_value = ttest_ind(normal_total, outlier_total)

# Set the significance level
alpha = 0.05

# Print the results
print(f'T-Statistic: {t_stat}')
print(f'P-Value: {p_value}')

# Compare p-value with the significance level
if p_value < alpha:
    print('The null hypothesis is rejected. There is a significant difference in total sales between normal transactions and outliers.')
else:
    print('The null hypothesis cannot be rejected. There is no significant difference in total sales between normal transactions and outliers.')

# Using Isolation Forest:

# In[134]:


# Extracting Outliers based on the following dimensions: 'Total', 'Unit price', 'cogs', and 'Quantity' columns:
X = outlier_df[['Total', 'Unit price', 'Quantity', 'cogs']]

model = IsolationForest(contamination=0.01)
outlier_df['IsOutlier'] = model.fit_predict(X)
outliers_df_if = outlier_df[outlier_df['IsOutlier'] == -1].copy()
outliers_df_if.drop(columns=['IsOutlier'], inplace=True)

print('Rows with outliers using Isolation Forest method:')
outliers_df_if

# In[135]:


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(outlier_df['Total'], outlier_df['Unit price'], outlier_df['Quantity'], c=outlier_df['IsOutlier'])
ax.set_xlabel('Total')
ax.set_ylabel('Unit price')
ax.set_zlabel('Quantity')

plt.show()
