import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
file_path = '/Users/chandukamma/Downloads/e555babb-3e1e-47b0-86c1-b0053de4a691.csv'
df = pd.read_csv(file_path)

# Display the first few rows and columns
print(df.head())
print(df.columns)

# Handle missing values with forward fill method
df.ffill(inplace=True)

# Visualize Data
# Histogram of 'Value' (assuming it represents sugar availability per day)
plt.figure(figsize=(10, 6))
sns.histplot(df['Value'], bins=30, kde=True)
plt.title('Distribution of Sugar Availability (g/day)')
plt.xlabel('Sugar Availability (g/day)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot between 'Period' and 'Value'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Period', y='Value')
plt.title('Period vs Sugar Availability (g/day)')
plt.xlabel('Period')
plt.ylabel('Sugar Availability (g/day)')
plt.show()

# Clustering
# Select features for clustering
features = ['Period', 'Value']
X = df[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Period', y='Value', hue='Cluster', palette='viridis')
plt.title('Clusters of Sugar Availability Data')
plt.xlabel('Period')
plt.ylabel('Sugar Availability (g/day)')
plt.show()

# Association Rule Mining
# Create a binary matrix for items
df_bin = df.pivot_table(index='Location', columns='Period', values='Value', aggfunc='mean').fillna(0)
df_bin = df_bin.applymap(lambda x: 1 if x > 0 else 0)

# Apply Apriori algorithm
frequent_itemsets = apriori(df_bin, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)

# Display rules
print(rules)

# Bar plot for 'Location' vs 'Value'
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Location', y='Value', estimator=sum)
plt.title('Total Sugar Availability by Location')
plt.xlabel('Location')
plt.ylabel('Sugar Availability (g/day)')
plt.xticks(rotation=90)
plt.show()

