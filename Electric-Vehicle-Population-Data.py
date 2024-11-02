#!/usr/bin/env python
# coding: utf-8

# In[18]:


get_ipython().system('pip install pandas matplotlib seaborn geopandas')
get_ipython().system('mkdir data')
get_ipython().system('wget https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD -O data/Electric_Vehicle_Population_Data.csv')


# In[19]:


import pandas as pd

df = pd.read_csv("data/Electric_Vehicle_Population_Data.csv")


df.drop_duplicates()


# In[20]:


df.isna().sum()


# In[21]:


missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})


missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False)


missing_data


# 

# In[22]:


df['Legislative District'].head(5)



# 

# ## 1) Cleaning strategy:
# 
#   - The rows where are column values are null will be removed.
# 
#   - The rows where not all columns are null will use median amputation if it is a numeric value.
# 
#   - Some columns have numeric values but are indeed encoeded categorical values (such as **Legislative district**). These will use mode imputation.
#   - Categorical columns will be mode imputated
# 

# In[23]:


rows_all_null = df.isnull().all(axis=1).sum()
print(f"Number of rows where all columns are null: {rows_all_null}")


# 
# 
# ### 1.1 Handling Missing Values for Categorical and Encoded Categorical Features
# 
# This section outlines the strategy for dealing with missing values in features that are either categorical or encoded as categorical.
# 
# **Imputation Method:**
# 
# For these features, the **mode imputation** method will be used. This means that missing values will be replaced with the most frequent value (mode) observed in the respective feature.
# 
# **Rationale:**
# 
# Categorical features represent distinct categories or groups. Directly substituting missing values with a numerical average (e.g., mean or median) wouldn't be appropriate as it wouldn't preserve the feature's categorical nature. Mode imputation, on the other hand, ensures that the substituted value aligns with the existing categories, maintaining the feature's integrity and distribution.
# 
# **Example:**
# 
# For a feature like 'Postal Code,' which is a numeric value but possesses no meaning, computing the average would contribute nothing to the analysis. On the other hand, calculating the mode gives a sensible result.
# 

# In[24]:


columns = ['Vehicle Location', 'Legislative District', 'Postal Code', 'City', 'County', 'Electric Utility']
# Get the mode for each column. If multiple modes exist, select the first one.
max_value = df[columns].mode().iloc[0]
df[columns] = df[columns].fillna(max_value)
df.isna().sum()


# 

# - Other numeric features such as **base MSRP**, imputing the missing values by the **median** does not cause wrong biasing of the analysis, which is why it was chosen over other methods.
# 
# - The code below imputes remaining missing values by the **median**.

# In[25]:


import pandas as pd

numeric_df = df.select_dtypes(include='number')

medians = numeric_df.median()

df[numeric_df.columns] = numeric_df.fillna(medians)

df.isna().sum()


# ## 2) Encoding Categorical Features Based on Cardinality
# 
# This approach involves encoding categorical features differently based on the number of unique categories (cardinality) they contain.
# 
# **Rationale:**
# 
# - **High Cardinality Features:**  Features with a large number of unique categories (e.g., more than 3) are often best handled with label encoding. This assigns a unique numerical label to each category. While this introduces an artificial ordinal relationship between categories, it is often preferred for high-cardinality features because one-hot encoding would create an excessive number of new columns, potentially leading to the curse of dimensionality.
# 
# - **Low Cardinality Features:** Features with a small number of unique categories (e.g., less than or equal to 3) are better suited for one-hot encoding. This creates a new binary column for each category, where a 1 indicates the presence of that category and a 0 indicates its absence. One-hot encoding effectively avoids introducing artificial ordinal relationships and is often more appropriate for low-cardinality features.
# 
# 
# **Example:**
# 
# Consider a dataset with a 'Color' feature that has values like 'Red', 'Blue', 'Green', 'Yellow', 'Orange'. Since this feature has more than 3 unique categories, label encoding would be a better choice.
# 
# - Label Encoding:
#     - Red: 0
#     - Blue: 1
#     - Green: 2
#     - Yellow: 3
#     - Orange: 4
# 
# However, if we had a 'Gender' feature with values 'Male', 'Female', we would apply one-hot encoding:
# 
# - One-Hot Encoding:
#     - Gender_Male: (1 if Male, 0 otherwise)
#     - Gender_Female: (1 if Female, 0 otherwise)

# In[26]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

df_copy = df.copy()

for column in df_copy.select_dtypes(include=['object']).columns:
  if df_copy[column].nunique() > 3:
    # Apply Label Encoding for categorical features with more than 3 unique values
    le = LabelEncoder()
    df_copy[column] = le.fit_transform(df_copy[column])
  else:
    # Apply One-Hot Encoding for categorical features with more than 3 unique values
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    feature_array = ohe.fit_transform(df_copy[[column]])
    feature_labels = [f"{column}_{class_label}" for class_label in ohe.categories_[0]]
    features_df = pd.DataFrame(feature_array, columns=feature_labels)
    df_copy = pd.concat([df_copy, features_df], axis=1)
    df_copy = df_copy.drop(column, axis=1)

df_copy.head()


# 

# ## 3) Normalization
# 
# Normalization was performed using **z-score standardization**. This involves subtracting the mean and dividing by the standard deviation for each feature, transforming the data to have a mean of 0 and a standard deviation of 1.
# 

# 

# In[27]:


from sklearn.preprocessing import StandardScaler

# Extract the 'Census Tract' column
census_tract = df_copy['2020 Census Tract']

# Create a StandardScaler object
scaler = StandardScaler()

# Reshape the data to fit the scaler
census_tract_reshaped = census_tract.values.reshape(-1, 1)

# Fit and transform the data to normalize it using Z-score
normalized_census_tract = scaler.fit_transform(census_tract_reshaped)

# Replace the original 'Census Tract' column with the normalized values
df_copy['2020 Census Tract new'] = normalized_census_tract
df_copy[['2020 Census Tract', '2020 Census Tract new']].head()


# ## 4) Descriptive Statistics
# 
# Non-numeric columns and things like **Postal Code** statistics are not of interest in this case. They are dropped to leave more space to the numeric columns so as to make the table less cluttered.

# 

# In[28]:


numeric_columns = df.select_dtypes(include='number')
columns_to_drop = ['Legislative District', 'Postal Code', 'DOL Vehicle ID']
numeric_columns.drop(columns = columns_to_drop, inplace = True)
numeric_columns.describe()


# 

# In[29]:


Models_by_count=df['Model'].value_counts()
Models_by_count


# In[30]:


Models_by_count[:20].plot(kind='bar')


# In[31]:


import pandas as pd

numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()
corr_matrix


# In[32]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[33]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd


# In[34]:


plt.figure(figsize=(10,6))

sns.histplot(df['Electric Range'],bins = 50)
plt.title("Distribution of Electric Range")
plt.xlabel("Electric Range (miles)")
plt.ylabel("Frequency")
plt.show()


# In[35]:


#print(df['Base MSRP'].describe())
car_counts_Cty = df['City'].value_counts().nlargest(10)


car_counts_Cty.plot(kind='bar')
plt.xlabel('City')
plt.ylabel('Number of Cars')
plt.title('Top 10 Count of Cars per City')
plt.show()


car_counts_cty_df = car_counts_Cty.to_frame()
car_counts_cty_df.style.background_gradient(cmap='Blues')


# In[36]:


City_df = pd.DataFrame(df.City[:20].value_counts().sort_index())
cities_by_Electric_Range=df.City.value_counts()
cities_by_Electric_Range[:20].plot(kind='pie')


# In[37]:


sns.scatterplot(data=df, x="Electric Range", y="Base MSRP")


# In[38]:


sns.scatterplot(data = df, x = 'Electric Range', y = 'Model Year')


# In[39]:


car_counts = df.groupby(['Make', 'Model']).size().reset_index(name='Count')
most_popular_car = car_counts.loc[car_counts['Count'].idxmax()]
popular_car_make = most_popular_car['Make']
popular_car_model = most_popular_car['Model']
popular_car_registrations = df[(df['Make'] == popular_car_make) & (df['Model'] == popular_car_model)]
city_counts = popular_car_registrations['City'].value_counts().reset_index(name='Count')
city_counts.columns = ['City', 'Count']
top_cities = city_counts.head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_cities, x='City', y='Count', palette='viridis')
plt.xticks(rotation=45)
plt.xlabel('City')
plt.ylabel('Number of Registrations')
plt.title(f'Top 10 Cities with Most Registrations of {popular_car_make} {popular_car_model} (Most Popular Car)')
plt.tight_layout()
plt.show()


# In[40]:


non_zero_range = df[df['Electric Range'] > 0]

sns.histplot(data=non_zero_range, x='Electric Range', hue='Electric Vehicle Type', bins=30, multiple='stack')
plt.title("Electric Range Distribution by Vehicle Type (Excluding Vehicles with zero mile electric range")
plt.xlabel("Electric Range (miles)")
plt.ylabel("Frequency")
plt.show()



# In[41]:


df_filtered = df[df['Electric Range'] > 0]

plt.figure(figsize=(12, 8))

sns.boxplot(data=df_filtered, x='Make', y='Electric Range')
plt.title("Electric Range Distribution by Make (Excluding Zero Range)")
plt.xlabel("Make")
plt.ylabel("Electric Range (miles)")
plt.xticks(rotation=90)
plt.show()


# In[42]:


df_filtered = df[df['Model Year'] >= 2010]


plt.figure(figsize=(12, 8))

# Plot the boxplot for Electric Range by Model Year
sns.boxplot(data=df_filtered, x='Model Year', y='Electric Range')
plt.title("Electric Range Distribution by Model Year (2010 and Above)")
plt.xlabel("Model Year")
plt.ylabel("Electric Range (miles)")
plt.xticks(rotation=90)
plt.show()


# In[43]:


plt.figure(figsize=(12, 6))

sns.histplot(data=df,
             x="Electric Range",
             hue="Clean Alternative Fuel Vehicle (CAFV) Eligibility",
             multiple="stack",
             linewidth=0,bins = 25)


plt.title('Electric Range Distribution by CAFV Eligibility')
plt.xlabel('Electric Range')
plt.ylabel('Count')


plt.show()


# In[44]:


city_counts = df['City'].value_counts().reset_index()
city_counts.columns = ['City', 'Count']

# Create a bar chart
plt.figure(figsize=(12, 6))
sns.barplot(data=city_counts.head(20), x='City', y='Count', palette='viridis')  # Display top 20 cities
plt.title('Number of Electric Vehicles by City')
plt.xlabel('City')
plt.ylabel('Number of EVs')
plt.xticks(rotation=45)
plt.show()


# In[45]:


county_counts = df['County'].value_counts().reset_index()
county_counts.columns = ['County', 'Count']


plt.figure(figsize=(12, 6))
sns.barplot(data=county_counts.head(20), x='County', y='Count', palette='magma')  # Display top 20 counties
plt.title('Number of Electric Vehicles by County')
plt.xlabel('County')
plt.ylabel('Number of EVs')
plt.xticks(rotation=45)
plt.show()


# In[46]:


vehicle_counts = df['State'].value_counts().reset_index()
vehicle_counts.columns = ['State', 'Vehicle Count']

vehicle_counts.style.background_gradient(cmap='Blues')


# In[47]:


get_ipython().system('pip install contextily')


# In[48]:


import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt

# Assuming df_evb['Vehicle Location'] contains WKT coordinates
data = df['Vehicle Location']
gs = gpd.GeoSeries.from_wkt(data)
gdf = gpd.GeoDataFrame(df, geometry=gs, crs="EPSG:4326")
gdf = gdf.to_crs(epsg=3857)

# Plot all vehicles in one plot
fig, ax = plt.subplots(figsize=(12, 12))

# Plot all vehicle locations, using different colors for each vehicle type
gdf.plot(
    ax=ax,
    column='Electric Vehicle Type',
    cmap='coolwarm',
    legend=True,
    legend_kwds={'loc': 'upper right'}
)

gdf.boundary.plot(ax=ax, color='black', linewidth=0.5)

# # Add state name labels (adjust parameters as needed)
# for x, y, label in zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y, gdf['State']):
#     ax.text(x, y, label, fontsize=8, ha='center', va='center')

ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)
ax.set_title('Electric Vehicles (BEV and PHEV)')

ax.set_xlim([-1.5e7, -0.7e7])  # Longitude range
ax.set_ylim([2e6, 6.5e6])    # Latitude range


plt.show()


# In[49]:


import numpy as np
df = pd.read_csv("data/Electric_Vehicle_Population_Data.csv")
years = np.arange(df['Model Year'].min(), 2024)
yearly_counts = df.groupby('Model Year').size().reindex(years, fill_value=0)


plt.figure(figsize=(12, 6))
sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o')
plt.title("Trend of Electric Vehicle Adoption by Model Year")
plt.xlabel("Model Year")
plt.ylabel("Number of Vehicles")
plt.show()

