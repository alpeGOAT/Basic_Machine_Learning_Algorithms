import pandas as pd # import pandas library

# Prepare Maximum column and width lengths
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 350)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 100)

video_game = pd.read_csv('vgsales.csv')

# First 5 Rows
print(video_game.head())
print("\n")

# Last 5 Rows
print(video_game.tail())
print("\n")

# Dataset Information
video_game.info()
print("\n")

# Dataset Description
print(video_game.describe())
print("\n")

# Missing values in the Dataset
print(video_game.isnull().sum())
print("\n")

# Dataset Shape (Row amount, Column amount)
print(video_game.shape)

# Amount of Rows
print(video_game.shape[0])
# Amount of Columns
print(video_game.shape[1])
# Think it as 2D Matrix

# Print specific row
print("\n")
print(video_game.iloc[15]) # gets the 16'th row
print("\n")
print(video_game.iloc[[0,4]]) # gets first and 5'th row
print("\n")

# Display all columns
print(video_game.columns)
print("\n")

# Display types of all columns
print(video_game.dtypes)
print("\n")

categorical_columns = ['Name', 'Platform', 'Genre', 'Publisher']
numerical_columns = ['Rank', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

video_games_categorical = video_game[categorical_columns] # Dataset that contains categorical columns
video_games_numerical = video_game[numerical_columns] # Dataset that contains numerical columns

print(video_games_categorical.head())
print("\n")
print(video_games_numerical.head())
print("\n")

# Drops duplicate rows
video_game = video_game.drop_duplicates()
# Drop rows where publisher information is missing
video_game = video_game.dropna(subset=['Publisher'])
# Give median Year value to rows that has empty year value
video_game['Year'] = video_game['Year'].fillna(video_game['Year'].median())
# Make sure to convert year into int just in case
video_game['Year'] = video_game['Year'].astype(int)

# Checking Dataset shape and missing values after filtering
print(video_game.shape)
print("\n")
print(video_game.isnull().sum())
print("\n")

# Games after the year 2010
video_games_after_2010 = video_game[video_game['Year'] >= 2010]
print(video_games_after_2010.head(20))
print("\n")

# Video games where global sale is bigger than 10
video_games_global_sales_bigger_than10 = video_game[video_game['Global_Sales'] > 10]
print(video_games_global_sales_bigger_than10[['Rank','Name','Global_Sales']].head(20))
print("\n")

# Get only Sports genre
video_game_sports = video_game[video_game['Genre'] == 'Sports']
print(video_game_sports.head())
print("\n")

# Games where genre is Action and publisher is Nintendo (there aren no rows for this filter)
video_game_1 = video_game[(video_game['Genre'] == 'Action') & (video_game['Publisher'] == 'Nintendo')]
print(video_game_1)
print("\n")

# Games where genre is Shooter and publisher is Nintendo (26 rows)
video_game_2 = video_game[(video_game['Genre'] == 'Shooter') & (video_game['Publisher'] == 'Nintendo')]
print(video_game_2)
print(video_game_2.shape)
print("\n")

# Sort values by global sales (descending)
video_game_3 = video_game.sort_values(by='Global_Sales', ascending=False)
print(video_game_3.head(15))
print("\n")

# Sort values by global sales (ascending)
video_game_4 = video_game.sort_values(by='Global_Sales', ascending=True)
print(video_game_4.head(15))
print("\n")

# Sort values by year (descending)
video_game_5 = video_game.sort_values(by='Year', ascending=False)
print(video_game_5[['Name', 'Year']].head(10))
print("\n")

# Sort values by year (ascending)
video_game_6 = video_game.sort_values(by='Year', ascending=True)
print(video_game_6[['Name', 'Year']].head(10))
print("\n")

# Sort by year and then by sales (descending)
video_game_7 = video_game.sort_values(by=['Year','Global_Sales'], ascending=False)
print(video_game_7[['Name', 'Year', 'Global_Sales']].head(15))
print("\n")

# Number of unique genres
no_of_unique_genres = video_games_categorical['Genre'].nunique()
list_of_unique_genres = video_games_categorical['Genre'].unique().tolist()
print(no_of_unique_genres)
print(list_of_unique_genres)
print("\n")

# Number of unique Platforms
no_of_unique_platforms = video_games_categorical['Platform'].nunique()
list_of_unique_platforms = video_games_categorical['Platform'].unique().tolist()
print(no_of_unique_platforms)
print(list_of_unique_platforms)
print("\n")

# Number of unique Publishers
no_of_unique_publishers = video_games_categorical['Publisher'].nunique()
list_of_unique_publishers = video_games_categorical['Publisher'].unique().tolist()
print(no_of_unique_publishers)
print(list_of_unique_publishers)
print("\n")

# Percentage of each genre, add new column for it
genre_percentage = video_game['Genre'].value_counts(normalize=True) * 100
video_game['Genre_Percentage'] = video_game['Genre'].map(genre_percentage)
print(video_game[['Name', 'Genre', 'Genre_Percentage']].head(10))
print("\n")

genre_percentage_table = video_game[['Genre', 'Genre_Percentage']].drop_duplicates().sort_values(by='Genre_Percentage', ascending=False)
print(genre_percentage_table)
print("\n")

# Find percentage of each platform, add new column for it
platform_percentage = video_game['Platform'].value_counts(normalize=True) * 100
video_game['Platform_Percentage'] = video_game['Platform'].map(platform_percentage)
print(video_game[['Name', 'Platform', 'Platform_Percentage']].head(10))
print("\n")

platform_percentage_table = video_game[['Platform', 'Platform_Percentage']].drop_duplicates().sort_values(by='Platform_Percentage', ascending=False)
print(platform_percentage_table)
print("\n")

# Percentage of each publisher, add new column for it
publisher_percentage = video_game['Publisher'].value_counts(normalize=True) * 100
video_game['Publisher_Percentage'] = video_game['Publisher'].map(publisher_percentage)
print(video_game[['Name', 'Publisher', 'Publisher_Percentage']].head(18))
print("\n")

publisher_percentage_table = video_game[['Publisher', 'Publisher_Percentage']].drop_duplicates().sort_values(by='Publisher_Percentage', ascending=False)
print(publisher_percentage_table.head(80))
print("\n")

# Amount of values in columns
print(video_game['Platform'].value_counts())
print("\n")
print(video_game['Publisher'].value_counts())
print("\n")
print(video_game['Name'].value_counts())
print("\n")

# Games with lowest and highest global sales
highest_sales = video_game.sort_values(by='Global_Sales', ascending=False).head(1)
print(highest_sales[['Name', 'Global_Sales']])
print("\n")

lowest_sales = video_game.sort_values(by='Global_Sales', ascending=True).head(1)
print(lowest_sales[['Name', 'Global_Sales']])
print("\n")

# Earliest and Latest year
earliest_year = video_game.sort_values(by='Year', ascending=True).head(1)
print(earliest_year[['Name', 'Year']])
print("\n")

latest_year = video_game.sort_values(by='Year', ascending=False).head(1)
print(latest_year[['Name', 'Year']])
print("\n")

# Categorical and Numerical Datasets after data filtering
video_games_categorical1 = video_game[categorical_columns]
video_games_numerical1 = video_game[numerical_columns]

print(video_games_categorical1.head(10))
print("\n")
print(video_games_numerical1.head(10))
print("\n")

# Number of games per genre
video_game_group_by1 = video_game.groupby('Genre')['Name'].count().sort_values(ascending=False).drop_duplicates()
print(video_game_group_by1)
print("\n")

# Total amount of games published each year
video_game_group_by2 = video_game.groupby('Year')['Name'].count().sort_values(ascending=False).drop_duplicates()
print(video_game_group_by2)
print("\n")

# How much game each publisher produced
video_game_group_by3 = video_game.groupby('Publisher')['Name'].count().sort_values(ascending=False).drop_duplicates()
print(video_game_group_by3)
print("\n")

# Statistical Operations for columns (df.describe already does this but we can print it out separately)
print(video_game['NA_Sales'].mean())
print(video_game['NA_Sales'].median())
print(video_game['NA_Sales'].max(), "/", video_game['NA_Sales'].min())
print("\n")

print(video_game['Global_Sales'].mean(), "/", video_game['Global_Sales'].median())
print(video_game['Global_Sales'].max(), "/", video_game['Global_Sales'].min())
print("\n")

print(video_game['EU_Sales'].mean(), "/", video_game['EU_Sales'].median())
print(video_game['EU_Sales'].max(), "/", video_game['EU_Sales'].min())
print("\n")

print(video_game['JP_Sales'].mean(), "/", video_game['JP_Sales'].median())
print(video_game['JP_Sales'].max(), "/", video_game['JP_Sales'].min())
print("\n")

print(video_game['Other_Sales'].mean(), "/", video_game['Other_Sales'].median())
print(video_game['Other_Sales'].max(), "/", video_game['Other_Sales'].min())
print("\n")

# Total global sales by genre
video_game_group_by4 = video_game.groupby('Genre')['Global_Sales'].sum()
print(video_game_group_by4)
print("\n")

video_game_group_by5 = video_game.groupby('Genre')['EU_Sales'].sum()
print(video_game_group_by5)
print("\n")

video_game_group_by6 = video_game.groupby('Genre')['JP_Sales'].sum()
print(video_game_group_by6)
print("\n")

video_game_group_by7 = video_game.groupby('Genre')['Other_Sales'].sum()
print(video_game_group_by7)
print("\n")

video_game_group_by25 = video_game.groupby('Genre')['NA_Sales'].sum()
print(video_game_group_by25)
print("\n")

# Average global sales by platform
video_game_group_by8 = video_game.groupby('Platform')['Global_Sales'].mean().reset_index()
print(video_game_group_by8)
print("\n")

video_game_group_by9 = video_game.groupby('Platform')['EU_Sales'].mean().reset_index()
print(video_game_group_by9)
print("\n")

video_game_group_by10 = video_game.groupby('Platform')['JP_Sales'].mean().reset_index()
print(video_game_group_by10)
print("\n")

video_game_group_by11 = video_game.groupby('Platform')['Other_Sales'].mean().reset_index()
print(video_game_group_by11)
print("\n")

video_game_group_by26 = video_game.groupby('Platform')['NA_Sales'].mean().reset_index()
print(video_game_group_by26)
print("\n")

# Total global sales by publisher
video_game_group_by12 = video_game.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).reset_index()
print(video_game_group_by12)
print("\n")

video_game_group_by13 = video_game.groupby('Publisher')['EU_Sales'].sum().sort_values(ascending=False).reset_index()
print(video_game_group_by13)
print("\n")

video_game_group_by14 = video_game.groupby('Publisher')['JP_Sales'].sum().sort_values(ascending=False).reset_index()
print(video_game_group_by14)
print("\n")

video_game_group_by15 = video_game.groupby('Publisher')['Other_Sales'].sum().sort_values(ascending=False).reset_index()
print(video_game_group_by15)
print("\n")

video_game_group_by27 = video_game.groupby('Publisher')['NA_Sales'].sum().sort_values(ascending=False).reset_index()
print(video_game_group_by27)
print("\n")

# Average sales by genre
video_game_group_by16 = video_game.groupby('Genre')['Global_Sales'].mean().reset_index()
print(video_game_group_by16)
print("\n")

video_game_group_by17 = video_game.groupby('Genre')['EU_Sales'].mean().reset_index()
print(video_game_group_by17)
print("\n")

video_game_group_by18 = video_game.groupby('Genre')['JP_Sales'].mean().reset_index()
print(video_game_group_by18)
print("\n")

video_game_group_by19 = video_game.groupby('Genre')['Other_Sales'].mean().reset_index()
print(video_game_group_by19)
print("\n")

video_game_group_by28 = video_game.groupby('Genre')['NA_Sales'].mean().reset_index()
print(video_game_group_by28)
print("\n")

# Highest selling games each genre
highest_selling_indices = video_game.groupby('Genre')['Global_Sales'].idxmax()

video_game_group_by20 = video_game.loc[highest_selling_indices,['Name','Genre','Global_Sales']
].sort_values(by='Global_Sales', ascending=False)
print(video_game_group_by20)
print("\n")

# Highest selling games each genre in Japan
highest_selling_indices1 = video_game.groupby('Genre')['JP_Sales'].idxmax()

video_game_group_by21 = video_game.loc[highest_selling_indices1,['Name','Genre','JP_Sales']
].sort_values(by='JP_Sales', ascending=False)
print(video_game_group_by21)
print("\n")

# Let's create a copy dataset
video_game_copy = video_game.copy()

# Adding new empty column
video_game_copy['New_Column'] = pd.NA
print(video_game_copy.head(5))
print("\n")

# Rename a column
video_game_copy.rename(columns={'Name': 'Game_Name'}, inplace=True)
print(video_game_copy.head(5))
print("\n")

# Change a specific character
video_game_copy.columns = video_game_copy.columns.str.replace('_', '/')
print(video_game_copy.head(5))
print("\n")

# Delete the added column
video_game_copy.drop('New/Column', axis=1, inplace=True)
print(video_game_copy.head(5))
print("\n")

# isin() function
publishers = ['Nintendo', 'Activision']
video_game_copy_filtered = video_game_copy[video_game_copy['Publisher'].isin(publishers)]
print(video_game_copy_filtered.head(25))
print("\n")

# save main dataset
video_game.to_csv('Video Games List.csv', index=False)
print("\n")
