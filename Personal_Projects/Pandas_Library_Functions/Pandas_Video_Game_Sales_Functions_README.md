# Pandas Function Examples with Video Game Sales Data

This project demonstrates commonly used **pandas** functions with the `vgsales.csv` dataset. It covers dataset inspection, row and column selection, missing-value treatment, filtering, sorting, categorical analysis, statistical calculations, grouping, DataFrame modification, and CSV export.

## What Is pandas?

pandas is a Python library for working with structured data. Its main data structure is the **DataFrame**, a two-dimensional table containing rows and columns. pandas provides tools for loading, cleaning, transforming, analyzing, and saving datasets.

This example uses pandas to explore worldwide video game sales and compare games, genres, platforms, publishers, years, and regional sales.

## Dataset

The program reads:

```text
vgsales.csv
```

The file must be stored in the same directory as the Python program.

### Dataset columns

- `Rank`: Sales ranking of the game
- `Name`: Name of the game
- `Platform`: Platform on which the game was released
- `Year`: Release year
- `Genre`: Game genre
- `Publisher`: Company that published the game
- `NA_Sales`: North American sales
- `EU_Sales`: European sales
- `JP_Sales`: Japanese sales
- `Other_Sales`: Sales in other regions
- `Global_Sales`: Total worldwide sales

Sales values in the commonly used version of this dataset are expressed in millions of units.

## Requirements

- Python 3
- pandas

Install pandas with:

```bash
pip install pandas
```

## Program Workflow

### 1. Importing pandas

```python
import pandas as pd
```

The alias `pd` provides a shorter way to access pandas functions.

### 2. Configuring display options

The program uses `pd.set_option()` to control how DataFrames appear in the console:

- `display.max_columns`: Maximum displayed columns
- `display.width`: Width of the printed output
- `display.max_rows`: Maximum displayed rows
- `display.max_colwidth`: Maximum displayed width of an individual column

These options change only the display. They do not modify the dataset.

### 3. Loading the dataset

```python
video_game = pd.read_csv('vgsales.csv')
```

`pd.read_csv()` reads the CSV file and creates a DataFrame.

## Dataset Inspection

### `head()`

`head()` displays the first five rows by default. It is useful for checking column names and the general structure of the data.

### `tail()`

`tail()` displays the last five rows by default. It helps inspect the end of the dataset.

### `info()`

`info()` displays:

- Column names
- Non-null value counts
- Data types
- Memory usage

It is particularly useful for locating incomplete columns and incorrect data types.

### `describe()`

`describe()` generates descriptive statistics for numerical columns, including count, mean, standard deviation, minimum, quartiles, and maximum.

### `isnull().sum()`

`isnull()` marks missing values as `True`. Applying `sum()` counts the missing values in each column.

### `shape`

`shape` returns a tuple in this form:

```text
(number of rows, number of columns)
```

- `shape[0]` returns the number of rows.
- `shape[1]` returns the number of columns.

### `columns`

`columns` returns the labels of every column in the DataFrame.

### `dtypes`

`dtypes` shows the data type assigned to every column.

## Selecting Rows and Columns

### Selecting rows with `iloc`

`iloc` selects rows and columns according to their integer positions.

```python
video_game.iloc[15]
```

This retrieves the 16th row because Python positions begin at `0`.

```python
video_game.iloc[[0, 4]]
```

This retrieves the first and fifth rows. A list is used when selecting multiple nonconsecutive positions.

### Selecting columns

Column-name lists separate categorical and numerical data:

```python
categorical_columns = ['Name', 'Platform', 'Genre', 'Publisher']
numerical_columns = [
    'Rank', 'Year', 'NA_Sales', 'EU_Sales',
    'JP_Sales', 'Other_Sales', 'Global_Sales'
]
```

Passing either list inside square brackets creates a new DataFrame containing only those columns.

## Data Cleaning

### `drop_duplicates()`

```python
video_game = video_game.drop_duplicates()
```

This removes rows whose values are identical across all columns.

### `dropna()`

```python
video_game = video_game.dropna(subset=['Publisher'])
```

This removes rows where `Publisher` is missing. The `subset` argument limits the missing-value check to the specified column.

### `fillna()` and `median()`

```python
video_game['Year'] = video_game['Year'].fillna(video_game['Year'].median())
```

Missing release years are replaced by the median year. The median is the middle value after sorting and is less affected by extreme values than the mean.

### `astype()`

```python
video_game['Year'] = video_game['Year'].astype(int)
```

This converts the `Year` column to integer values after its missing values have been filled.

The program checks `shape` and missing-value counts again after cleaning to show how the dataset changed.

## Filtering Rows

Boolean conditions can select rows that satisfy specific requirements.

### Single-condition filters

The examples select:

- Games released in or after 2010
- Games with global sales greater than 10
- Games belonging to the Sports genre

The comparison produces a Boolean Series, and placing it inside `video_game[...]` returns only matching rows.

### Multiple-condition filters

```python
video_game[
    (video_game['Genre'] == 'Shooter') &
    (video_game['Publisher'] == 'Nintendo')
]
```

The `&` operator requires both conditions to be true. Each condition must be enclosed in parentheses.

Use `|` when at least one of several conditions may be true, and use `~` to invert a Boolean condition.

### Selecting columns after filtering

The global-sales example displays only `Rank`, `Name`, and `Global_Sales`. This demonstrates how row filtering and column selection can be combined.

## Sorting Data

### `sort_values()`

`sort_values()` orders rows according to one or more columns.

The program demonstrates:

- Global sales from highest to lowest
- Global sales from lowest to highest
- Release year from newest to oldest
- Release year from oldest to newest
- Sorting by `Year` and then by `Global_Sales`

`ascending=False` creates descending order, while `ascending=True` creates ascending order.

When several columns are passed to `by`, the first column is the primary sorting key and the following columns resolve ties.

## Unique-Value Analysis

### `nunique()`

`nunique()` counts distinct non-missing values in a column. The program uses it to count genres, platforms, and publishers.

### `unique()`

`unique()` returns the distinct values themselves. Calling `tolist()` converts the resulting array into a standard Python list.

### `value_counts()`

`value_counts()` counts how often each distinct value occurs. It is used for platforms, publishers, and game names.

## Calculating Category Percentages

The program calculates percentages with:

```python
video_game['Genre'].value_counts(normalize=True) * 100
```

Setting `normalize=True` returns proportions instead of raw counts. Multiplication by `100` converts the proportions into percentages.

### `map()`

The calculated percentage Series is mapped back to the main DataFrame:

```python
video_game['Genre_Percentage'] = video_game['Genre'].map(genre_percentage)
```

Every game receives the percentage associated with its genre. The same procedure is applied to platforms and publishers.

### Creating summary percentage tables

The project selects the category and percentage columns, removes repeated category-percentage pairs, and sorts the result in descending order. This produces one summary row per genre, platform, or publisher.

## Finding Extreme Values

The program finds:

- The game with the highest global sales
- The game with the lowest global sales
- The game in the earliest release year
- The game in the latest release year

It does this by sorting the relevant column and selecting the first row with `head(1)`.

This returns one row only. If several games share the same minimum or maximum, the other tied rows are not shown. A Boolean comparison with `max()` or `min()` would return every tie.

## Statistical Operations

The project calculates individual statistics for regional and global sales:

- `mean()`: Arithmetic average
- `median()`: Middle value
- `max()`: Largest value
- `min()`: Smallest value

Although `describe()` already summarizes these values, calling the functions separately demonstrates how to retrieve a specific statistic.

## Grouping and Aggregation

### `groupby()`

`groupby()` divides rows into groups based on a categorical column. An aggregation function then summarizes each group.

### `count()`

The count examples calculate:

- Number of games per genre
- Number of games released per year
- Number of games produced by each publisher

### `sum()`

The sum examples calculate total sales:

- By genre for every sales region
- By publisher for every sales region

This answers questions such as which genre or publisher accumulated the most sales.

### `mean()`

The mean examples calculate average sales:

- By platform for every sales region
- By genre for every sales region

This measures average performance rather than total accumulated sales.

### `reset_index()`

After `groupby()`, the grouping column commonly becomes the result's index. `reset_index()` converts it back into a regular column and returns a standard DataFrame.

### `idxmax()` and `loc`

The program finds the highest-selling game in each genre by:

1. Grouping games by `Genre`
2. Calling `idxmax()` on the relevant sales column
3. Using `loc` to retrieve the complete rows at those index labels

This is performed for global sales and Japanese sales.

## Copying and Modifying a DataFrame

### `copy()`

```python
video_game_copy = video_game.copy()
```

`copy()` creates an independent DataFrame so experimental modifications do not change the main dataset.

### Adding a column

```python
video_game_copy['New_Column'] = pd.NA
```

This adds a column filled with pandas missing values. Assigning an empty Python list would fail unless its length exactly matched the number of DataFrame rows.

### `rename()`

```python
video_game_copy.rename(columns={'Name': 'Game_Name'}, inplace=True)
```

This changes the `Name` label to `Game_Name`. With `inplace=True`, the existing DataFrame is modified directly.

### Replacing characters in column labels

```python
video_game_copy.columns = video_game_copy.columns.str.replace('_', '/')
```

This applies a string replacement to every column label. For example, `Game_Name` becomes `Game/Name`.

### `drop()`

```python
video_game_copy.drop('New/Column', axis=1, inplace=True)
```

This deletes the added column. `axis=1` indicates a column operation; `axis=0` would refer to rows.

### `isin()`

```python
publishers = ['Nintendo', 'Activision']
video_game_copy[video_game_copy['Publisher'].isin(publishers)]
```

`isin()` returns `True` when a value belongs to the supplied collection. The example keeps games published by either Nintendo or Activision.

## Exporting the Result

```python
video_game.to_csv('Video Games List.csv', index=False)
```

`to_csv()` saves the processed DataFrame as a new CSV file. `index=False` prevents pandas from writing the DataFrame index as an additional column.

## Important Corrections and Observations

### Do not remove duplicates from grouped counts

These grouped count operations end with `drop_duplicates()` in the original script:

```python
video_game.groupby('Genre')['Name'].count().sort_values(...).drop_duplicates()
```

After grouping, different genres, years, or publishers may legitimately have the same count. `drop_duplicates()` removes repeated **count values**, even when they belong to different groups. It should be removed from the three grouped count statements so that every group remains visible.

### Recreate subsets after cleaning

`video_games_categorical` and `video_games_numerical` are created before duplicates and missing values are removed. They are separate snapshots based on the original DataFrame selection and do not automatically represent the later cleaned rows.

The script correctly creates `video_games_categorical1` and `video_games_numerical1` after cleaning. Analyses intended to describe the cleaned dataset should use those later subsets or the cleaned `video_game` DataFrame directly.

### Percentage columns repeat group information

Mapping genre, platform, and publisher percentages into the main DataFrame is valid, but the same percentage is repeated for every row in a category. If only a summary is required, the normalized `value_counts()` results can be displayed directly without adding three columns to every game row.

### Column naming

The variable name `video_game_group_by25` works, but descriptive names such as `total_na_sales_by_genre` make a long analysis easier to read and maintain.

## Function Summary

| Function or attribute | Purpose in this project |
| --- | --- |
| `read_csv()` | Loads the video game CSV file |
| `head()` / `tail()` | Displays the beginning or end of the data |
| `info()` | Shows structure, types, and non-null counts |
| `describe()` | Produces numerical descriptive statistics |
| `isnull().sum()` | Counts missing values |
| `shape` | Returns row and column counts |
| `columns` / `dtypes` | Shows column labels and data types |
| `iloc` | Selects rows by integer position |
| `drop_duplicates()` | Removes identical complete rows |
| `dropna()` | Removes rows with missing required values |
| `fillna()` | Replaces missing values |
| `astype()` | Converts a column's data type |
| `sort_values()` | Sorts rows by one or more columns |
| `nunique()` / `unique()` | Counts or returns distinct values |
| `value_counts()` | Counts category frequencies or proportions |
| `map()` | Maps category-based values back to rows |
| `groupby()` | Creates groups for aggregation |
| `count()` / `sum()` / `mean()` | Aggregates grouped values |
| `reset_index()` | Restores a grouped index as a column |
| `idxmax()` | Finds the index of the maximum in each group |
| `loc` | Selects rows by index labels |
| `copy()` | Creates an independent DataFrame copy |
| `rename()` | Renames columns |
| `drop()` | Deletes rows or columns |
| `isin()` | Tests membership in a collection |
| `to_csv()` | Exports a DataFrame to a CSV file |

## Conclusion

This project provides a broad practical introduction to pandas using video game sales data. It begins with loading and inspecting a DataFrame, then demonstrates cleaning, row and column selection, conditional filtering, sorting, frequency analysis, percentage calculation, statistics, and grouped aggregation. It also shows how to copy and modify a DataFrame safely and export the processed result. Together, these examples cover the main pandas operations needed for introductory exploratory data analysis.
