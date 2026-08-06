# Tourist Venue Association Mining with FP-Growth

This project applies the **FP-Growth (Frequent Pattern Growth)** algorithm to the `dataset_TSMC2014_NYC.csv` dataset. It identifies categories of tourist venues that are commonly visited by the same users in New York City and generates association rules between those categories.

## What Is FP-Growth?

FP-Growth is an association mining algorithm used to discover item combinations that frequently occur together in transactional data. It stores the transactions in a compressed structure called an **FP-tree** and extracts frequent itemsets from that structure.

Unlike Apriori, FP-Growth does not repeatedly create and test candidate itemsets. It is therefore generally more efficient when working with large datasets containing many transactions or possible items.

## Project Objective

The objective is to answer questions such as:

> If a user visits one category of tourist venue, which other venue category are they also likely to visit?

These patterns may support tourism analysis, destination recommendations, itinerary planning, and a better understanding of visitor interests.

## Dataset

The program reads the following file:

```text
dataset_TSMC2014_NYC.csv
```

The analysis mainly uses these columns:

- `userId`: Identifier of the user who recorded a check-in
- `venueCategory`: Category of the visited venue

The dataset must be stored in the same directory as the Python program.

## Program Workflow

### 1. Dataset inspection

After loading the CSV file, the program displays:

- Dataset shape
- First five rows
- Column names and data types
- Descriptive statistics
- Missing-value counts

This provides an initial understanding of the dataset before preprocessing.

### 2. Removing duplicate records

Duplicate rows are removed to prevent identical records from being counted more than once.

### 3. Handling missing venue categories

Text representations of missing values, including `"NaN"`, `"nan"`, `"NAN"`, and empty strings, are converted to actual pandas missing values. Rows without a valid `venueCategory` are then removed.

### 4. Filtering venue categories

The project focuses on tourism-related venue categories. The following categories are excluded:

- Medical Center
- Subway
- Office
- Student Center
- Bus Station
- Home (private)

Venue categories containing the word `Restaurant` are also removed. This narrows the analysis to the types of tourist venues relevant to the project.

### 5. Selecting the required columns

Only `userId` and `venueCategory` are retained because these are the fields required to build the transactions.

### 6. Creating user transactions

The dataset is grouped by `userId`. All unique venue categories visited by one user are combined into a single transaction.

For example, one transaction might be conceptually represented as:

```text
[Museum, Park, Art Gallery]
```

`dict.fromkeys()` removes repeated categories while preserving their original order. As a result, a category visited several times by one user appears only once in that user's transaction.

The model therefore analyzes **category co-occurrence per user**, not the number of visits or the chronological order of check-ins.

### 7. Encoding the transactions

`TransactionEncoder` converts the list of transactions into a Boolean table:

- Each row represents one user.
- Each column represents one venue category.
- `True` means the user visited that category.
- `False` means the user did not visit that category.

This Boolean format is required by the FP-Growth implementation in `mlxtend`.

### 8. Finding frequent itemsets

The program runs FP-Growth with these settings:

```python
min_support=0.05
max_len=3
```

- `min_support=0.05` means an itemset must occur in at least **5% of user transactions**.
- `max_len=3` limits each frequent itemset to a maximum of three venue categories.

The frequent itemsets are sorted by support in descending order so that the most common patterns appear first.

### 9. Generating association rules

Association rules are produced with lift as the filtering metric:

```python
metric='lift'
min_threshold=1.2
```

Only rules with a lift of at least **1.2** are retained. This means the antecedent and consequent occur together at least 1.2 times as often as expected if they were independent.

The `frozenset` representations of the antecedents and consequents are converted into readable, comma-separated text before the results are displayed.

## Understanding an Association Rule

An association rule has this general form:

```text
Antecedent → Consequent
```

For example:

```text
Museum → Art Gallery
```

This rule suggests that users who visited a museum also tended to visit an art gallery. It describes an association and does not prove that one visit caused the other.

## Evaluation Metrics

### Support

Support measures how frequently an itemset appears across all user transactions.

```text
support(A → B) = users who visited A and B / total users
```

A higher support value indicates that the combination is more common.

### Confidence

Confidence measures the proportion of users who visited B among the users who visited A.

```text
confidence(A → B) = support(A and B) / support(A)
```

For example, a confidence value of `0.70` means that 70% of users who visited A also visited B.

### Lift

Lift measures the strength of the relationship while considering how frequently the consequent occurs on its own.

```text
lift(A → B) = confidence(A → B) / support(B)
```

- `lift > 1`: Positive association
- `lift = 1`: No apparent association
- `lift < 1`: Negative association

This project uses a stricter minimum lift of `1.2` to keep rules with a more meaningful positive association.

## Requirements

- Python 3
- pandas
- mlxtend

Install the required packages with:

```bash
pip install pandas mlxtend
```

## Running the Program

Place the dataset and Python file in the same directory. Run the program with:

```bash
python main.py
```

Replace `main.py` with the actual name of the Python file when necessary.

## Important Notes

- The code uses a lift threshold of `1.2`, even though its nearby comment says lift must be at least 1. The effective threshold is the value supplied to `min_threshold`, which is **1.2**.
- Restaurant filtering is case-sensitive by default. If the dataset contains variations such as `restaurant`, a case-insensitive filter would make the cleaning more consistent.
- Each user becomes one transaction, regardless of how many times they checked in. Therefore, the results represent shared user interests rather than visit frequency.
- The association rules show co-occurrence patterns, not causal relationships or the order in which venues were visited.

## Conclusion

This project demonstrates how FP-Growth can identify tourism-related venue categories that are frequently visited by the same users. After removing duplicates, missing values, non-touristic locations, and restaurants, the program groups unique venue categories by user and converts them into Boolean transactions. It then discovers frequent combinations with at least 5% support and generates association rules with a minimum lift of 1.2. The resulting patterns can be useful for tourism recommendations and visitor-behavior analysis.
