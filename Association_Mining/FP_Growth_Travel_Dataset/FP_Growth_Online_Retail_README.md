# Association Rule Mining with FP-Growth

This project applies the **FP-Growth (Frequent Pattern Growth)** algorithm to the **Online Retail** dataset. Its purpose is to discover products that customers frequently purchase together and generate association rules from these purchasing patterns.

## What Is FP-Growth?

FP-Growth is an association mining algorithm used to find frequent combinations of items in transactional data. It compresses transactions into a structure called an **FP-tree** and extracts frequent itemsets from this tree.

Unlike the Apriori algorithm, FP-Growth does not repeatedly generate and test candidate itemsets. This generally makes it faster and more memory-efficient for large datasets containing many transactions.

## Dataset

The program uses the `Online Retail.xlsx` dataset. Each row represents a product included in a retail transaction. The main columns used by the analysis are:

- `InvoiceNo`: Identifier of the transaction
- `Description`: Name of the purchased product
- `Quantity`: Number of units purchased
- `UnitPrice`: Price of one unit
- `CustomerID`: Identifier of the customer

The dataset file must be stored in the same directory as the Python program.

## Program Workflow

### 1. Dataset inspection

The program displays:

- The first five rows
- Dataset structure and data types
- Descriptive statistics
- Number of missing values in each column

### 2. Data preparation

The data is cleaned by:

- Removing records without a product description
- Removing records without a customer ID
- Keeping only rows with positive quantities
- Keeping only rows with positive unit prices
- Removing cancelled invoices
- Removing unnecessary spaces from product descriptions

After cleaning, the products are grouped by `InvoiceNo`. Each invoice is then treated as one transaction containing a list of purchased products.

### 3. Transaction encoding

`TransactionEncoder` converts the transactions into a Boolean table:

- `True` means the product is present in the transaction.
- `False` means the product is absent from the transaction.

Each row represents an invoice and each column represents a product.

### 4. Frequent itemset discovery

The `fpgrowth()` function identifies frequent itemsets using a minimum support value of `0.01`. Therefore, an itemset must appear in at least **1% of all transactions** to be included.

The frequent itemsets are sorted from the highest support to the lowest, and the first results are displayed.

### 5. Association rule generation

Association rules are generated from the frequent itemsets. The program keeps rules with a lift value of at least `1.0` and displays the first 15 rules.

A rule has the following general form:

> If a customer buys product A, they may also buy product B.

In this structure, product A is the **antecedent**, while product B is the **consequent**.

## Evaluation Metrics

### Support

Support measures how frequently an itemset occurs across all transactions.

`support(A → B) = transactions containing A and B / total transactions`

A higher support value indicates that the combination occurs more frequently.

### Confidence

Confidence measures how often the consequent is purchased when the antecedent is purchased.

`confidence(A → B) = support(A and B) / support(A)`

For example, a confidence of `0.70` means that 70% of transactions containing A also contain B.

### Lift

Lift compares the observed relationship between two itemsets with the relationship expected if they were independent.

`lift(A → B) = confidence(A → B) / support(B)`

- `lift > 1`: A and B have a positive association.
- `lift = 1`: A and B appear to be independent.
- `lift < 1`: A and B have a negative association.

## Requirements

- Python 3
- pandas
- mlxtend
- openpyxl

Install the required packages with:

```bash
pip install pandas mlxtend openpyxl
```

## Running the Program

Place the Python file and `Online Retail.xlsx` in the same directory, and then run:

```bash
python main.py
```

Replace `main.py` with the actual name of the Python file if it is different.

## Important Code Note

In the original Online Retail dataset, cancelled invoice numbers normally begin with an uppercase `C`. Since string matching is case-sensitive, the cancellation filter should use uppercase `C` or normalize the invoice numbers before checking them:

```python
df = df[~df['InvoiceNo'].astype(str).str.upper().str.startswith('C')]
```

This ensures that cancelled transactions are excluded correctly.

## Conclusion

This project demonstrates how FP-Growth can be used for market basket analysis. After cleaning and transforming retail transactions, the algorithm discovers frequently purchased product combinations and produces association rules evaluated through support, confidence, and lift. These findings can support product recommendations, store layout planning, cross-selling, and promotional strategies.
