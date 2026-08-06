# Bank Marketing Classification with Logistic Regression

This project uses **Logistic Regression** to predict whether a bank customer will subscribe to a term deposit. It demonstrates dataset inspection, categorical encoding, feature scaling, stratified train-test splitting, model training, and classification evaluation.

## What Is Logistic Regression?

Logistic Regression is a supervised machine learning algorithm mainly used for classification. Despite the word *regression* in its name, it predicts the probability that an observation belongs to a particular class.

For binary classification, the model calculates a value between `0` and `1` using the logistic, or sigmoid, function. A probability threshold—usually `0.5`—is then used to assign a class. In this project, the two classes are:

- `0`: The customer did not subscribe to a term deposit.
- `1`: The customer subscribed to a term deposit.

Logistic Regression is commonly used because it is relatively fast, interpretable, and suitable for problems with numerical and categorical predictors.

## Project Objective

The objective is to train a binary classification model that predicts the target column `y` from customer, financial, and marketing-campaign information.

## Dataset

The program uses the Bank Marketing dataset stored as:

```text
bank-full.csv
```

The file uses semicolons instead of commas as separators, so it is loaded with:

```python
pd.read_csv('bank-full.csv', sep=';')
```

The target variable is `y`, which indicates whether the customer subscribed to a term deposit.

### Features used by the model

Numerical and binary features:

- `age`: Customer's age
- `default`: Whether the customer has credit in default
- `balance`: Average yearly account balance
- `housing`: Whether the customer has a housing loan
- `loan`: Whether the customer has a personal loan
- `day`: Day of the month of the last contact
- `duration`: Duration of the last contact
- `campaign`: Number of contacts during the current campaign
- `pdays`: Days since the customer was previously contacted
- `previous`: Number of contacts before the current campaign

Categorical features:

- `job`: Customer's occupation
- `marital`: Marital status
- `education`: Education level
- `contact`: Contact communication type
- `month`: Month of the last contact
- `poutcome`: Outcome of the previous marketing campaign

The `y` column is not included in the input features because it is the value the model must predict.

## Program Workflow

### 1. Dataset inspection

The program initially displays:

- First five rows
- Last five rows
- Descriptive statistics
- Column names and data types
- Dataset dimensions
- Missing-value counts

These operations provide a general understanding of the data before model preparation.

### 2. Separating the target

The `y` column is removed from the feature DataFrame. It is later converted into a numerical target:

- `yes` becomes `1`.
- `no` becomes `0`.

This creates the binary target required for Logistic Regression.

### 3. Converting binary features

The `default`, `housing`, and `loan` columns each contain `yes` and `no` values. They are converted into binary numbers:

- `yes` becomes `1`.
- `no` becomes `0`.

This allows the model to process these features numerically.

### 4. Encoding categorical features

The remaining categorical columns are transformed with `OneHotEncoder`. One-hot encoding creates a separate binary column for every category.

For example, a column such as `marital` can become columns similar to:

```text
marital_divorced
marital_married
marital_single
```

The encoder uses:

```python
sparse_output=False
handle_unknown='ignore'
```

- `sparse_output=False` returns a regular dense array.
- `handle_unknown='ignore'` prevents an error if an unseen category is encountered during transformation.

### 5. Building the feature matrix

The numerical DataFrame and the one-hot encoded categorical DataFrame are combined horizontally to create `X`, the final feature matrix.

The binary target is stored in `y`.

### 6. Splitting the dataset

The data is divided into training and testing sets:

- 70% is used for training.
- 30% is used for testing.
- `random_state=23` makes the split reproducible.
- `stratify=y` preserves approximately the same class proportions in both sets.

Stratification is especially useful when one target class appears much more frequently than the other.

### 7. Scaling numerical features

`StandardScaler` standardizes the numerical columns so that they are measured on comparable scales. Standardization generally gives each feature a mean near `0` and a standard deviation near `1`.

The scaler is fitted only to the training data:

```python
scaler.fit_transform(X_train[numerical_columns])
```

The learned transformation is then applied to the test data:

```python
scaler.transform(X_test[numerical_columns])
```

Fitting the scaler only on the training set prevents information from the test set from influencing model training.

The one-hot encoded columns are not standardized because they already contain binary values.

### 8. Training the model

The model is created with:

```python
LogisticRegression(
    max_iter=20000,
    class_weight='balanced',
    random_state=0
)
```

- `max_iter=20000` gives the optimization process enough iterations to converge.
- `class_weight='balanced'` gives more importance to the less frequent class and helps the model avoid favoring the majority class.
- `random_state=0` supports reproducibility where randomness is involved.

The model learns from `X_train` and `y_train`, then predicts the labels for `X_test`.

## Model Evaluation

### Accuracy

Accuracy is the percentage of test observations classified correctly:

```text
accuracy = correct predictions / all predictions
```

Although useful as a general measure, accuracy can be misleading for imbalanced datasets. A model may obtain high accuracy by predicting the majority class frequently while performing poorly on the minority class.

### Precision

Precision measures how many customers predicted as subscribers actually subscribed:

```text
precision = true positives / (true positives + false positives)
```

High precision means that the model produces relatively few false-positive subscription predictions.

### Recall

Recall measures how many actual subscribers were identified correctly:

```text
recall = true positives / (true positives + false negatives)
```

High recall means that the model misses relatively few customers who actually subscribed.

### F1-score

The F1-score is the harmonic mean of precision and recall. It is useful when both false positives and false negatives matter, particularly when the class distribution is imbalanced.

### Support

Support is the number of true observations belonging to each class in the test set. It shows how many examples were used to calculate the metrics for each class.

The `classification_report()` function displays precision, recall, F1-score, and support for both target classes.

## Requirements

- Python 3
- pandas
- scikit-learn

Install the required packages with:

```bash
pip install pandas scikit-learn
```

## Running the Program

Place `bank-full.csv` and the Python file in the same directory. Run the program with:

```bash
python main.py
```

Replace `main.py` with the actual name of the Python file if necessary.

## Important Methodology Notes

### Fit preprocessing only on training data

In the current program, `OneHotEncoder` is fitted before the train-test split. A stricter machine learning workflow splits the original data first, fits the encoder only on the training set, and then uses that fitted encoder to transform both sets. This avoids allowing the preprocessing stage to learn the complete list of categories from the test set.

A `ColumnTransformer` combined with a scikit-learn `Pipeline` is a convenient way to keep encoding, scaling, and model training inside a training-only workflow.

### Be careful with the `duration` feature

The `duration` value is known only after a marketing call has finished. It can be useful when analyzing completed calls, but it should be excluded if the goal is to predict subscription before making or completing the call. Including it in that scenario would provide information unavailable at prediction time.

### Class balancing changes the decision emphasis

`class_weight='balanced'` can improve the detection of the minority class, but it may also create more false positives. Therefore, precision, recall, and F1-score should be examined alongside accuracy.

## Conclusion

This project demonstrates a complete binary classification workflow using Logistic Regression. Categorical variables are one-hot encoded, binary variables are mapped to `0` and `1`, numerical features are standardized, and the target distribution is preserved through stratified splitting. The balanced Logistic Regression model predicts whether customers subscribe to a term deposit and is evaluated using accuracy and a detailed classification report. Because the dataset can be imbalanced, precision, recall, and F1-score provide essential information beyond accuracy alone.
