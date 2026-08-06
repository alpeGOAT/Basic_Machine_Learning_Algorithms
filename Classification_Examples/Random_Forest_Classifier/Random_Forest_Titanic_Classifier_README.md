# Titanic Survival Prediction with Random Forest

This project uses a **Random Forest Classifier** to predict whether a Titanic passenger survived. It demonstrates dataset inspection, feature selection, missing-value handling, categorical conversion, model training, classification evaluation, and prediction for an individual passenger.

## What Is a Random Forest Classifier?

A Random Forest Classifier is a supervised machine learning algorithm that combines many decision trees to make a classification. Each tree is trained on a random sample of the training data and considers a random subset of features when creating its splits.

For a classification problem, every tree produces a class prediction. The forest combines these predictions through majority voting, and the class receiving the most votes becomes the final prediction.

Using many varied trees usually makes Random Forest more accurate and less likely to overfit than a single decision tree. It can model nonlinear relationships, capture interactions between features, and does not normally require feature scaling.

## Project Objective

The objective is to predict the value of `Survived` for each passenger:

- `0`: The passenger did not survive.
- `1`: The passenger survived.

## Dataset

The program reads the Titanic dataset from:

```text
titanic.csv
```

The dataset must be stored in the same directory as the Python program.

## Features Used

The model uses six passenger characteristics:

- `Pclass`: Passenger class—normally first, second, or third class
- `Sex`: Passenger's sex
- `Age`: Passenger's age
- `SibSp`: Number of siblings or spouses aboard
- `Parch`: Number of parents or children aboard
- `Fare`: Ticket fare paid by the passenger

The target variable is:

- `Survived`: Whether the passenger survived

Other columns in the dataset are not used by this model.

## Program Workflow

### 1. Dataset inspection

The program displays:

- First five rows
- Last five rows
- Dataset dimensions
- Column names and data types
- Descriptive statistics
- Missing-value counts

These operations provide an overview of the data before model preparation.

### 2. Handling missing target values

Rows with a missing `Survived` value are removed. A supervised classification model cannot learn from a row when its correct target class is unknown.

### 3. Selecting inputs and target

The selected passenger features are stored in `X`, while `Survived` is stored in `y`.

```text
X = passenger characteristics
y = survival result
```

### 4. Converting the `Sex` feature

Machine learning models require numerical input. The `Sex` column is therefore mapped as follows:

- `female` becomes `0`.
- `male` becomes `1`.

This produces one binary numerical feature.

### 5. Handling missing ages

Missing values in `Age` are replaced with the median age. The median is less sensitive to unusually high or low values than the mean, so it is a practical choice for this feature.

### 6. Splitting the data

The dataset is divided into training and testing sets:

- 80% is used to train the model.
- 20% is used to test the model.
- `random_state=42` makes the split reproducible.

The test set contains data that is not used to fit the Random Forest and provides an estimate of its performance on unseen passengers.

### 7. Training the Random Forest

The classifier is created with:

```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

- `n_estimators=100` creates a forest containing 100 decision trees.
- `random_state=42` makes the random sampling and model results reproducible.

During training, the trees learn different patterns relating passenger features to survival.

### 8. Generating predictions

The trained model predicts survival classes for the passengers in `X_test`. The predicted values are compared with their true values in `y_test`.

## Model Evaluation

### Accuracy

Accuracy is the proportion of test passengers classified correctly:

```text
accuracy = correct predictions / all test predictions
```

An accuracy of `0.80`, for example, means that 80% of test passengers were assigned the correct survival class.

Accuracy gives a general summary but should be interpreted together with the classification report.

### Precision

Precision measures how many passengers predicted to belong to a class actually belonged to that class.

For the survived class:

```text
precision = correctly predicted survivors / all predicted survivors
```

High precision for class `1` means relatively few non-survivors were incorrectly predicted as survivors.

### Recall

Recall measures how many passengers from a class the model identified correctly.

For the survived class:

```text
recall = correctly predicted survivors / all actual survivors
```

High recall for class `1` means that the model missed relatively few actual survivors.

### F1-score

The F1-score combines precision and recall through their harmonic mean. It is useful when both false-positive and false-negative predictions are important.

### Support

Support is the number of true observations belonging to each class in the test set. It indicates how many examples were used to calculate the metrics for each class.

The `classification_report()` function displays precision, recall, F1-score, and support for both classes.

## Individual Passenger Prediction

The program selects the first passenger from the test set:

```python
sample = X_test.iloc[0:1]
```

Using `0:1` preserves the two-dimensional DataFrame structure required by scikit-learn. The trained model predicts this passenger's survival class.

The passenger's feature values are converted into a dictionary for readable output. The numerical prediction is then displayed as:

- `Survived` when the prediction is `1`
- `Did Not Survive` when the prediction is `0`

This demonstrates how the trained classifier can be used to predict a single new observation.

## Why Scaling Is Not Used

Random Forest splits observations according to feature thresholds. It does not calculate distances between observations or optimize coefficients based on feature magnitude. Therefore, features such as `Age` and `Fare` do not normally need to be standardized.

## Requirements

- Python 3
- pandas
- scikit-learn

Install the required packages with:

```bash
pip install pandas scikit-learn
```

## Running the Program

Place `titanic.csv` and the Python file in the same directory. Then run:

```bash
python main.py
```

Replace `main.py` with the actual name of the Python file if necessary.

## Important Methodology Notes

### Use an explicit copy of the selected features

The feature selection should ideally create a copy:

```python
X = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
```

This makes it explicit that the feature DataFrame can be modified independently and avoids pandas `SettingWithCopyWarning` behavior. Suppressing every warning can hide useful warnings such as this one.

### Calculate the median from training data only

In the current program, the median age is calculated before the train-test split. A stricter workflow splits the dataset first, calculates the median from `X_train`, and applies that same training median to both `X_train` and `X_test`. This prevents information from the test distribution from influencing preprocessing.

A scikit-learn `Pipeline` with a `SimpleImputer` is a convenient way to enforce this training-only workflow.

### Consider stratified splitting

Adding `stratify=y` to `train_test_split()` would preserve approximately the same survival-class proportions in both training and testing sets. This can make evaluation more reliable when the classes are not evenly distributed.

### Evaluate more than accuracy

Accuracy alone does not show whether the model performs equally well for survivors and non-survivors. Precision, recall, F1-score, and a confusion matrix can provide a more complete evaluation.

## Conclusion

This project demonstrates how a Random Forest Classifier can predict Titanic passenger survival using passenger class, sex, age, family relationships, and fare. After preparing the features and replacing missing ages, the data is divided into training and testing sets. A forest of 100 decision trees is trained, evaluated using accuracy and a classification report, and used to predict the survival of an individual passenger. The example also shows why Random Forest is a practical classification algorithm: it can capture complex feature relationships without requiring feature scaling.
