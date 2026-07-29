# Multinomial Naive Bayes

import pandas as pd
from pandas._libs.tslibs import vectorized
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 250)

data = {
    'text': [
        'Free money now',
        'Call now to claim your prize',
        'Meet me at the park',
        'Let’s catch up later',
        'Win a new car today!',
        'Lunch plans?',
        'Congratulations! You won a lottery',
        'Can you send me the report?',
        'Exclusive offer for you',
        'Are you coming to the meeting?'
    ],
    'label': ['spam', 'spam', 'not spam', 'not spam', 'spam', 'not spam', 'spam', 'not spam', 'spam', 'not spam']
}

df = pd.DataFrame(data)
print(df.head())

df['label'] = df['label'].map({'spam': 1, 'not spam': 0})

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectors, y_train)

y_pred = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

custom_message = ["Congratulations, you've won a free vacation"]
print(custom_message)
custom_vector = vectorizer.transform(custom_message)
prediction = model.predict(custom_vector)
print("Prediction for custom message:", "Spam" if prediction[0] == 1 else "Not Spam")

custom_message2 = ["Are you coming to lunch?"]
print(custom_message2)
custom_vector2 = vectorizer.transform(custom_message2)
prediction2 = model.predict(custom_vector2)
print("Prediction for custom message:", "Spam" if prediction2[0] == 1 else "Not Spam")

custom_message3 = ["Grüezi zäme, Ich bin Alperen."]
print(custom_message3)
custom_vector3 = vectorizer.transform(custom_message3)
prediction3 = model.predict(custom_vector3)
print("Prediction for custom message:", "Spam" if prediction3[0] == 1 else "Not Spam")

