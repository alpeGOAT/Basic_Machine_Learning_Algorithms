import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 450)
pd.set_option('display.max_colwidth', 100)

dataset = pd.read_csv('student-por.csv',sep=';')

print("\nFirst 5 Rows of the Dataset")
print(dataset.head())

print("\nLast 5 Rows of the Dataset")
print(dataset.tail())

print("\nDataset Shape")
print(dataset.shape)

print("\nDataset Information")
dataset.info()

print("\nDataset Description")
print(dataset.describe())

print("\nMissing Values in the Dataset")
print(dataset.isnull().sum())

if dataset.isnull().sum().sum() > 0:
    print("\nThere are missing values in the Dataset\n")
elif dataset.isnull().sum().sum() == 0:
    print("\nThere are no missing values in the Dataset\n")

dataset = dataset.drop_duplicates()

each_school_student_amount = dataset['school'].value_counts()
print("How much student each school has?")
print(each_school_student_amount, "\n")

each_student_gender = dataset['sex'].value_counts()
print("Students Gender Distribution")
print(each_student_gender, "\n")

age_student_amounts = dataset['age'].value_counts()
print("Students Age Distribution")
print(age_student_amounts, "\n")

binary_addresses = dataset['address'].value_counts()
print("Address Distribution (Binary)")
print(binary_addresses, "\n")

# List of Jobs of mothers and fathers
print(dataset['Fjob'].unique(), "\n")
print(dataset['Mjob'].unique(), "\n")

print("Father Job Distribution")
print(dataset['Fjob'].value_counts(), "\n")
print("Mother Job Distribution")
print(dataset['Mjob'].value_counts(), "\n")

# Students guardians and their count
students_guardians = dataset['guardian'].value_counts()
print("Guardian Distribution")
print(students_guardians, "\n")

# School Distribution with Matplotlib
plt.bar(dataset['school'].unique(), each_school_student_amount, color='royalblue', width=0.7, align='center')
plt.xlabel("School", fontsize=12, loc='center', labelpad=-6)
plt.ylabel("Student Count")
plt.title('Student Count of Each School')
plt.savefig('school_distribution.jpg')
plt.show()

# Gender Distribution of the Students
plt.bar(dataset['sex'].unique(), each_student_gender, color='royalblue', width=0.7)
plt.xlabel('Gender (Male / Female)', labelpad=-6, fontsize=12)
plt.ylabel('Student Count')
plt.title('Gender Distribution of the Students')
plt.savefig('gender_distribution.png')
plt.show()

# Students Age Distribution
plt.scatter(range(1,650), dataset['age'], color='black', label='Age', alpha=0.5)
plt.xlabel('Students ID')
plt.ylabel('Age', labelpad=7,fontsize=12)
plt.title('Student Age Distribution (Scatter)')
plt.legend()
plt.savefig('student_age_distribution_scatter.jpg')
plt.show()

plt.bar(dataset['age'].unique(), age_student_amounts, color='red', width=0.7, align='center', alpha=0.6)
plt.xlabel('Students Age')
plt.ylabel('Student Amount')
plt.title('Amount of each unique Student Age')
plt.savefig('student_age_distribution_bar.png')
plt.show()

# Students Guardians Distribution (Mother, Father, Other)
plt.bar(dataset['guardian'].unique(), students_guardians, color='red', width=0.8, align='center', alpha=0.6)
plt.xlabel('Student Guardian')
plt.ylabel('Student Count',labelpad=6)
plt.title('Amount of each Student Guardian')
plt.savefig('guardian_distribution.jpg')
plt.show()

# Mother Job Distribution (matplotlib)
plt.bar(dataset['Mjob'].unique(), dataset['Mjob'].value_counts(), color='magenta', alpha=0.5, width=0.7)
plt.xlabel('Mother Job')
plt.ylabel('Student Count',labelpad=6)
plt.title('Jobs of Mothers and their amount')
plt.savefig('mother_job_distribution.png')
plt.show()

# Father Job Distribution (matplotlib)
plt.bar(dataset['Fjob'].unique(), dataset['Fjob'].value_counts(), color='royalblue', alpha=0.5, width=0.7)
plt.xlabel('Father Job')
plt.ylabel('Student Count',labelpad=6)
plt.title('Jobs of Fathers and their amount')
plt.savefig('father_job_distribution.png')
plt.show()

print("Average Travel time of the Students")
print(dataset['traveltime'].mean().round(2), "\n")

print("Average Study time of the Students")
print(dataset['studytime'].mean().round(2), "\n")

health_status = dataset['health'].value_counts()
print("Count of Students in each Health Status")
print(health_status, "\n")

# Health Status Distribution
plt.bar(dataset['health'].unique(), health_status, color='red', width=0.7, align='center', alpha=0.6)
plt.xlabel('Health Status')
plt.ylabel('Count')
plt.title('Health Status Distribution')
plt.savefig('health_status_distribution.jpg')
plt.show()

print("Absence Average of Students")
print(dataset['absences'].mean().round(2), "\n")

absence_distribution = dataset['absences'].value_counts().sort_index()
print(absence_distribution, "\n")

# Absence Distribution with matplotlib
plt.scatter(dataset['absences'].unique(), absence_distribution, color='red', label='Absences', alpha=0.6)
plt.xlabel('Absences')
plt.ylabel('Student Count',labelpad=6)
plt.title('Student amount of each Absence value')
plt.legend()
plt.savefig('absence_amount_distribution.png')
plt.show()

plt.scatter(range(1,650), dataset['absences'], color='black', alpha=0.6)
plt.xlabel('Student ID')
plt.ylabel('Absence Count')
plt.title('Absence Distribution')
plt.savefig('absence_distribution.jpg')
plt.show()

print("\nAverage value of First Grade: ", dataset['G1'].mean().round(2))
print("\nAverage value of Second Grade: ", dataset['G2'].mean().round(2))
print("\nAverage value of Third Grade: ", dataset['G3'].mean().round(2))

average_grades = dataset[['G1','G2', 'G3']].mean().round(2)

fig, ax = plt.subplots()
ax.bar(['G1', 'G2', 'G3'], average_grades, alpha=0.6, width=0.7)
ax.set_xlabel('Grades (G1, G2, G3)')
ax.set_ylabel('Average Scores (G1, G2, G3)')
plt.savefig('each_grades_average_score.jpg')
plt.show()

plt.scatter(range(1,650), dataset['G1'], color='red', label='Grades (G1)', alpha=0.6)
plt.xlabel('Student ID')
plt.ylabel('Scores (G1)')
plt.title('Grade Distribution (G1)')
plt.legend()
plt.grid(True)
plt.savefig('grade1_distribution.png')
plt.show()

plt.scatter(range(1,650), dataset['G2'], color='red', label='Grades (G2)', alpha=0.6)
plt.xlabel('Student ID')
plt.ylabel('Scores (G2)')
plt.title('Grade Distribution (G2)')
plt.legend()
plt.grid(True)
plt.savefig('grade2_distribution.png')
plt.show()

plt.scatter(range(1,650), dataset['G3'], color='blue', label='Grades (G3)', alpha=0.6)
plt.xlabel('Student ID')
plt.ylabel('Scores (G3)')
plt.title('Grade Distribution (G3)')
plt.legend()
plt.grid(True)
plt.savefig('grade3_distribution.png')
plt.show()

X = dataset[['studytime']]
y = dataset['G1']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.scatter(X, y, color='black', label='Data Points')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title('Linear Regression on G1')
plt.xlabel('Study Time')
plt.ylabel('G1 Scores')
plt.legend()
plt.grid(True)
plt.title('Relationship Between Study Time and G1 Score')
plt.savefig('linear_regression_G1.png')
plt.show()

X2 = dataset[['studytime']]
y2 = dataset['G2']

model2 = LinearRegression()
model2.fit(X2, y2)
y2_pred = model.predict(X2)

plt.scatter(X2, y2, color='black', label='Data Points')
plt.plot(X2, y2_pred, color='red', label='Regression Line')
plt.title('Linear Regression on G2')
plt.xlabel('Study Time')
plt.ylabel('G2 Scores')
plt.legend()
plt.grid(True)
plt.title('Relationship Between Study Time and G2 Score')
plt.savefig('linear_regression_G2.png')
plt.show()

X3 = dataset[['studytime']]
y3 = dataset['G3']

model3 = LinearRegression()
model3.fit(X3, y3)
y3_pred = model.predict(X3)

plt.scatter(X3, y3, color='black', label='Data Points')
plt.plot(X3, y3_pred, color='red', label='Regression Line')
plt.title('Linear Regression on G3')
plt.xlabel('Study Time')
plt.ylabel('G3 Scores')
plt.legend()
plt.grid(True)
plt.title('Relationship Between Study Time and G3 Score')
plt.savefig('linear_regression_G3.png')
plt.show()