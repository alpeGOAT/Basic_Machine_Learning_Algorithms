import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 125)
pd.set_option('display.width', 300)
pd.set_option('display.max_colwidth', 100)

student_info = pd.read_csv('data.csv',sep=';')

print("\nFirst 5 rows of the Dataset")
print(student_info.head())

print("\nLast 5 rows of the Dataset")
print(student_info.tail())

print("\nDataset Information")
student_info.info()

print("\nDataset Shape")
print(student_info.shape)

print("\nDataset Description")
print(student_info.describe())

print("\nMissing Values in the Dataset")
print(student_info.isnull().sum())

if student_info.isnull().sum().sum() > 0:
    print("\nThere are missing values in the Dataset\n")
elif student_info.isnull().sum().sum() == 0:
    print("\nThere are no missing values in the Dataset\n")

student_info = student_info.drop_duplicates()

marital_status_distribution = student_info['Marital status'].value_counts().sort_index()
print("Marital Status of the Students")
print(marital_status_distribution, "\n")

plt.figure(figsize=(10,9))
plt.bar(marital_status_distribution.index, marital_status_distribution.values, color='red', alpha=0.6)
plt.xlabel('Marital Status', labelpad=15, fontsize=14)
plt.xticks([1,2,3,4,5,6], ['Single','Married','Widower','Divorce','Facto Union','Legally Separated'])
plt.ylabel('Student Count',labelpad=15, fontsize=14)
plt.title('Student Marital Status Distribution', fontsize=16, y=1.02)
plt.savefig('marital_status_distribution.jpg')
plt.show()

education_level_students = student_info['Previous qualification'].value_counts().sort_index()
print("Education Level of the Students")
print(education_level_students, "\n")

grade_of_students = student_info['Previous qualification (grade)'].value_counts().sort_index()
print("Grade of previous qualification (between 0 and 200)")
print(grade_of_students, "\n")

previous_qualification_average = student_info['Previous qualification (grade)'].mean().round(2)
print("Average Previous Qualification Grade of the Students: ", previous_qualification_average, "\n")

nationality_distribution = student_info['Nationality'].value_counts().sort_index()
print("Distribution of Nationality of the Students")
print(nationality_distribution, "\n")

education_level_mom = student_info["Mother's qualification"].value_counts().sort_index()
print("Education Level of the Mothers of the students")
print(education_level_mom, "\n")

print("Most common education level among Mothers: ", education_level_mom.idxmax(), "\n")

education_level_father = student_info["Father's qualification"].value_counts().sort_index()
print("Education Level of the Fathers of the students")
print(education_level_father, "\n")

print("Most common education level among Fathers: ", education_level_father.idxmax(), "\n")

average_age_of_enrollment = student_info['Age at enrollment'].mean().round(2)
print("Average Age of the Enrollment of the Students: ", average_age_of_enrollment, "\n")

age_at_enrollment_distribution = student_info['Age at enrollment'].value_counts().sort_index()
print("Distribution of Age of the Enrollment of the Students")
print(age_at_enrollment_distribution, "\n")

plt.scatter(range(1,4425), student_info['Age at enrollment'], color='red', alpha=0.6, label='Age at enrollment')
plt.xlabel('Student ID')
plt.ylabel('Age at enrollment')
plt.legend()
plt.title('Student Age of Enrollment Distribution')
plt.grid(True)
plt.savefig('age_at_enrollment_distribution.jpg')
plt.show()

student_gender_distribution = student_info['Gender'].value_counts().sort_index()
print("Distribution of Gender of the Students")
print(student_gender_distribution, "\n")

plt.bar(student_gender_distribution.index, student_gender_distribution.values, color='royalblue', alpha=0.6, linewidth=4)
plt.xlabel('Gender (1->Male, 0->Female)')
plt.xticks([0,1], ['Female (0)', 'Male (1)'])
plt.ylabel('Student Count')
plt.title('Student Gender Distribution')
plt.grid(True)
plt.savefig('student_gender_distribution.png')
plt.show()

mother_occupation_distribution = student_info["Mother's occupation"].value_counts().sort_index()
print("Distribution of Mother's occupation of the Students")
print(mother_occupation_distribution, "\n")

print("Most Common Occupation (Mother): ",mother_occupation_distribution.idxmax(), "\n")

father_occupation_distribution = student_info["Father's occupation"].value_counts().sort_index()
print("Distribution of Father's occupation of the Students")
print(father_occupation_distribution, "\n")

print("Most Common Occupation (Father): ",father_occupation_distribution.idxmax(), "\n")

admission_grade_average = student_info['Admission grade'].mean().round(2)
print("Average Admission Grade of the Students")
print(admission_grade_average, "\n")

admission_grade_distribution = student_info['Admission grade'].value_counts().sort_index()
print("Distribution of Admission grade of the Students")
print(admission_grade_distribution, "\n")

print("Most Common Admission Grade: ",admission_grade_distribution.idxmax(), "\n")

international_students_distribution = student_info['International'].value_counts().sort_index()
print("Student Distribution (International or not)")
print(international_students_distribution, "\n")

plt.bar(international_students_distribution.index, international_students_distribution.values, color='red', alpha=0.5)
plt.xlabel('Internationality')
plt.xticks([0,1], ['Local','International'])
plt.ylabel('Student Count')
plt.yticks([0,200,1000,2000,3000,4000])
plt.title('Student Distribution (International or not)')
plt.grid(True)
plt.savefig('international_student_distribution.png')
plt.show()

special_ed_distribution = student_info['Educational special needs'].value_counts().sort_index()
print('Student Distribution (Special Ed or not')
print(special_ed_distribution, "\n")

plt.bar(special_ed_distribution.index, special_ed_distribution.values, color='royalblue', alpha=0.5)
plt.xlabel('Student Status')
plt.xticks([0,1], ['Normal','Special Ed'])
plt.ylabel('Student Count')
plt.yticks([0,200,1000,2000,3000,4000])
plt.title('Student Distribution (Special Ed or not)')
plt.grid(True)
plt.savefig('special_education_distribution.png')
plt.show()

scholar_distribution = student_info['Scholarship holder'].value_counts().sort_index()
print('Student Distribution (Scholarship or not)')
print(scholar_distribution, "\n")

plt.bar(scholar_distribution.index, scholar_distribution.values, color='green', alpha=0.4)
plt.xlabel('Student Status')
plt.xticks([0,1], ['Normal','Scholarship'])
plt.ylabel('Student Count')
plt.title('Student Distribution (Scholarship or not)')
plt.grid(True)
plt.savefig('scholar_distribution.png')
plt.show()

average_GDP = student_info['GDP'].mean()
print("Average GDP of the Students", average_GDP, "\n")

target_column_distribution = student_info['Target'].value_counts().sort_index()
print("Distribution of Target Column")
print(target_column_distribution, "\n")

plt.bar(target_column_distribution.index, target_column_distribution.values, color='red', alpha=0.5)
plt.xlabel('Target', labelpad=10, fontweight='bold')
plt.ylabel('Student Count', fontweight='bold')
plt.grid(True)
plt.title('Student Distribution', fontweight='bold')
plt.savefig('target_column_distribution.png')
plt.show()

avg_inflation_rate = student_info['Inflation rate'].mean().round(2)
print("Average Inflation rate of the Students", avg_inflation_rate, "\n")

avg_unemployment_rate = student_info['Unemployment rate'].mean().round(2)
print("Average Unemployment rate of the Students", avg_unemployment_rate, "\n")

student_info1 = student_info.copy() # Copying because percentage operations apply

target_percentages = (student_info1['Target'].value_counts(normalize=True).mul(100).round(2).reset_index())
target_percentages.columns = ['Target', 'Target_Percentage']
print(target_percentages, "\n")

gender_percentages = (student_info1['Gender'].value_counts(normalize=True).mul(100).round(2).reset_index())
gender_percentages.columns = ['Gender', 'Gender_Percentage']
print(gender_percentages, "\n")

special_ed_percentages = (student_info1['Educational special needs'].value_counts(normalize=True).mul(100).round(2).reset_index())
special_ed_percentages.columns = ['Special_Ed', 'Special_Ed_Percentage']
print(special_ed_percentages, "\n")

scholarship_percentages = (student_info1['Scholarship holder'].value_counts(normalize=True).mul(100).round(2).reset_index())
scholarship_percentages.columns = ['Scholarship', 'Scholarship_Percentage']
print(scholarship_percentages, "\n")

international_percentages = (student_info1['International'].value_counts(normalize=True).mul(100).round(2).reset_index())
international_percentages.columns = ['International', 'International_Percentage']
print(international_percentages, "\n")

age_17_30 = student_info[student_info['Age at enrollment'].between(17, 30)]

sns.countplot(data=age_17_30, x='Age at enrollment', hue='Gender')
plt.xlabel('Age at Enrollment', fontsize=12, labelpad=8)
plt.ylabel('Student Count', fontsize=12, labelpad=10)
plt.title('Gender Distribution by Age at Enrollment (17–30)', fontsize=15, pad=15, fontweight='bold')
plt.legend(title='Gender', labels=['Female', 'Male'])
plt.grid(axis='y', alpha=0.5)
plt.savefig('age_gender_relation.png')
plt.show()

sns.countplot(data=age_17_30, x='Age at enrollment', hue='Target')
plt.xlabel('Age at Enrollment', fontsize=12, labelpad=8)
plt.ylabel('Student Count', fontsize=12, labelpad=10)
plt.title('Target Distribution by Age (17-30)', fontsize=15, pad=15, fontweight='bold')
plt.legend(title='Target')
plt.grid(True)
plt.savefig('age_target_distribution.png')
plt.show()

print("Each Targets average Admission Grade")
print(student_info.groupby('Target')['Admission grade'].mean(), "\n")

print("Each Targets average Previous Qualification Grade")
print(student_info.groupby('Target')['Previous qualification (grade)'].mean(), "\n")

print("Each Targets First Semester Grade Average")
print(student_info.groupby('Target')['Curricular units 1st sem (grade)'].mean(), "\n")

print("Each Targets Second Semester Grade Average")
print(student_info.groupby('Target')['Curricular units 2nd sem (grade)'].mean(), "\n")

numerical_features = [
    'Previous qualification (grade)',
    'Admission grade',
    'Age at enrollment',

    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',

    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',

    'Unemployment rate',
    'Inflation rate',
    'GDP'
]

X = student_info[numerical_features]
y = student_info['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=26, stratify=y)

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

