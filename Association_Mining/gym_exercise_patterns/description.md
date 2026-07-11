# Gym Exercise Recommendation Project
This project uses a workout dataset to discover exercises that frequently appear in the same workout. Empty and unnecessary columns are removed, and rows missing the date, workout name, or exercise name are deleted. 
A workout ID is created by combining the date and workout name, and exercises are grouped into workout transactions. TransactionEncoder and Apriori are used to find frequent exercise pairs, while association rules are sorted using lift and confidence. 
A recommendation function suggests exercises that are commonly performed together with an exercise selected by the user.
