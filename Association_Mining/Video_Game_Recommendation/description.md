#  Video Game Recommendation Project
This project uses a Steam game ownership dataset to find games that are frequently purchased by the same users. Since the original dataset has no column names, the required columns are added manually. 
Only purchase records are kept, duplicate games are removed, and users who purchased only one game are excluded. Apriori and association rules are used to find relationships between games. 
DLCs, deleted scenes, and test servers are filtered out to improve recommendation quality. A recommendation function then suggests related games based on a game entered by the user, using confidence and lift to rank the results.
