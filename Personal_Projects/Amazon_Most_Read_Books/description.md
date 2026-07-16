
# Amazon Books Data Analysis and Visualization

This project combines book, main genre, and subgenre datasets to analyze the most-rated and highest-rated books. The results are presented using horizontal bar charts.

## Project Objectives

The main objectives of this project are:

- To examine book and genre information stored in different CSV files
- To clean missing and unnecessary data
- To combine book data with main genre and subgenre tables
- To identify the 20 most-rated books
- To identify the 15 highest-rated books
- To display book titles written in different scripts using suitable fonts
- To save the analysis results as PNG images

## Technologies Used

- Python
- pandas
- Matplotlib
- pathlib
- unicodedata

## Dataset Files

The following CSV files are required to run the project:

| File | Description |
| --- | --- |
| `Books_df.csv` | Book titles, authors, ratings, rating counts, and genre information |
| `Genre_df.csv` | Information about the main genres |
| `Sub_Genre_df.csv` | Information about the subgenres |

The CSV files must be located in the same directory as the Python script.

## Data Preparation

The following data preparation steps are performed:

1. The three datasets are loaded with pandas, and their first rows are examined.
2. Data types, descriptive statistics, and missing values are inspected.
3. Book records with missing author values are removed.
4. URL columns that are not needed for the analysis are removed.
5. The `No. of People rated` and `No. of Books` columns are converted to integers.
6. Invalid characters in book titles and author names are cleaned.
7. The book dataset is combined with the main genre and subgenre datasets using inner joins.
8. Unnecessary columns are removed, and the remaining columns are renamed.
9. Duplicate book records are removed.

## Analyses

### Most-Rated Books

The books are sorted in descending order according to the `No. of People rated` column. The first 20 books are printed to the console and displayed in a horizontal bar chart.

Generated file:

```text
Most Rated 20 Books.png
```

### Highest-Rated Books

The books are sorted in descending order according to the `Rating` column. The first 20 results are printed to the console, while the first 15 books are displayed in the chart.

Generated file:

```text
Highest Rated 15 Books.png
```

## Multilingual Font Support

Some book titles in the dataset may use Arabic, Devanagari, Bengali, Gujarati, Tamil, or Malayalam scripts instead of the Latin alphabet. To reduce missing-character and empty-box issues, the following font files are added to the project:

```text
fonts/
├── AnekDevanagari-Regular.ttf
├── AnekGujarati-Regular.ttf
├── AnekMalayalam-Regular.ttf
├── NotoNaskhArabic-Regular.ttf
├── NotoSansBengali-Regular.ttf
└── NotoSansTamil-Regular.ttf
```

The program registers the available fonts with Matplotlib's font manager. If a font file cannot be found, a warning is printed to the console, and the program continues using the remaining fonts.

## Installation

Python 3 must be installed. Install the required libraries by running:

```bash
pip install pandas matplotlib
```

## Project Structure

```text
project/
├── analysis.py
├── Books_df.csv
├── Genre_df.csv
├── Sub_Genre_df.csv
├── fonts/
│   ├── AnekDevanagari-Regular.ttf
│   ├── AnekGujarati-Regular.ttf
│   ├── AnekMalayalam-Regular.ttf
│   ├── NotoNaskhArabic-Regular.ttf
│   ├── NotoSansBengali-Regular.ttf
│   └── NotoSansTamil-Regular.ttf
└── README.md
```

`analysis.py` is an example filename. If your Python file has a different name, use that name in the command below.

## Running the Project

Open a terminal in the project directory and run:

```bash
python analysis.py
```

When the program runs:

- Information about the datasets is printed to the console.
- The cleaned and merged datasets are displayed.
- The most-rated and highest-rated books are listed.
- Two charts are saved in the project directory as PNG files.
- The chart windows are displayed on the screen.

## Conclusion

This project demonstrates how multiple data sources can be cleaned, combined, sorted, and analyzed using pandas. The Matplotlib charts make it easy to compare books based on popularity and user ratings. Multilingual font support also helps display book titles written in different scripts more accurately.
