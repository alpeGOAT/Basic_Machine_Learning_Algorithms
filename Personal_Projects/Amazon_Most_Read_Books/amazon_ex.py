import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from pathlib import Path
import unicodedata
from matplotlib import figure

font_files = {
    "devanagari": "fonts/AnekDevanagari-Regular.ttf",
    "gujarati": "fonts/AnekGujarati-Regular.ttf",
    "malayalam": "fonts/AnekMalayalam-Regular.ttf",
    "arabic": "fonts/NotoNaskhArabic-Regular.ttf",
    "bengali": "fonts/NotoSansBengali-Regular.ttf",
    "tamil": "fonts/NotoSansTamil-Regular.ttf"
}

font_properties = {}
font_names = []

for language, font_path in font_files.items():
    path = Path(font_path)

    if not path.exists():
        print("Font file not found:", path)
        continue

    fm.fontManager.addfont(path)

    font_property = fm.FontProperties(fname=path)
    font_name = font_property.get_name()

    font_properties[language] = font_property
    font_names.append(font_name)

    print(f"Font successfully added: {language} -> {font_name}")

plt.rcParams["font.family"] = "sans-serif"

plt.rcParams["font.sans-serif"] = font_names + [
    "Nirmala UI",
    "Segoe UI Symbol",
    "DejaVu Sans"
]

plt.rcParams["axes.unicode_minus"] = False

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', 200)

books =pd.read_csv('Books_df.csv')
sub_genres = pd.read_csv('Sub_Genre_df.csv')
genres = pd.read_csv('Genre_df.csv')

print("First 5 rows of Books Dataset")
print(books.head())
print("\n")

print("Book Dataset Information")
print(books.info())
print("\n")

print("Missing Values in Books Dataset")
print(books.isnull().sum())
print("\n")

print("Book Dataset Description")
print(books.describe())
print("-----" * 10)

print("First 5 rows of Genres Dataset")
print(genres.head())
print("\n")

print("Genre Dataset Information")
print(genres.info())
print("\n")

print("Missing Values in Genres Dataset")
print(genres.isnull().sum())
print("\n")

print("Genres Dataset Description")
print(genres.describe())
print("----" * 10)

print("First 5 rows of Sub Genres Dataset")
print(sub_genres.head())
print("\n")

print("Sub Genres Dataset Information")
print(sub_genres.info())
print("\n")

print("Missing Values in Sub Genres Dataset")
print(sub_genres.isnull().sum())
print("\n")

print("Sub Genres Dataset Description")
print(sub_genres.describe())
print("----" * 10)

books = books.dropna(subset=['Author'])

books = books.drop(columns='URLs')
genres = genres.drop(columns='URL')
sub_genres = sub_genres.drop(columns='URLs')

books['No. of People rated'] = books['No. of People rated'].astype(int)
sub_genres['No. of Books'] = sub_genres['No. of Books'].astype(int)

for column in ["Title", "Author"]:
    books[column] = (
        books[column]
        .astype(str)
        .str.replace("\x92", "'", regex=False)
        .str.replace("\ufffd", "", regex=False)
    )

print("Books Dataset after Data Preparation (First 5 rows)")
print(books.head())
print("\n")

print("Sub Genres Dataset after Data Preparation (First 5 rows)")
print(sub_genres.head())
print("\n")

print("Genres Dataset after Data Preparation (First 5 rows)")
print(genres.head())
print("\n")

books_merged_w_genre = books.merge(genres, left_on='Main Genre', right_on='Title', how='inner')
books_merged_w_sub_genre = books.merge(sub_genres, left_on='Sub Genre', right_on='Title', how='inner')

print("Merged Books and Genres Dataset (First 5 rows)")
print(books_merged_w_genre.head())
print("\n")

print("Merged Books and Sub Genres Dataset (First 5 rows)")
print(books_merged_w_sub_genre.head())
print("\n")

books_merged_w_genre = books_merged_w_genre.drop(columns=['Type', 'Price', 'Number of Sub-genres', 'Title_y', 'Sub Genre'])
books_merged_w_sub_genre = books_merged_w_sub_genre.drop(columns=['Type', 'Price', 'No. of Books','Title_y','Main Genre_y','Main Genre_x'])

books_merged_w_genre = books_merged_w_genre.rename(columns={'Title_x':'Title'})
books_merged_w_sub_genre = books_merged_w_sub_genre.rename(columns={'Title_x':'Title'})

books_merged_w_genre = books_merged_w_genre.drop_duplicates(subset=['Title'])
books_merged_w_sub_genre = books_merged_w_sub_genre.drop_duplicates(subset=['Title'])

print("Merged Datasets after Data Preparation (First 5 rows)")
print(books_merged_w_genre.head())
print("-----" * 40)
print(books_merged_w_sub_genre.head())
print("\n")

most_rated_books = books_merged_w_sub_genre.sort_values(by=['No. of People rated'], ascending=False)
most_rated_books = most_rated_books.drop(columns=['Rating', 'Author', 'Sub Genre','ID'])
print("Most Rated 20 Books")
print(most_rated_books.head(20))
print("\n")

highest_ratings = books_merged_w_genre.sort_values(by=['Rating'], ascending=False)
highest_ratings = highest_ratings.drop(columns=['Main Genre', 'ID', 'Author','No. of People rated'])
print("Highest Rated 20 Books")
print(highest_ratings.head(20))
print("\n")

def clean_plot_text(text):
    text = str(text)

    replacements = {
        "\x92": "'",
        "\ufffd": "",
        "\u2015": "-",
        "\u2032": "'",
        "\u2764": "",
        "\u01c0": "|",
    }

    for bad_character, replacement in replacements.items():
        text = text.replace(bad_character, replacement)

    return unicodedata.normalize("NFC", text)

def select_font(text):
    for character in str(text):
        code = ord(character)

        # Arabic, Persian and Urdu
        if (
            0x0600 <= code <= 0x06FF
            or 0x0750 <= code <= 0x077F
            or 0x08A0 <= code <= 0x08FF
        ):
            return font_properties["arabic"]

        # Devanagari
        if 0x0900 <= code <= 0x097F:
            return font_properties["devanagari"]

        # Bengali
        if 0x0980 <= code <= 0x09FF:
            return font_properties["bengali"]

        # Gujarati
        if 0x0A80 <= code <= 0x0AFF:
            return font_properties["gujarati"]

        # Tamil
        if 0x0B80 <= code <= 0x0BFF:
            return font_properties["tamil"]

        # Malayalam
        if 0x0D00 <= code <= 0x0D7F:
            return font_properties["malayalam"]

    return fm.FontProperties(family="DejaVu Sans")

plt.figure(figsize=(20,20),edgecolor='black')
plt.barh(most_rated_books['Title'].head(20), most_rated_books['No. of People rated'].head(20))
plt.title("Most Rated 20 Books",fontsize=36)
plt.xlabel('No. of People rated',fontsize=36)
plt.ylabel('Title')
plt.yticks(fontsize=20,fontweight='bold')
plt.xticks(fontsize=30)

plt.subplots_adjust(left=0.20,
    right=0.98,
    top=0.92,
    bottom=0.10)

plt.savefig('Most Rated 20 Books.png')

plt.figure(figsize=(20,20),edgecolor='black')
plt.barh(highest_ratings['Title'].head(15), highest_ratings['Rating'].head(15))
plt.title("Highest Rated 15 Books",fontsize=36)
plt.xlabel("Rating",fontsize=36)
plt.ylabel("Title",fontsize=15)
plt.yticks(fontsize=20, fontweight='bold')
plt.xticks(fontsize=30, fontweight='bold')

plt.subplots_adjust(left=0.40,
    right=0.96,
    top=0.92,
    bottom=0.10)

plt.savefig('Highest Rated 15 Books.png')

plt.show()













