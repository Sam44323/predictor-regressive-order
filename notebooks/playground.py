import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# pandas — for loading and working with tabular data
# matplotlib — for basic plots
# seaborn — for nicer statistical visualizations

df = pd.read_csv("../data/dataset.csv")

print(df.shape)
# Returns (rows, columns) — tells you how big of the dataset

print(df.head())
# Shows first 5 rows — gives you a feel for the data

print(df.dtypes)
# Shows data type of each column
# int64/float64 = numeric, object = text/categorical

print(df.isnull().sum())
# For each column, counts how many values are missing (NaN)
# A column with 0 means no missing values — good
# A column with many missing values needs attention

df["selling_price"].describe()
# Shows count, mean, min, max, std, and quartiles
# If max is way higher than 75th percentile — there are outliers

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
# corr() calculates correlation between all numeric columns
# annot=True shows the actual numbers inside each cell
# cmap="coolwarm" — red means positive correlation, blue means negative
plt.title("Correlation between numeric features")
plt.show()

plt.scatter(df["km_driven"], df["selling_price"], alpha=0.4)
# alpha=0.4 makes dots semi-transparent so overlapping points are visible
# x-axis = km driven, y-axis = price we want to predict
plt.xlabel("KM Driven")
plt.ylabel("Selling Price")
plt.title("KM Driven vs Selling Price")
plt.show()
# If you see a downward trend — our intuition is correct
# If you see random scatter — km_driven might not be a strong feature