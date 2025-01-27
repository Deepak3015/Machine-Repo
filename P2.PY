import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

print(df.head())

print("shape of the data:",df.shape)

print(df.describe())

print("missing value in the data ",df.isnull().sum())

print("species distribution:\n",df['species'].value_counts())

sepal_length = df['sepal_length'].values

mean_sepal_length = np.mean(sepal_length)

std_sepal_length  = np.std(sepal_length)

normalized_sepal_length = (sepal_length - np.min(sepal_length)) / (np.max(sepal_length) - np.min(sepal_length))
print("Normalized sepal length:\n", normalized_sepal_length[:5])


plt.figure(figsize=(8,6))

plt.scatter(df['sepal_length'],df['sepal_width'], c ="blue",alpha = 0.7)

plt.title('Sepal Length vs. Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

plt.grid(True)

plt.show()





