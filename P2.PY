# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
# df = pd.read_csv(url)

# print(df.head())

# print("shape of the data:",df.shape)

# print(df.describe())

# print("missing value in the data ",df.isnull().sum())

# print("species distribution:\n",df['species'].value_counts())

# sepal_length = df['sepal_length'].values

# mean_sepal_length = np.mean(sepal_length)

# std_sepal_length  = np.std(sepal_length)

# normalized_sepal_length = (sepal_length - np.min(sepal_length)) / (np.max(sepal_length) - np.min(sepal_length))
# print("Normalized sepal length:\n", normalized_sepal_length[:5])


# plt.figure(figsize=(8,6))

# plt.scatter(df['sepal_length'],df['sepal_width'], c ="blue",alpha = 0.7)

# plt.title('Sepal Length vs. Sepal Width')
# plt.xlabel('Sepal Length (cm)')
# plt.ylabel('Sepal Width (cm)')

# plt.grid(True)

# plt.show()
import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate synthetic data
applicant_income = np.random.randint(20000, 120000, size=n_samples)  # Annual income in USD
credit_score = np.random.randint(300, 850, size=n_samples)  # Credit score
loan_amount = np.random.randint(5000, 100000, size=n_samples)  # Loan amount in USD

# Generate loan status based on some rules (for simplicity)
# Let's assume that loan approval depends on:
# 1. Applicant income > 40000
# 2. Credit score > 600
# 3. Loan amount < 0.5 * applicant income
loan_status = np.where(
    (applicant_income > 40000) & (credit_score > 600) & (loan_amount < 0.5 * applicant_income),
    "Approved",  # Loan approved
    "Rejected"   # Loan rejected
)

# Create a DataFrame
data = pd.DataFrame({
    'Applicant Income': applicant_income,
    'Credit Score': credit_score,
    'Loan Amount': loan_amount,
    'Loan Status': loan_status
})

# Display the first few rows of the dataset
print(data.head())

# Save the dataset to a CSV file (optional)
data.to_csv('loan_application_data.csv', index=False)




