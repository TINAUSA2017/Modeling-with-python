
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set pandas options to display all columns in the terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# In [2]: Load Dataset
dataset = pd.read_csv('Placement_Data_Full.csv')

# In [3]: View first few rows
print("--- dataset.head() ---")
print(dataset.head())

# In [6]: Check dataset info / data types (Matching notebook In [6])
print("\n--- dataset.info() ---")
print(dataset.info())

# In [4]: Drop 'salary' and 'sl_no' columns
dataset.drop(['salary', 'sl_no'], axis=1, inplace=True)

# In [5]: Missing values checking
print("\n--- dataset.isnull().sum() ---")
print(dataset.isnull().sum())