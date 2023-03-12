# pip install pandas
import pandas as pd
import numpy as np
# pip install matplotlib
import matplotlib.pyplot as plt
# pip install scikit-learn
from sklearn.linear_model import LogisticRegression
import pickle

# Load the CSV
dataset = pd.read_csv('diabetes.csv')
#print(dataset.head());

cols = list(dataset.columns.values)
cols = cols[1:6]
print(cols)
import seaborn as sns
for col_name in cols:
    percent_of_zero_values = (dataset[dataset[str(col_name)] == 0].shape[0])/(dataset[str(col_name)].shape[0])*100
    print("% Of Data from columns",str(col_name)," having value as 0 : ",percent_of_zero_values)
    plt.title(f"{str(col_name)} Distribution")
    sns.distplot(dataset[str(col_name)])
    plt.show()
# Replace null values
dataset['Glucose'] = dataset['Glucose'].replace(0, dataset['Glucose'].mean())
sns.distplot(dataset['Glucose'])
plt.show()
dataset['BloodPressure'] = dataset['BloodPressure'].replace(0, dataset['BloodPressure'].mean())
sns.distplot(dataset['BloodPressure'])
plt.show()
dataset['SkinThickness'] = dataset['SkinThickness'].replace(0, dataset['SkinThickness'].median())
sns.distplot(dataset['SkinThickness'])
plt.show()
dataset['Insulin'].replace(0,np.nan,inplace=True)
dataset['Insulin'].fillna(method='ffill',inplace=True)
dataset['Insulin'].fillna(method='bfill',inplace=True)
sns.distplot(dataset['Insulin'])
plt.show()
dataset['BMI'] = dataset['BMI'].replace(0, dataset['BMI'].mean())
sns.distplot(dataset['BMI'])
plt.show()
print(dataset.describe(include='all'))

#Explanation
features_listed = ['Glucose', 'BMI', 'DiabetesPedigreeFunction']
plt.figure(figsize = [20, 15])
counter = 0
for i in features_listed:
    counter += 1
    print(counter, ':', i)
    sns.displot(data = dataset, kde=True, x = dataset[str(i)], hue='Outcome')
plt.plot()
plt.show()
# Graph
#plt.scatter(dataset.temperature, dataset.prognosis)
#plt.show()

# Convert strings to numeric


# Create the Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']], dataset.Outcome)
# Save the model
with open('logistic.pk', 'wb') as f:
	pickle.dump(model, f)

# Test the model
test_Pregnancies = 6
test_Glucose = 148
test_BloodPressure = 72
test_SkinThickness = 35
test_Insulin = 0
test_BMI = 33.6
test_Pedigree = 0.627
test_Age = 50

output = model.predict_proba([[test_Pregnancies, test_Glucose, test_BloodPressure, test_SkinThickness, test_Insulin, test_BMI, test_Pedigree, test_Age]])
print("OUTCOME", "{:.2f}".format(output[0][1]))
