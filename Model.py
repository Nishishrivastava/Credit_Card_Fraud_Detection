import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# reading csv file

df = pd.read_csv('creditcard.csv')
print(df)
print(df.shape)
print(df.head(10))

#analysing dataset

print(df.info())
print(df.describe)

#how many fraud and not fraud?

class_name = {0:'Not Fraud',1:'Fraud'}
print(df.Class.value_counts().rename(index=class_name))

#train and test the dataset

from sklearn.model_selection import train_test_split
feature_names = df.iloc[:, 1:30].columns
target = df.iloc[:1, 30: ].columns
print(feature_names)
print(target)

data_features = df[feature_names]
data_targets = df[target]

X_train, X_test, y_train, y_test = train_test_split(data_features, data_targets, train_size=0.70, test_size=0.30, random_state=1)
print("Length of X_train is: {X_train}".format(X_train = len(X_train)))
print("Length of X_test is: {X_test}".format(X_test = len(X_test)))
print("Length of y_train is: {y_train}".format(y_train = len(y_train)))
print("Length of y_test is: {y_test}".format(y_test = len(y_test)))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

model = LogisticRegression()
model.fit(X_train, y_train.values.ravel())

pred = model.predict(X_test)
class_names = ['not_fraud', 'fraud']
matrix = confusion_matrix(y_test, pred)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

#Evaluation
from sklearn.metrics import f1_score, recall_score
f1_score = round(f1_score(y_test, pred), 2)
recall_score = round(recall_score(y_test, pred), 2)
print("Sensitivity/Recall for Logistic Regression Model 1 : {recall_score}".format(recall_score = recall_score))
print("F1 Score for Logistic Regression Model 1 : {f1_score}".format(f1_score = f1_score))