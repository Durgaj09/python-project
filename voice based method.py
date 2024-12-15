# Importing Libraries
import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import joblib

# URL for Dataset
url_string = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'

# Downloading and saving the dataset
url_content = requests.get(url_string).content
with open('data.csv', 'wb') as data_file:
    data_file.write(url_content)

# Reading dataset into pandas DataFrame
df = pd.read_csv('data.csv')

# Dropping the redundant column "name"
df.drop(['name'], axis=1, inplace=True)

# Converting 'status' column to uint8
df['status'] = df['status'].astype('uint8')

# Splitting Features and Target
X = df.drop(columns=['status'])
y = df['status']

# Normalizing features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Defining the Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
predRF = rfc.predict(X_test)

print("Classification Report for Default Random Forest Classifier:\n")
print(classification_report(y_test, predRF))

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': range(100, 300, 25),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': range(1, 10),
    'random_state': range(100, 250, 50),
    'criterion': ['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
CV_rfc.fit(X_train, y_train)

print("Best Parameters from GridSearchCV:\n", CV_rfc.best_params_)

# Best Model with Tuned Hyperparameters
rfc1 = RandomForestClassifier(random_state=200, max_features='auto', n_estimators=125, max_depth=7, criterion='entropy')
rfc1.fit(X_train, y_train)
predRFC = rfc1.predict(X_test)

print("Classification Report for Tuned Random Forest Classifier:\n")
print(classification_report(y_test, predRFC))

# Plotting Confusion Matrix
conf_matrix = confusion_matrix(y_test, predRFC)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=rfc1.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Random Forest', y=1.1)
plt.show()

# ROC Curve and AUC Score
y_pred_proba = rfc1.predict_proba(X_test)[::, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="Random Forest, AUC=" + str(round(auc, 2)))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc=4)
plt.show()

# Saving the Model using joblib
joblib.dump(rfc1, 'rf_clf.pkl')
print("Random Forest Classifier model saved as 'rf_clf.pkl'")
