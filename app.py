import pandas as pd
from sklearn.impute import SimpleImputer

# 1. Load data
data = pd.read_csv("data.csv")

# 2. Remove duplicates
data = data.drop_duplicates()

# 3. Fill missing values (numerical with mean, categorical with mode)
num_cols = data.select_dtypes(include='number').columns
cat_cols = data.select_dtypes(include='object').columns

imputer_num = SimpleImputer(strategy='mean')
imputer_cat = SimpleImputer(strategy='most_frequent')

data[num_cols] = imputer_num.fit_transform(data[num_cols])
data[cat_cols] = imputer_cat.fit_transform(data[cat_cols])


# Check basic info
print(data.info())

# Check missing values
print(data.isnull().sum())

# Check distributions
print(data.describe())

# Example: value counts for categorical column
print(data['category'].value_counts())


from sklearn.preprocessing import OneHotEncoder, StandardScaler

# One-hot encode categorical
data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)

# Scale numeric
scaler = StandardScaler()
data_encoded[num_cols] = scaler.fit_transform(data_encoded[num_cols])


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Suppose target column is 'churned' (0/1)
X = data_encoded.drop('churned', axis=1)
y = data_encoded['churned']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ML model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
