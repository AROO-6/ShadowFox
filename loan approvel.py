import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

rows = 500

data = {
    "Loan_ID": [f"LP{1000+i}" for i in range(rows)],
    "Gender": np.random.choice(["Male", "Female"], rows),
    "Married": np.random.choice(["Yes", "No"], rows),
    "Dependents": np.random.choice(["0", "1", "2", "3+"], rows),
    "Education": np.random.choice(["Graduate", "Not Graduate"], rows),
    "Self_Employed": np.random.choice(["Yes", "No"], rows),
    "ApplicantIncome": np.random.randint(1500, 25000, rows),
    "CoapplicantIncome": np.random.randint(0, 15000, rows),
    "LoanAmount": np.random.randint(50, 700, rows),
    "Loan_Amount_Term": np.random.choice([120, 180, 240, 360], rows),
    "Credit_History": np.random.choice([0, 1], rows, p=[0.2, 0.8]),
    "Property_Area": np.random.choice(["Urban", "Semiurban", "Rural"], rows),
    "Loan_Status": np.random.choice(["Y", "N"], rows, p=[0.7, 0.3])
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV (for Colab download/use)
df.to_csv("loan_approval_dataset.csv", index=False)

# Display first 5 rows
df.head()

data = pd.read_csv("loan_approval_dataset.csv")
data.head()

data.info()
data.describe()
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col])

y = le.fit_transform(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Loan Approval Confusion Matrix")
plt.show()


