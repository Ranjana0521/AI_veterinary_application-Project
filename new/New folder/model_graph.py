import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -----------------------------
# Load the dataset
# -----------------------------
CSV_PATH = r"C:\Users\hp\OneDrive\Desktop\vet mini\Animal_Health_Record_500.csv"
df = pd.read_csv(CSV_PATH)

print("🔹 Columns found:", df.columns.tolist())

# -----------------------------
# Handle BP (Blood Pressure)
# -----------------------------
if 'BP' in df.columns:
    # Split BP into two columns
    bp_split = df['BP'].astype(str).str.split('/', expand=True)
    df['BP_Systolic'] = pd.to_numeric(bp_split[0], errors='coerce')
    df['BP_Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce')
    df.drop(columns=['BP'], inplace=True)

# -----------------------------
# Drop unnecessary text columns
# -----------------------------
drop_cols = ['Animal ID', 'Name', 'Species', 'Breed', 'Symptom 1', 'Symptom 2', 'others']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# -----------------------------
# Encode categorical columns
# -----------------------------
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# -----------------------------
# Drop rows with missing values
# -----------------------------
df = df.dropna()

# -----------------------------
# Verify data types
# -----------------------------
non_numeric_cols = df.select_dtypes(exclude='number').columns.tolist()
if non_numeric_cols:
    print("⚠️ Still non-numeric columns:", non_numeric_cols)
    for col in non_numeric_cols:
        print(df[col].unique())

# -----------------------------
# Split features and target
# -----------------------------
if 'Health Status' not in df.columns:
    raise KeyError("❌ 'Health Status' column not found in CSV!")

X = df.drop(columns=['Health Status'])
y = df['Health Status']

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train both models
# -----------------------------
rf = RandomForestClassifier()
svm = SVC()

rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

# -----------------------------
# Evaluate models
# -----------------------------
rf_acc = accuracy_score(y_test, rf.predict(X_test))
svm_acc = accuracy_score(y_test, svm.predict(X_test))

# -----------------------------
# Plot accuracy comparison
# -----------------------------
plt.bar(['Random Forest', 'SVM'], [rf_acc, svm_acc])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()

print(f"✅ Random Forest Accuracy: {rf_acc:.2f}")
print(f"✅ SVM Accuracy: {svm_acc:.2f}")
