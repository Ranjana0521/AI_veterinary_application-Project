import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Initialize the Flask application
app = Flask(__name__)

# ------------------- Machine Learning Section -------------------

# 1. Load your CSV data
try:
    df = pd.read_csv('Animal_Health_Record_500.csv')
except FileNotFoundError:
    print("ERROR: Animal_Health_Record_500.csv not found.")
    exit()

# 2. Create binary "Outcome" target
fatal_diseases = ['Rabies', 'Anthrax', 'PPR (Peste des petits ruminants)']
df['Outcome'] = df['Disease'].apply(lambda x: 'Will Not Live' if x in fatal_diseases else 'Will Live')

# 3. Fill missing Symptom values
df['Symptom 1'] = df['Symptom 1'].fillna('None')
df['Symptom 2'] = df['Symptom 2'].fillna('None')

# 4. Feature Engineering for BP (safe handling)
bp_split = df['BP'].str.split('/', expand=True)
df['BP_Systolic'] = pd.to_numeric(bp_split[0], errors='coerce')
df['BP_Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce')

# Fill missing BP values with median
df['BP_Systolic'] = df['BP_Systolic'].fillna(df['BP_Systolic'].median())
df['BP_Diastolic'] = df['BP_Diastolic'].fillna(df['BP_Diastolic'].median())

# 5. Ensure Heart Rate is numeric and fill NaNs
df['Heart Rate (bpm)'] = pd.to_numeric(df['Heart Rate (bpm)'], errors='coerce').fillna(df['Heart Rate (bpm)'].median())

# 6. Define Features and Targets
numerical_features = ['Heart Rate (bpm)', 'BP_Systolic', 'BP_Diastolic']
categorical_features = ['Species', 'Breed', 'Sex', 'Health Status', 'Symptom 1', 'Symptom 2']
all_features = numerical_features + categorical_features

X = df[all_features]
y_survival = df['Outcome']       # Model 1 target
y_disease = df['Disease']         # Model 2 target

# 7. Preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 8. Train Survival Model
print("Training Survival Model...")
model_survival = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
model_survival.fit(X, y_survival)
print("Survival Model trained.")

# 9. Train Disease Model
print("Training Disease Model...")
model_disease = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
model_disease.fit(X, y_disease)
print("Disease Model trained. App is ready!")

# ------------------- Flask Web Routes -------------------

@app.route('/')
def home():
    """Render main input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        species = request.form['species']
        breed = request.form['breed']
        sex = request.form['sex']
        bp_str = request.form['bp']
        heart_rate = request.form['heart_rate']
        health_status = request.form['health_status']
        symptom1 = request.form['symptom_1']
        symptom2 = request.form['symptom_2']

        # Process Heart Rate safely
        try:
            heart_rate = int(heart_rate)
        except:
            heart_rate = int(df['Heart Rate (bpm)'].median())

        # Process BP safely
        try:
            bp_systolic, bp_diastolic = map(int, bp_str.split('/'))
        except:
            bp_systolic = int(df['BP_Systolic'].median())
            bp_diastolic = int(df['BP_Diastolic'].median())

        # Create DataFrame for prediction
        input_data = pd.DataFrame([[
            heart_rate, bp_systolic, bp_diastolic,
            species, breed, sex, health_status, symptom1, symptom2
        ]], columns=all_features)

        # --- Make Predictions ---
        prediction_survive = model_survival.predict(input_data)[0]
        probabilities_survive = model_survival.predict_proba(input_data)[0]
        live_index = np.where(model_survival.classes_ == 'Will Live')[0][0]
        chance_of_living = round(probabilities_survive[live_index] * 100, 2)

        prediction_disease = model_disease.predict(input_data)[0]

        return render_template('index.html',
                               prediction_disease=f'Predicted Disease: {prediction_disease}',
                               prediction_survival=f'Predicted Outcome: {prediction_survive}',
                               living_chance=f'Chance of Living: {chance_of_living}%',
                               show_result=True)

    except Exception as e:
        return render_template('index.html',
                               prediction_disease=f'Error: {e}',
                               show_result=True)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
