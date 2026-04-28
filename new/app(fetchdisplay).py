
from flask import Flask, render_template_string, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import os 
import re 

app = Flask(__name__)

# ---------------------------
# Path to uploaded CSV (local)
# ---------------------------
CSV_FILE = r"c:/Users/hp/OneDrive/Desktop/new/Animal_Health_Record_500.csv"

# ---------------------------
# Utility: ensure CSV exists
# ---------------------------
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"CSV dataset not found at {CSV_FILE}. Please upload the file to this path.")

# =========================================================
# 🧠 MODEL 1 — Health Status Prediction (simple RF))
# =========================================================
def train_model():
    df_local = pd.read_csv(CSV_FILE)
    required = ["BP", "Heart Rate (bpm)", "Age (years)", "Health Status"]
    for col in required:
        if col not in df_local.columns:
            raise KeyError(f"Missing column: {col} in CSV. Required: {required}")

    # helper to extract systolic
    def extract_systolic(bp_str):
        try:
            return float(str(bp_str).split("/")[0])
        except:
            return 0.0

    df_local["BP_num"] = df_local["BP"].apply(extract_systolic)
    df_local["Heart Rate (bpm)"] = pd.to_numeric(df_local["Heart Rate (bpm)"], errors="coerce").fillna(0)
    df_local["Age (years)"] = pd.to_numeric(df_local["Age (years)"], errors="coerce").fillna(0)

    X = df_local[["BP_num", "Heart Rate (bpm)", "Age (years)"]]
    y = df_local["Health Status"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return clf, acc

# Train base health model (will raise if CSV missing required columns)
model, accuracy = train_model()

# =========================================================
# 🐾 Symptom → Disease Mapping
# =========================================================
disease_map = {
    "fever": "Bacterial Infection",
    "cough": "Respiratory Infection",
    "diarrhea": "Gastroenteritis",
    "vomit": "Food Poisoning",
    "wound": "Skin Infection",
    "cold": "Viral Fever",
    "rashes": "Allergic Reaction",
    "eye infection": "Conjunctivitis",
    "weakness": "Anemia",
    "loss of appetite": "Digestive Disorder"
}

# =========================================================
# Templates (embedded so this is a single-file app)
# =========================================================

display_template = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Animal Record</title>
<style>
body {
  font-family: 'Poppins', sans-serif;
  background: linear-gradient(135deg, #c9ffbf, #ffafbd);
  text-align: center;
  padding-top: 40px;
  overflow-x: hidden;
  position: relative;
}
body::before {
  content: "🐄 🐕 🐈 🐘 🐓 🦌";
  position: absolute;
  top: 10%;
  left: 50%;
  transform: translateX(-50%);
  font-size: 80px;
  opacity: 0.15;
  animation: float 8s ease-in-out infinite alternate;
}
@keyframes float {
  from { transform: translate(-50%, 0px); }
  to { transform: translate(-50%, 20px); }
}
.card {
  display: inline-block;
  background: white;
  padding: 30px 40px;
  border-radius: 25px;
  box-shadow: 0 6px 20px rgba(0,0,0,0.2);
  max-width: 900px;
  width: 92%;
  text-align:left;
}
h2 { text-align:center; }
.vaccine-box {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 20px;
  margin-top: 15px;
  padding: 14px;
  background: #f1faff;
  border-radius: 12px;
}
.v-section {
  flex: 1 1 30%;
  min-width: 220px;
  padding: 12px;
  background: white;
  border-radius: 10px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}
.v-section h4 { margin-top: 0; color:#0077b6; }
.add-btn {
  margin-top: 10px;
  padding: 8px 12px;
  border: none;
  font-size: 14px;
  background: #38b000;
  color: white;
  border-radius: 8px;
  cursor: pointer;
}
.add-btn:hover { background: #2a8500; }
.hidden-box { display:none; margin-top:10px; }
button.save-btn {
  background:#0077b6; color:white; padding:8px 16px; border:none; border-radius:8px; cursor:pointer;
}
button.save-btn:hover { background:#023e8a; }
.info-row { margin:6px 0; }
.small { font-size:0.9em; color:#555; }
.center { text-align:center; }
.meta { font-size:0.85em; color:#555; }
.star { color: gold; font-size:20px; vertical-align:middle; margin-left:8px; }
</style>
</head>
<body>
<div style="font-size:70px; text-align:center;">🐴🐘🐤🦌</div>
<h2>Animal Health Record</h2>

{% if found %}
  <div class="card">
    <h3 class="center">Animal ID: {{ data['Animal ID'] }}
      {% if data.get('Special Care','').lower() == 'yes' %}<span class="star">⭐</span>{% endif %}
    </h3>

    <div class="info-row"><b>Name:</b> {{ data['Name'] }}</div>
    <div class="info-row"><b>Breed:</b> {{ data['Breed'] }}</div>
    <div class="info-row"><b>Age:</b> {{ data.get('Age (years)', data.get('Age','')) }}</div>
    <div class="info-row"><b>BP:</b> {{ data['BP'] }}</div>
    <div class="info-row"><b>Heart Rate:</b> {{ data.get('Heart Rate (bpm)', data.get('Heart Rate','')) }}</div>
    <div class="info-row"><b>Oxygen Saturation:</b> {{ data.get('Oxygen Saturation (%)', data.get('Oxygen Saturation','')) }}</div>
    <div class="info-row"><b>Symptoms:</b> {{ data.get('Symptom 1','') }}, {{ data.get('Symptom 2','') }}</div>
    <div class="info-row"><b>Detected Disease:</b> 🧬 <b>{{ prediction }}</b></div>
    <div class="info-row small"><b>Model Accuracy:</b> {{ accuracy }}%</div>

    <hr>

    <div class="vaccine-box">
      <!-- Vaccination History -->
      <div class="v-section">
        <h4>💉 Vaccination History</h4>
        <p><b>1)</b> {{ data.get('Vaccination 1','—') }}</p>
        <p><b>2)</b> {{ data.get('Vaccination 2','—') }}</p>

        <button class="add-btn" onclick="document.getElementById('vaccineForm').style.display='block'; this.style.display='none';">
          + Add Vaccination
        </button>

        <form id="vaccineForm" class="hidden-box" method="POST" action="{{ url_for('add_vaccine') }}">
          <input type="hidden" name="animal_id" value="{{ data['Animal ID'] }}">
          <input type="text" name="new_vaccine" placeholder="Enter Vaccine Name" required style="width:100%; padding:8px; margin-top:8px;"><br>
          <input type="date" name="vaccine_date" required style="width:100%; padding:8px; margin-top:8px;"><br>
          <div style="margin-top:8px;">
            <button type="submit" class="add-btn">Save Vaccine</button>
            <button type="button" class="save-btn" onclick="document.getElementById('vaccineForm').style.display='none'; document.querySelector('.add-btn').style.display='inline-block';">Cancel</button>
          </div>
        </form>
      </div>

      <!-- Recommended Vaccines -->
      <div class="v-section">
        <h4>📌 Recommended Vaccines</h4>
        <p><b>Large Animals:</b></p>
        <ul>
          <li>FMD (Foot & Mouth Disease)</li>
          <li>HS (Hemorrhagic Septicemia)</li>
          <li>BQ (Black Quarter)</li>
          <li>Anthrax</li>
          <li>Tetanus (Horses)</li>
        </ul>

        <p><b>Small Animals:</b></p>
        <ul>
          <li>Rabies</li>
          <li>Parvo</li>
          <li>Distemper</li>
          <li>Leptospirosis</li>
          <li>Feline Panleukopenia</li>
        </ul>
      </div>
    </div>

    <hr>

    <!-- Update Symptoms Form (also saves Special Care) -->
    <form method="POST" style="margin-top:12px;">
      <input type="text" name="Symptom1" placeholder="Update Symptom 1" style="width:48%; padding:8px;" value="{{ data.get('Symptom 1','') }}">
      <input type="text" name="Symptom2" placeholder="Update Symptom 2" style="width:48%; padding:8px; float:right;" value="{{ data.get('Symptom 2','') }}"><br><br>
      <input type="text" name="suggestion" placeholder="Doctor Suggestion" style="width:98%; padding:8px;" value="{{ data.get('Doctor Suggestion','') }}"><br><br>

      <label style="font-size:0.95em; margin-right:8px;">
        <input type="checkbox" name="special_care" value="Yes" {% if data.get('Special Care','').lower()=='yes' %}checked{% endif %}>
        Mark Special Care
      </label>
      <br><br>

      <button type="submit" class="save-btn">💾 Save</button>
    </form>

    <br>
    <!-- Generate PDF placed under Save as requested -->
    <form method="GET" action="{{ url_for('generate_pdf') }}">
      <input type="hidden" name="animal_id" value="{{ data['Animal ID'] }}">
      <button type="submit" class="save-btn" style="background:#6a4c93;">📄 Generate PDF Report</button>
    </form>

  </div>
{% else %}
  <div class="card center">
    <h3>⚠️ No record found for Animal ID: {{ animal_id }}</h3>
    <p><a href="{{ url_for('home') }}">Back to Home</a></p>
  </div>
{% endif %}

</body>
</html>
"""

index_template = """ (UNCHANGED - kept from your original code) """
# We'll render your index_template and scan_template via render_template_string below (they were defined earlier in your original file).
# For brevity in this merged file, we will reuse the same index_template and scan_template strings the user had earlier.
# If you prefer, you can replace the placeholders below with your previously defined index_template and scan_template content.

# For compatibility, re-use the index_template and scan_template from your prior code:
index_template = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Health Chatbot</title>

  <style>
    body { 
      font-family: 'Poppins', sans-serif; 
      text-align:center; 
      padding:40px; 
      background:linear-gradient(135deg,#f6d365,#fda085); 
      position: relative;
      overflow: hidden;
    }

    /* Background animal emojis */
    .emoji {
      position: absolute;
      font-size: 70px;
      opacity: 0.15;
      pointer-events: none;
      user-select: none;
    }

    .card { 
      background:white; 
      display:inline-block; 
      padding:20px 30px; 
      border-radius:12px; 
      box-shadow:0 6px 18px rgba(0,0,0,0.12); 
      position: relative;
      z-index: 2;
    }

    input, select { padding:8px; margin:6px; width:220px; }
    button { padding:8px 14px; background:#0077b6; color:white; border:none; border-radius:8px; cursor:pointer; }
  </style>
</head>

<body>

  <!-- Static animal emojis -->
  <div class="emoji" style="top:10%; left:5%;">🐶</div>
  <div class="emoji" style="top:20%; right:8%;">🐱</div>
  <div class="emoji" style="bottom:15%; left:10%;">🐰</div>
  <div class="emoji" style="bottom:20%; right:15%;">🐮</div>
  <div class="emoji" style="top:60%; left:50%;">🐹</div>

  <div class="card">
    <h2>Animal Health Chatbot</h2>

    <form method="post" action="{{ url_for('predict') }}">
      <input name="species" placeholder="Species" required><br>
      <input name="breed" placeholder="Breed" required><br>
      <input name="sex" placeholder="Sex" required><br>
      <input name="bp" placeholder="BP (systolic/diastolic) e.g. 120/80" required><br>
      <input name="heart_rate" placeholder="Heart rate (bpm)" required><br>
      <input name="health_status" placeholder="Health Status" required><br>
      <input name="symptom_1" placeholder="Symptom 1"><br>
      <input name="symptom_2" placeholder="Symptom 2"><br>
      <button type="submit">Predict</button>
    </form>

    {% if show_result %}
      <hr>
      <p>{{ prediction_disease }}</p>
      <p>{{ prediction_survival }}</p>
      <p>{{ living_chance }}</p>
    {% endif %}

    <p style="margin-top:12px;"><a href="{{ url_for('home') }}">Back</a></p>
  </div>

</body>
</html>

"""

scan_template = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Scan QR</title>

  <style>
    body { 
      font-family:'Poppins', sans-serif; 
      text-align:center; 
      padding:80px; 
      background:linear-gradient(135deg,#c9ffbf,#ffafbd); 
    }
    .card { 
      display:inline-block; 
      background:white; 
      padding:20px 30px; 
      border-radius:12px; 
      box-shadow:0 6px 18px rgba(0,0,0,0.12); 
    }
    button { 
      padding:8px 14px; 
      background:#0077b6; 
      color:white; 
      border:none; 
      border-radius:8px; 
      cursor:pointer; 
    }
    #reader {
      width: 320px;
      margin: 0 auto;
    }
  </style>

  <!-- QR Scanner Library -->
  <script src="https://unpkg.com/html5-qrcode" type="text/javascript"></script>
</head>

<body>
  <div class="card">
    <h3>QR Scanner</h3>

    <!-- CAMERA PREVIEW BOX -->
    <div id="reader"></div>

    <p id="status" style="margin-top:10px; color:#333;">Initializing camera...</p>

    <button onclick="window.location.href='{{ url_for('dashboard') }}'">Back</button>
  </div>

<script>
let scanner;

function onScanSuccess(decodedText) {
    console.log("QR Detected:", decodedText);

    // Extract ID from QR (numbers only)
    const match = decodedText.match(/(\\d+)/);


    if (match) {
        const animalId = match[1];
        window.location.href = "/display?animal_id=" + animalId;
    } else {
        document.getElementById("status").innerText = "QR scanned but no ID found.";
    }

    scanner.stop().catch(()=>{});
}

function startScanner() {
    scanner = new Html5Qrcode("reader");

    scanner.start(
        { facingMode: "environment" }, 
        { fps: 10, qrbox: { width: 250, height: 250 }},
        onScanSuccess,
        () => {} // ignore failure
    )
    .then(() => {
        document.getElementById("status").innerText = "Camera Activated — Scan the QR Code";
    })
    .catch(err => {
        document.getElementById("status").innerText = "Camera Error: " + err;
    });
}

document.addEventListener("DOMContentLoaded", startScanner);
</script>

</body>
</html>
"""
@app.route('/scan')
def scan_page():
    return render_template('scan.html')



VALID_DOCTORS = {"doctor123": {"name": "Default Doctor", "password": "admin@123"}}

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        doctor_name = request.form.get("doctorName", "").strip()
        doctor_id = request.form.get("doctorId", "").strip()
        password = request.form.get("doctorPassword", "")

        # Password validation: 6+ chars and special char
        if len(password) < 6 or not re.search(r'[^A-Za-z0-9]', password):
            return "<h3 style='color:red;'>Invalid password</h3><a href='/login'>Try again</a>"

        # Existing doctor
        if doctor_id in VALID_DOCTORS:
            if VALID_DOCTORS[doctor_id]["password"] == password:
                return redirect(url_for("dashboard"))
            else:
                return "<h3 style='color:red;'>Incorrect password</h3><a href='/login'>Try again</a>"

        # New doctor registration
        VALID_DOCTORS[doctor_id] = {"name": doctor_name, "password": password}
        return redirect(url_for("dashboard"))

    # Login page with animated emoji background
    return """
    <html>
    <head>
    <style>
    body {
        margin:0; padding:0; height:100vh; font-family: 'Poppins', sans-serif;
        display:flex; justify-content:center; align-items:center;
        background: linear-gradient(135deg, #c9ffbf, #ffafbd);
        overflow:hidden;
    }
    .emoji-background {
        position: fixed; width: 100%; height: 100%; top: 0; left: 0;
        pointer-events: none; z-index: 0;
    }
    .emoji {
        position: absolute; font-size: 50px; animation: float 10s linear infinite;
        opacity: 0.2;
    }
    @keyframes float {
        0% { transform: translateY(100vh); }
        100% { transform: translateY(-10%); }
    }
    .login-box {
        position: relative; z-index: 1;
        background: rgba(255,255,255,0.95); padding: 30px 40px;
        border-radius: 20px; box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        text-align:center; width: 350px;
    }
    input {width:100%; padding:12px; margin:10px 0; border-radius:8px; border:1px solid #ccc;}
    button {width:100%; padding:12px; margin-top:12px; border:none; border-radius:8px; cursor:pointer;
            font-weight:bold; background:#38b000; color:white; font-size:16px;}
    button:hover {background:#2a8500;}
    a {text-decoration:none; color:#0077b6; font-weight:bold; display:block; margin-top:12px;}
    </style>
    <script>
    const emojis = ["🐄","🐕","🐈","🐘","🐓","🦌","🦙","🐎"];
    window.onload = function() {
        const container = document.createElement('div');
        container.className = 'emoji-background';
        document.body.appendChild(container);
        for(let i=0; i<20; i++) {
            const span = document.createElement('span');
            span.className = 'emoji';
            span.innerText = emojis[Math.floor(Math.random()*emojis.length)];
            span.style.left = Math.random()*100 + 'vw';
            span.style.fontSize = (30 + Math.random()*50) + 'px';
            span.style.animationDuration = (5 + Math.random()*10) + 's';
            container.appendChild(span);
        }
    }
    </script>
    </head>
    <body>
    <div class="login-box">
        <h2>Doctor Login</h2>
        <form method="POST">
            <input type="text" name="doctorName" placeholder="Doctor Name" required><br>
            <input type="text" name="doctorId" placeholder="Doctor ID" required><br>
            <input type="password" name="doctorPassword" placeholder="Password" required><br>
            <button type="submit">Login</button>
        </form>
    </div>
    </body>
    </html>
    """


# =========================================================
@app.route("/dashboard")
def dashboard():
    return """
    <html>
    <head>
    <style>
    body {
        margin:0; padding:0; font-family:Poppins,sans-serif;
        text-align:center; min-height:100vh;
        background: linear-gradient(135deg, #c9ffbf, #ffafbd);
        overflow:hidden;
    }
    .emoji-background {
        position: fixed; width: 100%; height: 100%; top: 0; left: 0;
        pointer-events: none; z-index: 0;
    }
    .emoji {
        position: absolute; font-size: 50px; animation: float 10s linear infinite;
        opacity: 0.2;
    }
    @keyframes float {
        0% { transform: translateY(100vh); }
        100% { transform: translateY(-10%); }
    }
    .dashboard-box {position:relative; z-index:1; padding:50px;}
    a {margin:0 10px; text-decoration:none; font-weight:bold; color:#0077b6;}
    input {padding:10px; margin:10px; border-radius:8px; border:1px solid #ccc;}
    button {padding:10px 20px; border:none; border-radius:8px; background:#38b000; color:white; cursor:pointer;}
    button:hover {background:#2a8500;}
    </style>
    <script>
    const emojis = ["🐄","🐕","🐈","🐘","🐓","🦌","🦙","🐎"];
    window.onload = function() {
        const container = document.createElement('div');
        container.className = 'emoji-background';
        document.body.appendChild(container);
        for(let i=0; i<20; i++) {
            const span = document.createElement('span');
            span.className = 'emoji';
            span.innerText = emojis[Math.floor(Math.random()*emojis.length)];
            span.style.left = Math.random()*100 + 'vw';
            span.style.fontSize = (30 + Math.random()*50) + 'px';
            span.style.animationDuration = (5 + Math.random()*10) + 's';
            container.appendChild(span);
        }
    }
    </script>
    </head>
    <body>
    <div class="dashboard-box">
        <h1>Animal Health Dashboard</h1>
        <p>
            <a href='/display_all'>View All Records</a> | 
            <a href='/scan'>Scan</a> | 
            <a href='/chat'>Chatbot</a> | 
            <a href='/add_vaccine'>Add Vaccine</a> | 
            <a href='/login'>Logout</a>
        </p>
        <form action='/display' method='get'>
            <input type='text' name='animal_id' placeholder='Enter Animal ID' required>
            <button type='submit'>View Record</button>
        </form>
    </div>
    </body>
    </html>
    """

# =========================================================
# Display record and allow updates (symptoms + vaccines + special care)
# =========================================================
@app.route('/display', methods=['GET', 'POST'])
def display():
    animal_id = request.args.get('animal_id')
    if not animal_id:
        return "<h3>No Animal ID provided!</h3>"

    df = pd.read_csv(CSV_FILE)

    # Ensure optional columns exist
    for col in ["Doctor Suggestion", "Vaccination 1", "Vaccination 2", "Special Care"]:
        if col not in df.columns:
            df[col] = ""

    # find record
    record_index = df.index[df["Animal ID"].astype(str) == str(animal_id)]
    if record_index.empty:
        return render_template_string(display_template, found=False, animal_id=animal_id)

    idx = record_index[0]

    # -------------------------------
    # Update symptoms (POST)
    # -------------------------------
    if request.method == "POST" and request.form.get("new_vaccine") is None:
        new_symptom1 = request.form.get("Symptom1", "").strip()
        new_symptom2 = request.form.get("Symptom2", "").strip()
        suggestion = request.form.get("suggestion", "").strip()
        special_checked = request.form.get("special_care")

        df.at[idx, "Symptom 1"] = new_symptom1
        df.at[idx, "Symptom 2"] = new_symptom2
        df.at[idx, "Doctor Suggestion"] = suggestion

        # Save special care as Yes/No to CSV
        if special_checked:
            df.at[idx, "Special Care"] = "Yes"
        else:
            df.at[idx, "Special Care"] = "No"

        # Clean NaNs and save
        df = df.replace({np.nan: ""})
        df.to_csv(CSV_FILE, index=False)
        return redirect(url_for('display', animal_id=animal_id))

    # -------------------------------
    # Prepare data for display
    # -------------------------------
    data = df.iloc[idx].to_dict()

    # Clean NaN / "nan" => blank strings
    for k, v in data.items():
        if pd.isna(v) or str(v).lower() == "nan":
            data[k] = ""
        else:
            data[k] = v

    # -------------------------------
    # Disease prediction logic
    # -------------------------------
    try:
        bp_value = float(str(data.get("BP", "0")).split("/")[0])
        hr = float(data.get("Heart Rate (bpm)", 0) or data.get("Heart Rate", 0))
        age = float(data.get("Age (years)", 0) or data.get("Age", 0))
        sym1 = str(data.get("Symptom 1", "")).lower()
        sym2 = str(data.get("Symptom 2", "")).lower()

        disease = None
        for k, v in disease_map.items():
            if k in sym1 or k in sym2:
                disease = v
                break

        prediction = f"{disease} 🔴" if disease else f"{model.predict([[bp_value, hr, age]])[0]} 🟢"

    except Exception as e:
        print("⚠️ Prediction Error:", e)
        prediction = "Healthy 🟢"

    return render_template_string(
        display_template,
        found=True,
        data=data,
        prediction=prediction,
        accuracy=round(accuracy * 100, 2)
    )

# =========================================================
# Add Vaccine route - shifts Vaccine1 -> Vaccine2 and stores new vaccine with date
# =========================================================
@app.route('/add_vaccine', methods=['POST'])
def add_vaccine():
    animal_id = request.form.get('animal_id')
    new_vaccine = request.form.get('new_vaccine', '').strip()
    vaccine_date = request.form.get('vaccine_date', '').strip()

    if not animal_id or not new_vaccine or not vaccine_date:
        return "Missing data", 400

    df = pd.read_csv(CSV_FILE)

    # Ensure vaccine columns exist
    if "Vaccination 1" not in df.columns:
        df["Vaccination 1"] = ""
    if "Vaccination 2" not in df.columns:
        df["Vaccination 2"] = ""

    record_index = df.index[df["Animal ID"].astype(str) == str(animal_id)]
    if record_index.empty:
        return f"No record found for Animal ID {animal_id}", 404

    idx = record_index[0]

    # Shift Vaccine1 -> Vaccine2 and add new Vaccine1
    previous_v1 = df.at[idx, "Vaccination 1"] if pd.notna(df.at[idx, "Vaccination 1"]) else ""
    df.at[idx, "Vaccination 2"] = previous_v1
    df.at[idx, "Vaccination 1"] = f"{new_vaccine} ({vaccine_date})"

    df = df.replace({np.nan: ""})
    df.to_csv(CSV_FILE, index=False)

    return redirect(url_for('display', animal_id=animal_id))

# =========================================================
# Generate PDF (plain text) and return as download
# =========================================================
@app.route('/generate_pdf', methods=['GET'])
def generate_pdf():
    animal_id = request.args.get('animal_id')
    if not animal_id:
        return "Missing animal_id", 400

    df = pd.read_csv(CSV_FILE)
    record_index = df.index[df["Animal ID"].astype(str) == str(animal_id)]
    if record_index.empty:
        return f"No record found for Animal ID {animal_id}", 404

    idx = record_index[0]
    data = df.iloc[idx].to_dict()

    # Clean data
    for k, v in data.items():
        if pd.isna(v) or str(v).lower() == "nan":
            data[k] = ""
        else:
            data[k] = str(v)

    # Create PDF
    folder = os.path.dirname(CSV_FILE)
    pdf_name = f"Animal_{animal_id}_report.pdf"
    pdf_path = os.path.join(folder, pdf_name)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []

    story.append(Paragraph(f"<b>Animal Health Report — ID: {animal_id}</b>", styles['Title']))
    story.append(Spacer(1, 12))

    lines = [
        ("Name", data.get('Name','')),
        ("Breed", data.get('Breed','')),
        ("Age (years)", data.get('Age (years)','') or data.get('Age','')),
        ("BP", data.get('BP','')),
        ("Heart Rate (bpm)", data.get('Heart Rate (bpm)','') or data.get('Heart Rate','')),
        ("Oxygen Saturation (%)", data.get('Oxygen Saturation (%)','') or data.get('Oxygen Saturation','')),
        ("Symptoms", f"{data.get('Symptom 1','')}{', ' + data.get('Symptom 2','') if data.get('Symptom 2','') else ''}"),
        ("Detected Disease", request.args.get('prediction','') or data.get('Detected Disease','') or data.get('Disease','')),
        ("Model Accuracy", f"{round(accuracy*100,2)}%"),
        ("Vaccination 1", data.get('Vaccination 1','')),
        ("Vaccination 2", data.get('Vaccination 2','')),
        ("Doctor Suggestion", data.get('Doctor Suggestion','')),
        ("Special Care", data.get('Special Care',''))
    ]

    for label, text in lines:
        story.append(Paragraph(f"<b>{label}:</b> {text}", styles['Normal']))
        story.append(Spacer(1, 8))

    doc.build(story)

    return send_file(pdf_path, as_attachment=True)

# =========================================================
# Scan page (placeholder)
# =========================================================
@app.route('/scan')
def scan_page():
    return render_template_string(scan_template)

# =========================================================
# Chatbot model training (survival + disease) — uses pipelines
# (unchanged from your original code)
# =========================================================
# Load chat df
df_chat = pd.read_csv(CSV_FILE)

# fill critical columns safely
if 'Disease' not in df_chat.columns:
    df_chat['Disease'] = 'Unknown'
if 'Symptom 1' not in df_chat.columns:
    df_chat['Symptom 1'] = 'None'
if 'Symptom 2' not in df_chat.columns:
    df_chat['Symptom 2'] = 'None'

fatal_diseases = ['Rabies', 'Anthrax', 'PPR (Peste des petits ruminants)']
df_chat['Outcome'] = df_chat['Disease'].apply(lambda x: 'Will Not Live' if x in fatal_diseases else 'Will Live')

# Split BP safely
bp_split = df_chat['BP'].astype(str).str.split('/', expand=True)
df_chat['BP_Systolic'] = pd.to_numeric(bp_split[0], errors='coerce')
df_chat['BP_Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce')
df_chat['BP_Systolic'] = df_chat['BP_Systolic'].fillna(df_chat['BP_Systolic'].median())
df_chat['BP_Diastolic'] = df_chat['BP_Diastolic'].fillna(df_chat['BP_Diastolic'].median())
df_chat['Heart Rate (bpm)'] = pd.to_numeric(df_chat['Heart Rate (bpm)'], errors='coerce').fillna(df_chat['Heart Rate (bpm)'].median())

numerical_features = ['Heart Rate (bpm)', 'BP_Systolic', 'BP_Diastolic']
categorical_features = ['Species', 'Breed', 'Sex', 'Health Status', 'Symptom 1', 'Symptom 2']
all_features = numerical_features + categorical_features

# Ensure categorical columns exist in df_chat (fill with 'Unknown' if missing)
for c in categorical_features:
    if c not in df_chat.columns:
        df_chat[c] = 'Unknown'

X = df_chat[all_features]
y_survival = df_chat['Outcome']
y_disease = df_chat['Disease']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

print("Training Chatbot Models...")
model_survival = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
model_survival.fit(X, y_survival)

model_disease = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
model_disease.fit(X, y_disease)
print("✅ Chatbot Models Ready!")

# =========================================================
# Chat page & predict endpoint
# =========================================================
@app.route('/chat')
def chat_page():
    return render_template_string(index_template)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        species = request.form['species']
        breed = request.form['breed']
        sex = request.form['sex']
        bp_str = request.form['bp']
        heart_rate = request.form['heart_rate']
        health_status = request.form['health_status']
        symptom1 = request.form.get('symptom_1', '').lower().strip()
        symptom2 = request.form.get('symptom_2', '').lower().strip()

        # ----------------------------
        #  DISEASE KEYWORD DATABASE
        # ----------------------------
        disease_keywords = {
            "viral infection": ["fever", "weakness", "cough", "runny nose"],
            "worm infection": ["diarrhea", "weight loss", "worms", "bloating"],
            "avian influenza": ["nasal discharge", "respiratory distress", "swelling", "cyanosis"],
            "rabies": ["aggression", "foaming", "paralysis", "biting"],
            "anthrax": ["sudden death", "bloody discharge", "swelling"],
            "mastitis": ["swollen udder", "udder pain", "milk change"],
            "fmd": ["blisters", "mouth lesions", "foot lesions", "drooling"],
            "ppr": ["mouth ulcers", "diarrhea", "pneumonia", "ocular discharge"]
        }

        # ----------------------------
        #  DISEASE SCORING (One Best Disease)
        # ----------------------------
        scores = {}

        for disease, keywords in disease_keywords.items():
            count = 0
            for kw in keywords:
                if kw in symptom1:
                    count += 1
                if kw in symptom2:
                    count += 1
            scores[disease] = count

        best_disease = max(scores, key=scores.get)

        if scores[best_disease] == 0:
            best_disease = "Unknown Disease"

        # ----------------------------
        #  HEART RATE PROCESSING
        # ----------------------------
        try:
            heart_rate = int(heart_rate)
        except:
            heart_rate = int(df_chat['Heart Rate (bpm)'].median())

        # ----------------------------
        #  BP PROCESSING
        # ----------------------------
        try:
            bp_systolic, bp_diastolic = map(int, bp_str.split('/'))
        except:
            bp_systolic = int(df_chat['BP_Systolic'].median())
            bp_diastolic = int(df_chat['BP_Diastolic'].median())

        # ----------------------------
        #  PREPARE INPUT FOR ML MODELS
        # ----------------------------
        input_data = pd.DataFrame([[heart_rate, bp_systolic, bp_diastolic,
                                    species, breed, sex, health_status,
                                    symptom1, symptom2]],
                                  columns=all_features)

        # ----------------------------
        #  SURVIVAL PREDICTION MODEL
        # ----------------------------
        prediction_survive = model_survival.predict(input_data)[0]
        probabilities_survive = model_survival.predict_proba(input_data)[0]

        # Index for "Will Live" class
        if "Will Live" in model_survival.classes_:
            live_index = list(model_survival.classes_).index("Will Live")
        else:
            live_index = 0

        chance_of_living = round(probabilities_survive[live_index] * 100, 2)

        # ----------------------------
        #  DISEASE MODEL (OPTIONAL)
        # ----------------------------
        prediction_disease_ml = model_disease.predict(input_data)[0]

        # ----------------------------
        #  FINAL DISEASE = KEYWORD RESULT
        # ----------------------------
        final_disease = best_disease

        # ----------------------------
        #  RETURN RESULT TO USER
        # ----------------------------
        return render_template_string(
            index_template,
            prediction_disease=f'Predicted Disease: {final_disease}',
            prediction_survival=f'Predicted Outcome: {prediction_survive}',
            living_chance=f'Chance of Living: {chance_of_living}%',
            show_result=True
        )

    except Exception as e:
        return render_template_string(
            index_template,
            prediction_disease=f'Error: {e}',
            show_result=True
        )

# =========================================================
# Run App
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
