#!/usr/bin/env python3
"""
Single-file Flask app — cleaned and ready-to-run.

Usage:
  python app_fixed.py

Notes:
 - Ensure CSV file path below is valid. Edit CSV_FILE variable if needed.
 - Camera / html5-qrcode requires HTTPS or localhost to allow camera access in modern browsers.
"""

from flask import Flask, render_template_string, render_template, request, redirect, url_for, send_file
import pandas as pd
import numpy as np
import os
import re
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

app = Flask(__name__)

# ---------------------------
# CSV location (edit if needed)
# ---------------------------
CSV_FILE = r"c:/Users/hp/OneDrive/Desktop/new/Animal_Health_Record_500.csv"
FALLBACK_CSV = "/mnt/data/Animal_Health_Record_500.csv"

if not os.path.exists(CSV_FILE):
    if os.path.exists(FALLBACK_CSV):
        CSV_FILE = FALLBACK_CSV
    else:
        raise FileNotFoundError(
            f"CSV dataset not found at either:\n - {CSV_FILE}\n - {FALLBACK_CSV}\n"
            "Place the CSV at one of those paths or change CSV_FILE in the script."
        )
    
@app.route("/")
def home():
    return redirect("/dashboard")


# ===========================
# Simple health model (RF)
# ===========================
def train_health_model(csv_path):
    df_local = pd.read_csv(csv_path)
    required = ["BP", "Heart Rate (bpm)", "Age (years)", "Health Status"]
    for col in required:
        if col not in df_local.columns:
            raise KeyError(f"Missing column: {col} in CSV. Required: {required}")

    def _systolic(bp):
        try:
            return float(str(bp).split("/")[0])
        except:
            return 0.0

    df_local["BP_num"] = df_local["BP"].apply(_systolic)
    df_local["Heart Rate (bpm)"] = pd.to_numeric(df_local["Heart Rate (bpm)"], errors="coerce").fillna(0)
    df_local["Age (years)"] = pd.to_numeric(df_local["Age (years)"], errors="coerce").fillna(0)

    X = df_local[["BP_num", "Heart Rate (bpm)", "Age (years)"]]
    y = df_local["Health Status"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return clf, acc

model, accuracy = train_health_model(CSV_FILE)

# ============================================
# Symptom -> disease quick mapping (fallback)
# ============================================
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

# ===========================
# Templates (embedded)
# ===========================
# display_template (uses url_for('dashboard') for the back link)
display_template = """(the same long HTML as in previous messages was used here)"""
# To avoid cluttering this response with extremely long HTML, we will reinsert the templates
# as used previously in your files below when calling render_template_string.
# But in the real file, the full HTML strings are placed (they are included further down).

# For practicality and to keep the script readable, declare the actual templates now:
display_template = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Animal Record</title>
<style>
body { font-family: 'Poppins', sans-serif; background: linear-gradient(135deg, #c9ffbf, #ffafbd); text-align: center; padding-top: 40px; overflow-x: hidden; position: relative; }
.card { display:inline-block; background:white; padding:30px 40px; border-radius:25px; box-shadow:0 6px 20px rgba(0,0,0,0.2); max-width:900px; width:92%; text-align:left; }
.add-btn{ background:#38b000; color:white; border:none; padding:8px 12px; border-radius:8px; cursor:pointer }
.save-btn{ background:#0077b6; color:white; padding:8px 16px; border:none; border-radius:8px; cursor:pointer }
.small{ font-size:0.9em; color:#555 }
.center{text-align:center}
.star{ color: gold; font-size:20px; vertical-align:middle; margin-left:8px }
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

    <div><b>Name:</b> {{ data.get('Name','') }}</div>
    <div><b>Breed:</b> {{ data.get('Breed','') }}</div>
    <div><b>Age:</b> {{ data.get('Age (years)', data.get('Age','')) }}</div>
    <div><b>BP:</b> {{ data.get('BP','') }}</div>
    <div><b>Heart Rate:</b> {{ data.get('Heart Rate (bpm)', data.get('Heart Rate','')) }}</div>
    <div><b>Oxygen Saturation:</b> {{ data.get('Oxygen Saturation (%)', data.get('Oxygen Saturation','')) }}</div>
    <div><b>Symptoms:</b> {{ data.get('Symptom 1','') }} {% if data.get('Symptom 2','') %}, {{ data.get('Symptom 2','') }}{% endif %}</div>
    <div><b>Detected Disease:</b> 🧬 <b>{{ prediction }}</b></div>
    <div class="small"><b>Model Accuracy:</b> {{ accuracy }}%</div>

    <hr>

    <div style="display:flex; gap:20px; flex-wrap:wrap;">
      <div style="flex:1; min-width:220px; background:#f1faff; padding:12px; border-radius:10px;">
        <h4>💉 Vaccination History</h4>
        <p><b>1)</b> {{ data.get('Vaccination 1','—') }}</p>
        <p><b>2)</b> {{ data.get('Vaccination 2','—') }}</p>

        <button class="add-btn" onclick="document.getElementById('vaccineForm').style.display='block'; this.style.display='none';">
          + Add Vaccination
        </button>

        <form id="vaccineForm" style="display:none; margin-top:10px;" method="POST" action="{{ url_for('add_vaccine') }}">
          <input type="hidden" name="animal_id" value="{{ data['Animal ID'] }}">
          <input type="text" name="new_vaccine" placeholder="Enter Vaccine Name" required style="width:100%; padding:8px; margin-top:8px;"><br>
          <input type="date" name="vaccine_date" required style="width:100%; padding:8px; margin-top:8px;"><br>
          <div style="margin-top:8px;">
            <button type="submit" class="add-btn">Save Vaccine</button>
            <button type="button" class="save-btn" onclick="document.getElementById('vaccineForm').style.display='none'; document.querySelector('.add-btn').style.display='inline-block';">Cancel</button>
          </div>
        </form>
      </div>

      <div style="flex:1; min-width:220px; background:#fff; padding:12px; border-radius:10px;">
        <h4>📌 Recommended Vaccines</h4>
        <p><b>Large Animals:</b></p>
        <ul>
          <li>FMD</li>
          <li>HS</li>
          <li>BQ</li>
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
    <form method="GET" action="{{ url_for('generate_pdf') }}">
      <input type="hidden" name="animal_id" value="{{ data['Animal ID'] }}">
      <button type="submit" class="save-btn" style="background:#6a4c93;">📄 Generate PDF Report</button>
    </form>

  </div>
{% else %}
  <div class="card center">
    <h3>⚠️ No record found for Animal ID: {{ animal_id }}</h3>
    <p><a href="{{ url_for('dashboard') }}">Back to Dashboard</a></p>
  </div>
{% endif %}

</body>
</html>
"""

index_template = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Health Chatbot</title>
  <style>
    body { font-family: 'Poppins', sans-serif; text-align:center; padding:40px; background:linear-gradient(135deg,#f6d365,#fda085); }
    .card { background:white; display:inline-block; padding:20px 30px; border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,0.12); }
    input, select { padding:8px; margin:6px; width:220px; }
    button { padding:8px 14px; background:#0077b6; color:white; border:none; border-radius:8px; cursor:pointer; }
  </style>
</head>
<body>
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

    <p style="margin-top:12px;"><a href="{{ url_for('dashboard') }}">Back</a></p>
  </div>
</body>
</html>
"""



# ---------------------------
# Simple in-memory doctor store
# ---------------------------
VALID_DOCTORS = {"doctor123": {"name": "Default Doctor", "password": "admin@123"}}

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        doctor_name = request.form.get("doctorName", "").strip()
        doctor_id = request.form.get("doctorId", "").strip()
        password = request.form.get("doctorPassword", "")

        if len(password) < 6 or not re.search(r'[^A-Za-z0-9]', password):
            return "<h3 style='color:red;'>Invalid password (6+ chars and 1+ special char required)</h3><a href='/'>Back</a>"

        if doctor_id in VALID_DOCTORS:
            if VALID_DOCTORS[doctor_id]["password"] == password:
                return redirect(url_for("dashboard"))
            else:
                return "<h3 style='color:red;'>Incorrect password</h3><a href='/'>Try again</a>"

        VALID_DOCTORS[doctor_id] = {"name": doctor_name, "password": password}
        return redirect(url_for("dashboard"))

    return """
    <!DOCTYPE html><html><head><meta charset="utf-8"><title>Login</title></head>
    <body style="font-family:Arial; display:flex; align-items:center; justify-content:center; height:100vh;">
      <div style="background:#16324a; color:white; padding:30px; border-radius:10px; width:320px;">
        <h2>Doctor Login</h2>
          <div style="font-size:3rem; margin-bottom:15px;">
          🩺🧑‍⚕️🏥
          </div>
          <div class="chicks-run">
          🐥🐥🐥🐥🐥
          </div>

          <style>
          .chicks-run {
          position: absolute;
          top: 40px;      /* Distance from top – adjust if needed */
          left: -300px;   /* Start off-screen */
          font-size: 2.5rem;
          white-space: nowrap;
          animation: runAcross 8s linear infinite;
          }

          @keyframes runAcross {
          0% { left: -300px; }
          100% { left: 100%; }
          }
          </style>


         <form method="POST">
          <input name="doctorName" placeholder="Doctor name" required style="width:100%; padding:8px; margin:6px 0;"><br>
          <input name="doctorId" placeholder="Doctor ID" required style="width:100%; padding:8px; margin:6px 0;"><br>
          <input type="password" name="doctorPassword" placeholder="Password" required style="width:100%; padding:8px; margin:6px 0;"><br>
          <button type="submit" style="width:100%; padding:10px; background:#00d4ff; border:none; border-radius:6px;">Login</button>
        </form>
        <p style="margin-top:10px;"><a href="/reset" style="color:#ff9aa2;">Reset Password</a></p>
      </div>
    </body>
    <footer style="
    text-align:center;
    padding: 12px;
    font-size: 1.1rem;
    background-color:black;
    color:white;
    position: fixed;
    bottom: 0;
    width: 100%;
    border-top: 1px solid #ccc;">
    🐾 Govt Animal health care center|Dr. Krishna Murthy N E|govtanimal@healthcare.gmail.com | Chikballapura,Karnataka-560074 🐾
    </footer>
    </html>
    """

@app.route("/reset", methods=["GET", "POST"])
def reset_password():
    if request.method == "POST":
        doctor_id = request.form.get("doctorId", "").strip()
        new_pass = request.form.get("newPassword")
        confirm = request.form.get("confirmPassword")
        if new_pass != confirm:
            return "<h3>Passwords do not match</h3><a href='/reset'>Back</a>"
        if len(new_pass) < 6 or not re.search(r'[^A-Za-z0-9]', new_pass):
            return "<h3>Password must have 6+ chars and a special character</h3><a href='/reset'>Back</a>"
        if doctor_id not in VALID_DOCTORS:
            return "<h3>Doctor ID not found</h3><a href='/reset'>Back</a>"
        VALID_DOCTORS[doctor_id]["password"] = new_pass
        return "<h3>Password reset</h3><a href='/'>Login</a>"

    return """
    <!DOCTYPE html>
<html>
<head>
    <title>Reset Password</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #4b79a1, #283e51);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #fff;
        }

        .container {
            background: rgba(255, 255, 255, 0.15);
            padding: 30px;
            border-radius: 12px;
            backdrop-filter: blur(8px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            width: 320px;
            text-align: center;
        }

        h2 {
            margin-bottom: 20px;
            font-weight: 600;
        }

        input {
            width: 90%;
            padding: 10px;
            margin: 10px 0;
            border-radius: 6px;
            border: none;
            outline: none;
            font-size: 15px;
        }

        button {
            background: #00c6ff;
            border: none;
            padding: 10px 20px;
            color: #fff;
            font-size: 16px;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            margin-top: 15px;
        }

        button:hover {
            background: #009cd3;
        }
    </style>
</head>

<body>

    <div class="container">
        <h2>Reset Password</h2>
        <form method="POST">
            <input name="doctorId" placeholder="Doctor ID" required><br>
            <input type="password" name="newPassword" placeholder="New Password" required><br>
            <input type="password" name="confirmPassword" placeholder="Confirm Password" required><br>
            <button type="submit">Reset</button>
        </form>
    </div>

</body>
</html>
"""

# ---------------------------
# Dashboard Page
# ---------------------------
@app.route('/dashboard')
def dashboard():
    return """
    <body style="background: linear-gradient(135deg, #d4fc79, #96e6a1); text-align:center; font-family: Poppins, sans-serif; padding-top:80px;">
      <div style="font-size:80px;">🐄🐕🐈</div>
      <h1>Animal Health Prediction Dashboard</h1>
      <p>Enter Animal ID to view or update record:</p>
      <form action="/display" method="get">
        <input type="text" name="animal_id" placeholder="Enter Animal ID" required style="padding:10px; width:250px;">
        <br><br>
        <button type="submit" style="padding:10px 20px; background:#0077b6; color:white; border:none; border-radius:8px;">View Record</button>
      </form>
      <br><br>

      <!-- FIXED SCAN BUTTON -->
      <button onclick="window.location.href='/scan'"
              style="padding:10px 20px; background:#0077b6; color:white; border:none;
                     border-radius:8px; font-size:18px;">
        🔍 Scan QR Code
      </button>
      <br><br>
      <a href="/chat" style="font-size:18px; color:#004c70; text-decoration:none;">💬 Try Health Chatbot</a>
    </body>
    """

# ---- NEW /scan ROUTE (paste this once) ----
# ---------------------------
# QR Scan Page
# ---------------------------
scan_template = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>QR Scan</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="text-align:center; padding:20px;">

<h2>Scan Animal QR</h2>

<div id="reader" style="width:300px; margin:auto;"></div>
<div id="result" style="margin-top:15px; font-weight:bold;">Waiting for camera...</div>

<p><a href="/dashboard">⬅ Back</a></p>

<script src="https://unpkg.com/html5-qrcode"></script>

<script>
const resultEl = document.getElementById('result');
const qr = new Html5Qrcode("reader");

Html5Qrcode.getCameras().then(cameras => {
  if (cameras.length > 0) {
    qr.start(
      cameras[0].id,
      { fps: 10, qrbox: { width: 250, height: 250 } },
      (decoded) => {
        resultEl.innerHTML = "Detected: " + decoded;
        window.location.href = "/display?animal_id=" + encodeURIComponent(decoded);
      },
      (err) => {}
    );
  } else {
    resultEl.innerHTML = "No camera found";
  }
});
</script>

</body>
</html>
"""

@app.route('/scan')
def scan():
    return render_template_string(scan_template)
# ---------------------------
# Display & update record
# ---------------------------
@app.route('/display', methods=['GET', 'POST'])
def display():
    animal_id = request.args.get('animal_id')
    if not animal_id:
        return "<h3>No Animal ID provided!</h3><a href='/dashboard'>Back</a>"

    df = pd.read_csv(CSV_FILE)
    for col in ["Doctor Suggestion", "Vaccination 1", "Vaccination 2", "Special Care"]:
        if col not in df.columns:
            df[col] = ""

    record_index = df.index[df["Animal ID"].astype(str) == str(animal_id)]
    if record_index.empty:
        return render_template_string(display_template, found=False, animal_id=animal_id)

    idx = record_index[0]

    # update symptoms / suggestion / special care
    if request.method == "POST" and request.form.get("new_vaccine") is None:
        new_symptom1 = request.form.get("Symptom1", "").strip()
        new_symptom2 = request.form.get("Symptom2", "").strip()
        suggestion = request.form.get("suggestion", "").strip()
        special_checked = request.form.get("special_care")
        df.at[idx, "Symptom 1"] = new_symptom1
        df.at[idx, "Symptom 2"] = new_symptom2
        df.at[idx, "Doctor Suggestion"] = suggestion
        df.at[idx, "Special Care"] = "Yes" if special_checked else "No"
        df = df.replace({np.nan: ""})
        df.to_csv(CSV_FILE, index=False)
        return redirect(url_for('display', animal_id=animal_id))

    data = df.iloc[idx].to_dict()
    for k, v in list(data.items()):
        if pd.isna(v) or str(v).lower() == "nan":
            data[k] = ""
        else:
            data[k] = v

    # disease prediction: try symptom map first, else health model
    try:
        bp_value = float(str(data.get("BP", "0")).split("/")[0])
        hr = float(data.get("Heart Rate (bpm)", 0) or data.get("Heart Rate", 0) or 0)
        age = float(data.get("Age (years)", 0) or data.get("Age", 0) or 0)
        sym1 = str(data.get("Symptom 1", "")).lower()
        sym2 = str(data.get("Symptom 2", "")).lower()
        disease = None
        for k, v in disease_map.items():
            if k in sym1 or k in sym2:
                disease = v
                break
        if disease:
            prediction = f"{disease} 🔴"
        else:
            prediction = f"{model.predict([[bp_value, hr, age]])[0]} 🟢"
    except Exception as e:
        print("Prediction error:", e)
        prediction = "Healthy 🟢"

    return render_template_string(display_template, found=True, data=data, prediction=prediction, accuracy=round(accuracy*100, 2))



  # your filename


# ---------------------------
# Vaccine add
# ---------------------------
@app.route('/add_vaccine', methods=['POST'])
def add_vaccine():
    animal_id = request.form.get('animal_id')
    new_vaccine = request.form.get('new_vaccine', '').strip()
    vaccine_date = request.form.get('vaccine_date', '').strip()
    if not animal_id or not new_vaccine or not vaccine_date:
        return "Missing data", 400
    df = pd.read_csv(CSV_FILE)
    if "Vaccination 1" not in df.columns:
        df["Vaccination 1"] = ""
    if "Vaccination 2" not in df.columns:
        df["Vaccination 2"] = ""
    record_index = df.index[df["Animal ID"].astype(str) == str(animal_id)]
    if record_index.empty:
        return f"No record found for Animal ID {animal_id}", 404
    idx = record_index[0]
    previous_v1 = df.at[idx, "Vaccination 1"] if pd.notna(df.at[idx, "Vaccination 1"]) else ""
    df.at[idx, "Vaccination 2"] = previous_v1
    df.at[idx, "Vaccination 1"] = f"{new_vaccine} ({vaccine_date})"
    df = df.replace({np.nan: ""})
    df.to_csv(CSV_FILE, index=False)
    return redirect(url_for('display', animal_id=animal_id))


# ---------------------------
# PDF generation
# ---------------------------
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

    for k, v in list(data.items()):
        if pd.isna(v) or str(v).lower() == "nan":
            data[k] = ""
        else:
            data[k] = str(v)

    folder = os.path.dirname(CSV_FILE)
    pdf_name = f"Animal_{animal_id}_report.pdf"
    pdf_path = os.path.join(folder, pdf_name)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)

    story = []

    # ---------------- HEADER START ----------------
    from reportlab.platypus import Image, Table, TableStyle
    from reportlab.lib import colors

    logo_path = "C:\\Users\\hp\\OneDrive\\画像\\Saved Pictures\\animal.jpg" # PLACE YOUR LOGO HERE

    try:
        logo = Image(logo_path, width=60, height=60)
    except:
        logo = Paragraph("<b>[Logo Missing]</b>", styles["Normal"])

    hospital_info = [
        ["Govt Animal Health Care Center",
         "Chikkaballapura, Karnataka - 560074<br/>"
         "Phone: +9821345633<br/>"
         "Email: govtanimal@heathcare.gmail.com"]
    ]

    table = Table(
        [[logo, Paragraph(
            "<b>Govt Animal Health Care Center</b><br/>"
            "Chikkaballapura, Karnataka - 560074<br/>"
            "Phone: +9821345633<br/>"
            "Email: govtanimal@heathcare.gmail.com",
            styles["Normal"]
        )]],
        colWidths=[70, 400]
    )

    table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12)
    ]))

    story.append(table)
    story.append(Spacer(1, 20))
    # ---------------- HEADER END ----------------

    story.append(Paragraph(f" <b>Animal Health Report — ID: {animal_id}</b>", styles['Title']))
    story.append(Spacer(1, 12))

    lines = [
        ("Name", data.get('Name','')),
        ("Breed", data.get('Breed','')),
        ("Age (years)", data.get('Age (years)','') or data.get('Age','')),
        ("BP", data.get('BP','')),
        ("Heart Rate (bpm)", data.get('Heart Rate (bpm)','') or data.get('Heart Rate','')),
        ("Oxygen Saturation (%)", data.get('Oxygen Saturation (%)','') or data.get('Oxygen Saturation','')),
        ("Symptoms",
         f"{data.get('Symptom 1','')}{', ' + data.get('Symptom 2','') if data.get('Symptom 2','') else ''}"),
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


# ===========================
# Chatbot training (single, consistent)
# ===========================
# Load dataframe for chat models:
df_chat = pd.read_csv(CSV_FILE)

# Ensure columns exist
if 'Disease' not in df_chat.columns:
    df_chat['Disease'] = 'Unknown'
if 'Symptom 1' not in df_chat.columns:
    df_chat['Symptom 1'] = 'None'
if 'Symptom 2' not in df_chat.columns:
    df_chat['Symptom 2'] = 'None'

fatal_diseases_list = ['Rabies', 'Anthrax', 'PPR (Peste des petits ruminants)']
df_chat['Outcome'] = df_chat['Disease'].apply(lambda x: 'Will Not Live' if x in fatal_diseases_list else 'Will Live')

bp_split = df_chat['BP'].astype(str).str.split('/', expand=True)
df_chat['BP_Systolic'] = pd.to_numeric(bp_split[0], errors='coerce').fillna(df_chat['BP_Systolic'].median() if 'BP_Systolic' in df_chat else 0)
df_chat['BP_Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce').fillna(df_chat['BP_Diastolic'].median() if 'BP_Diastolic' in df_chat else 0)
df_chat['Heart Rate (bpm)'] = pd.to_numeric(df_chat['Heart Rate (bpm)'], errors='coerce').fillna(df_chat['Heart Rate (bpm)'].median() if 'Heart Rate (bpm)' in df_chat else 0)

numerical_features = ['Heart Rate (bpm)', 'BP_Systolic', 'BP_Diastolic']
categorical_features = ['Species', 'Breed', 'Sex', 'Health Status', 'Symptom 1', 'Symptom 2']
all_features = numerical_features + categorical_features

for c in categorical_features:
    if c not in df_chat.columns:
        df_chat[c] = 'Unknown'

X_chat = df_chat[all_features]
y_survival = df_chat['Outcome']
y_disease = df_chat['Disease']

# Use sparse_output for modern sklearn versions
preprocessor_chat = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

model_survival = Pipeline([('preprocessor', preprocessor_chat),
                           ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
model_survival.fit(X_chat, y_survival)

model_disease = Pipeline([('preprocessor', preprocessor_chat),
                          ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
model_disease.fit(X_chat, y_disease)


# ===========================
# Chat routes
# ===========================
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

    "viral infection ": ["fever", "weakness", "cough", "runny nose"],
    "worm infection : appear in parasitic infestations like tapeworm and hookworm": ["diarrhea", "weight loss", "worms", "bloating"],
    "avian influenza : viral respiratory diseases seen in poultry.": ["nasal discharge", "respiratory distress", "swelling", "cyanosis"],
    "rabies similaritiesn : with neurological disorders causing behavioral changes": ["aggression", "foaming", "paralysis", "biting"],
    "anthrax : hemorrhagic septicemia in severe cases": ["sudden death", "bloody discharge", "swelling"],
    "mastitis : sociated with bacterial udder infections in dairy animals": ["swollen udder", "udder pain", "milk change"],
    "fmd : vesicular stomatitis due to mouth and foot lesions": ["blisters", "mouth lesions", "foot lesions", "drooling"],
    "ppr : signs with rinderpest-like viral diseases in small ruminants": ["mouth ulcers", "diarrhea", "pneumonia", "ocular discharge"],

    # Added diseases
    "parvovirus : less in water content": ["vomiting", "bloody diarrhea", "dehydration", "lethargy"],
    "distemper : ": ["fever", "nasal discharge", "seizures", "neurological signs"],
    "leptospirosis": ["jaundice", "vomiting", "kidney failure", "muscle pain"],
    "brucellosis": ["abortion", "infertility", "swollen joints", "weakness"],
    "kennel cough": ["dry cough", "gagging", "sneezing", "nasal discharge"],
    "hemorrhagic septicemia": ["high fever", "swelling of throat", "difficulty breathing"],
    "black quarter": ["swollen limb", "lameness", "fever", "crepitus"],
    "rsv infection": ["coughing", "rapid breathing", "nasal discharge"],
    "canine influenza": ["cough", "fever", "nasal discharge", "lethargy"],
    "tick fever": ["pale gums", "high fever", "tick infestation", "weakness"],
    "babesiosis": ["dark urine", "anemia", "fever", "lethargy"],
    "theileriosis": ["swollen lymph nodes", "fever", "weakness", "labored breathing"],
    "scrapie": ["itching", "behavior changes", "tremors", "loss of coordination"],
    "enterotoxemia": ["sudden death", "diarrhea", "abdominal pain"],
    "colibacillosis": ["diarrhea", "dehydration", "poor growth"],
    "newcastle disease": ["twisted neck", "respiratory distress", "green diarrhea"],
    "canine hepatitis": ["fever", "abdominal pain", "vomiting", "jaundice"],
    "feline panleukopenia": ["vomiting", "bloody diarrhea", "severe dehydration"]
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
            best_disease = "More test recommended"

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

    
# ===========================
# Run app
# ===========================
if __name__ == "__main__":
    # If you want to test camera on phone, run on host='0.0.0.0' and open via the machine's LAN IP.
    # Remember: many browsers require HTTPS (or localhost) to grant camera permissions.
  app.run(debug=True)
