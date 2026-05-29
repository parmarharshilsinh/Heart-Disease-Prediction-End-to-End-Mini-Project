from flask import Flask, request, jsonify, render_template_string
import pickle, json, numpy as np, os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

try:
    with open(os.path.join(MODELS_DIR, 'rf_model.pkl'), 'rb') as f:
        rf = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'lr_model.pkl'), 'rb') as f:
        lr = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'features.json'), 'r') as f:
        features = json.load(f)
except FileNotFoundError as e:
    raise SystemExit(f"Model data file not found: {e.filename}. Ensure all files are present in {MODELS_DIR}")
except Exception as e:
    raise SystemExit(f"Failed to load model resources: {e}")

def encode_input(data):
    row = []
    for feat in features:
        val = data.get(feat)
        if feat in encoders:
            le = encoders[feat]
            val = le.transform([str(val)])[0]
        row.append(float(val))
    return np.array(row).reshape(1,-1)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_type = data.pop('model', 'rf')
    X = encode_input(data)
    if model_type == 'lr':
        proba = lr.predict_proba(scaler.transform(X))[0][1]
    else:
        proba = rf.predict_proba(X)[0][1]
    
    prob_pct = round(float(proba)*100, 1)
    if proba < 0.30:
        risk_level = 'LOW'
    elif proba < 0.55:
        risk_level = 'MODERATE'
    else:
        risk_level = 'HIGH'
    
    return jsonify({
        'probability': prob_pct,
        'prediction': 'Yes' if proba > 0.5 else 'No',
        'risk_level': risk_level
    })

@app.route('/')
def home():
    html = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CardioScan — Heart Disease Risk Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0d14;
    --surface: #10141f;
    --surface2: #161b2e;
    --border: rgba(255,255,255,0.07);
    --red: #ff3d5a;
    --red-glow: rgba(255,61,90,0.25);
    --amber: #ffb830;
    --green: #00e5a0;
    --blue: #4d9fff;
    --text: #e8eaf2;
    --muted: #5a6080;
    --accent: #ff3d5a;
  }

  * { margin:0; padding:0; box-sizing:border-box; }

  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Animated background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
      radial-gradient(ellipse 80% 50% at 20% 10%, rgba(255,61,90,0.06) 0%, transparent 60%),
      radial-gradient(ellipse 60% 40% at 80% 80%, rgba(77,159,255,0.05) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
  }

  .page { position: relative; z-index: 1; max-width: 1100px; margin: 0 auto; padding: 40px 24px 80px; }

  /* Header */
  .header { text-align: center; margin-bottom: 56px; }
  .logo {
    display: inline-flex; align-items: center; gap: 12px;
    background: rgba(255,61,90,0.08);
    border: 1px solid rgba(255,61,90,0.2);
    padding: 8px 20px; border-radius: 100px;
    margin-bottom: 28px;
    animation: fadeDown 0.6s ease both;
  }
  .logo-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--red);
    box-shadow: 0 0 12px var(--red);
    animation: pulse 1.8s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(0.8)} }
  .logo span { font-size: 12px; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: var(--red); }

  h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(36px, 5vw, 60px);
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin-bottom: 16px;
    animation: fadeDown 0.6s 0.1s ease both;
  }
  h1 em { font-style: italic; color: var(--red); }
  .subtitle {
    color: var(--muted); font-size: 16px; font-weight: 300;
    max-width: 480px; margin: 0 auto;
    animation: fadeDown 0.6s 0.2s ease both;
  }

  @keyframes fadeDown { from{opacity:0;transform:translateY(-16px)} to{opacity:1;transform:translateY(0)} }

  /* ECG line decoration */
  .ecg-line { width: 100%; max-width: 600px; margin: 28px auto 0; opacity: 0.3; animation: fadeDown 0.6s 0.3s ease both; }

  /* Card */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 40px;
    margin-bottom: 24px;
    animation: fadeUp 0.6s 0.3s ease both;
  }
  @keyframes fadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }

  .card-title {
    font-size: 11px; font-weight: 600; letter-spacing: 0.15em;
    text-transform: uppercase; color: var(--muted);
    margin-bottom: 24px; display: flex; align-items: center; gap: 10px;
  }
  .card-title::after { content:''; flex:1; height:1px; background:var(--border); }

  /* Grid */
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; }

  /* Form field */
  .field { display: flex; flex-direction: column; gap: 8px; }
  .field label {
    font-size: 12px; font-weight: 500; color: var(--muted);
    letter-spacing: 0.04em; text-transform: uppercase;
  }

  .field input[type="number"],
  .field input[type="range"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    font-family: inherit;
    font-size: 15px;
    font-weight: 500;
    padding: 12px 14px;
    transition: border-color 0.2s, box-shadow 0.2s;
    -moz-appearance: textfield;
  }
  .field input[type="number"]::-webkit-inner-spin-button { -webkit-appearance: none; }
  .field input:focus { outline: none; border-color: var(--red); box-shadow: 0 0 0 3px var(--red-glow); }

  /* Toggle group (Low / Medium / High) */
  .toggle-group { display: flex; border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }
  .toggle-group input { display: none; }
  .toggle-group label {
    flex: 1; text-align: center;
    padding: 11px 4px;
    font-size: 12px; font-weight: 600;
    letter-spacing: 0.05em; text-transform: uppercase;
    cursor: pointer;
    background: var(--surface2);
    color: var(--muted);
    transition: background 0.2s, color 0.2s;
    border-right: 1px solid var(--border);
  }
  .toggle-group label:last-of-type { border-right: none; }
  .toggle-group input:checked + label { color: var(--bg); }
  .toggle-group.tri input[value="Low"]:checked   + label { background: var(--green); color: #000; }
  .toggle-group.tri input[value="Medium"]:checked + label { background: var(--amber); color: #000; }
  .toggle-group.tri input[value="High"]:checked   + label { background: var(--red);   color: #fff; }
  .toggle-group.bin input[value="No"]:checked  + label { background: var(--green); color: #000; }
  .toggle-group.bin input[value="Yes"]:checked + label { background: var(--red);   color: #fff; }
  .toggle-group.gender input[value="Male"]:checked   + label { background: var(--blue); color: #fff; }
  .toggle-group.gender input[value="Female"]:checked + label { background: #e85d9f;    color: #fff; }

  /* Number display */
  .num-display {
    font-size: 22px; font-weight: 600; color: var(--text);
    display: flex; align-items: baseline; gap: 4px;
  }
  .num-display small { font-size: 11px; color: var(--muted); font-weight: 400; }

  /* Range input */
  input[type="range"] {
    -webkit-appearance: none;
    width: 100%; height: 4px;
    background: var(--surface2);
    border-radius: 4px;
    border: none !important;
    padding: 0 !important;
    cursor: pointer;
    accent-color: var(--red);
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 18px; height: 18px;
    border-radius: 50%;
    background: var(--red);
    box-shadow: 0 0 10px var(--red-glow);
    cursor: pointer;
  }

  /* Model selector */
  .model-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 4px; }
  .model-option { position: relative; }
  .model-option input { display: none; }
  .model-option label {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 18px;
    border: 1px solid var(--border);
    border-radius: 10px;
    cursor: pointer;
    font-size: 14px; font-weight: 500;
    background: var(--surface2);
    transition: all 0.2s;
  }
  .model-option label .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--border); border: 1px solid var(--muted);
    transition: all 0.2s;
  }
  .model-option input:checked + label { border-color: var(--red); color: var(--red); background: rgba(255,61,90,0.06); }
  .model-option input:checked + label .dot { background: var(--red); border-color: var(--red); box-shadow: 0 0 8px var(--red); }

  /* Submit button */
  .btn-predict {
    width: 100%; padding: 18px;
    background: var(--red);
    border: none; border-radius: 14px;
    color: #fff;
    font-family: 'DM Sans', sans-serif;
    font-size: 16px; font-weight: 600;
    letter-spacing: 0.04em;
    cursor: pointer;
    margin-top: 8px;
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
    box-shadow: 0 8px 32px rgba(255,61,90,0.35);
    display: flex; align-items: center; justify-content: center; gap: 10px;
  }
  .btn-predict:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(255,61,90,0.45); }
  .btn-predict:active { transform: translateY(0); }
  .btn-predict:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  /* Spinner */
  .spinner {
    width: 20px; height: 20px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Result panel */
  #resultPanel {
    display: none;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 40px;
    animation: fadeUp 0.4s ease both;
  }
  #resultPanel.show { display: block; }

  .risk-meter { margin: 28px 0 32px; }
  .risk-bar-track {
    width: 100%; height: 10px;
    background: var(--surface2);
    border-radius: 100px;
    overflow: hidden;
    margin-bottom: 8px;
  }
  .risk-bar-fill {
    height: 100%; border-radius: 100px;
    transition: width 1.2s cubic-bezier(0.25, 1, 0.5, 1);
    width: 0;
  }
  .risk-bar-fill.low      { background: linear-gradient(90deg, #00e5a0, #00b87a); }
  .risk-bar-fill.moderate { background: linear-gradient(90deg, #ffb830, #ff8c00); }
  .risk-bar-fill.high     { background: linear-gradient(90deg, #ff6b6b, #ff3d5a); }

  .risk-labels { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); }

  .result-headline {
    font-family: 'DM Serif Display', serif;
    font-size: 40px; font-weight: 400;
    line-height: 1;
  }
  .result-headline.low      { color: var(--green); }
  .result-headline.moderate { color: var(--amber); }
  .result-headline.high     { color: var(--red); text-shadow: 0 0 40px var(--red-glow); }

  .result-sub { color: var(--muted); font-size: 14px; margin-top: 8px; }

  .stats-row { display: flex; gap: 20px; flex-wrap: wrap; margin-top: 24px; }
  .stat-box {
    flex: 1; min-width: 140px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
  }
  .stat-box .val { font-size: 28px; font-weight: 600; }
  .stat-box .lbl { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }

  .disclaimer {
    margin-top: 24px;
    padding: 14px 16px;
    background: rgba(255,184,48,0.06);
    border: 1px solid rgba(255,184,48,0.15);
    border-radius: 10px;
    font-size: 12px; color: var(--muted);
    line-height: 1.6;
  }

  .footer { text-align: center; margin-top: 56px; color: var(--muted); font-size: 12px; }
  .footer a { color: var(--red); text-decoration: none; }
</style>
</head>
<body>

<div class="page">

  <!-- Header -->
  <div class="header">
    <div class="logo">
      <div class="logo-dot"></div>
      <span>CardioScan AI</span>
    </div>
    <h1>Predict your <em>heart<br>disease risk</em></h1>
    <p class="subtitle">Enter patient vitals & lifestyle data below. Our ML models return a probability score in seconds.</p>

    <!-- ECG SVG decoration -->
    <svg class="ecg-line" viewBox="0 0 600 60" fill="none" xmlns="http://www.w3.org/2000/svg">
      <polyline points="0,30 80,30 100,30 110,10 120,50 130,5 145,55 155,30 200,30 220,30 230,20 240,40 250,30 600,30"
                stroke="#ff3d5a" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
        <animate attributeName="stroke-dasharray" from="0,2000" to="2000,0" dur="2.5s" fill="freeze" calcMode="spline" keySplines="0.4 0 0.2 1"/>
      </polyline>
    </svg>
  </div>

  <form id="form">

    <!-- ── Section 1: Demographics & Vitals ── -->
    <div class="card">
      <div class="card-title">Patient Demographics &amp; Vitals</div>
      <div class="grid">

        <!-- Age -->
        <div class="field" style="grid-column: span 1">
          <label>Age (years)</label>
          <div class="num-display"><span id="ageVal">50</span> <small>yrs</small></div>
          <input type="range" name="Age" min="18" max="100" value="50" oninput="document.getElementById('ageVal').textContent=this.value">
        </div>

        <!-- Gender -->
        <div class="field">
          <label>Gender</label>
          <div class="toggle-group gender">
            <input type="radio" name="Gender" id="gM" value="Male" checked>
            <label for="gM">Male</label>
            <input type="radio" name="Gender" id="gF" value="Female">
            <label for="gF">Female</label>
          </div>
        </div>

        <!-- Blood Pressure -->
        <div class="field">
          <label>Blood Pressure <small style="text-transform:none;font-size:10px">(mmHg)</small></label>
          <input type="number" name="Blood Pressure" min="80" max="220" value="120" placeholder="e.g. 120">
        </div>

        <!-- Cholesterol -->
        <div class="field">
          <label>Cholesterol Level <small style="text-transform:none;font-size:10px">(mg/dL)</small></label>
          <input type="number" name="Cholesterol Level" min="100" max="400" value="200" placeholder="e.g. 200">
        </div>

        <!-- BMI -->
        <div class="field">
          <label>BMI</label>
          <input type="number" name="BMI" min="10" max="60" step="0.1" value="25.0" placeholder="e.g. 25.0">
        </div>

        <!-- Sleep Hours -->
        <div class="field">
          <label>Sleep Hours <small style="text-transform:none;font-size:10px">(per night)</small></label>
          <input type="number" name="Sleep Hours" min="2" max="12" step="0.5" value="7" placeholder="e.g. 7">
        </div>

        <!-- Triglyceride -->
        <div class="field">
          <label>Triglycerides <small style="text-transform:none;font-size:10px">(mg/dL)</small></label>
          <input type="number" name="Triglyceride Level" min="50" max="700" value="150" placeholder="e.g. 150">
        </div>

        <!-- Fasting Blood Sugar -->
        <div class="field">
          <label>Fasting Blood Sugar <small style="text-transform:none;font-size:10px">(mg/dL)</small></label>
          <input type="number" name="Fasting Blood Sugar" min="60" max="400" value="90" placeholder="e.g. 90">
        </div>

        <!-- CRP Level -->
        <div class="field">
          <label>CRP Level <small style="text-transform:none;font-size:10px">(mg/L)</small></label>
          <input type="number" name="CRP Level" min="0" max="30" step="0.1" value="2.0" placeholder="e.g. 2.0">
        </div>

        <!-- Homocysteine -->
        <div class="field">
          <label>Homocysteine <small style="text-transform:none;font-size:10px">(µmol/L)</small></label>
          <input type="number" name="Homocysteine Level" min="3" max="40" step="0.1" value="8.0" placeholder="e.g. 8.0">
        </div>

      </div>
    </div>

    <!-- ── Section 2: Lifestyle Factors ── -->
    <div class="card">
      <div class="card-title">Lifestyle Factors</div>
      <div class="grid">

        <div class="field">
          <label>Exercise Habits</label>
          <div class="toggle-group tri">
            <input type="radio" name="Exercise Habits" id="exL" value="Low" checked>
            <label for="exL">Low</label>
            <input type="radio" name="Exercise Habits" id="exM" value="Medium">
            <label for="exM">Med</label>
            <input type="radio" name="Exercise Habits" id="exH" value="High">
            <label for="exH">High</label>
          </div>
        </div>

        <div class="field">
          <label>Alcohol Consumption</label>
          <div class="toggle-group tri">
            <input type="radio" name="Alcohol Consumption" id="alL" value="Low" checked>
            <label for="alL">Low</label>
            <input type="radio" name="Alcohol Consumption" id="alM" value="Medium">
            <label for="alM">Med</label>
            <input type="radio" name="Alcohol Consumption" id="alH" value="High">
            <label for="alH">High</label>
          </div>
        </div>

        <div class="field">
          <label>Stress Level</label>
          <div class="toggle-group tri">
            <input type="radio" name="Stress Level" id="stL" value="Low" checked>
            <label for="stL">Low</label>
            <input type="radio" name="Stress Level" id="stM" value="Medium">
            <label for="stM">Med</label>
            <input type="radio" name="Stress Level" id="stH" value="High">
            <label for="stH">High</label>
          </div>
        </div>

        <div class="field">
          <label>Sugar Consumption</label>
          <div class="toggle-group tri">
            <input type="radio" name="Sugar Consumption" id="sgL" value="Low" checked>
            <label for="sgL">Low</label>
            <input type="radio" name="Sugar Consumption" id="sgM" value="Medium">
            <label for="sgM">Med</label>
            <input type="radio" name="Sugar Consumption" id="sgH" value="High">
            <label for="sgH">High</label>
          </div>
        </div>

        <div class="field">
          <label>Smoking</label>
          <div class="toggle-group bin">
            <input type="radio" name="Smoking" id="smN" value="No" checked>
            <label for="smN">No</label>
            <input type="radio" name="Smoking" id="smY" value="Yes">
            <label for="smY">Yes</label>
          </div>
        </div>

      </div>
    </div>

    <!-- ── Section 3: Medical History ── -->
    <div class="card">
      <div class="card-title">Medical History</div>
      <div class="grid">

        <div class="field">
          <label>Family Heart Disease</label>
          <div class="toggle-group bin">
            <input type="radio" name="Family Heart Disease" id="fhN" value="No" checked>
            <label for="fhN">No</label>
            <input type="radio" name="Family Heart Disease" id="fhY" value="Yes">
            <label for="fhY">Yes</label>
          </div>
        </div>

        <div class="field">
          <label>Diabetes</label>
          <div class="toggle-group bin">
            <input type="radio" name="Diabetes" id="dbN" value="No" checked>
            <label for="dbN">No</label>
            <input type="radio" name="Diabetes" id="dbY" value="Yes">
            <label for="dbY">Yes</label>
          </div>
        </div>

        <div class="field">
          <label>High Blood Pressure</label>
          <div class="toggle-group bin">
            <input type="radio" name="High Blood Pressure" id="hbpN" value="No" checked>
            <label for="hbpN">No</label>
            <input type="radio" name="High Blood Pressure" id="hbpY" value="Yes">
            <label for="hbpY">Yes</label>
          </div>
        </div>

        <div class="field">
          <label>Low HDL Cholesterol</label>
          <div class="toggle-group bin">
            <input type="radio" name="Low HDL Cholesterol" id="hdlN" value="No" checked>
            <label for="hdlN">No</label>
            <input type="radio" name="Low HDL Cholesterol" id="hdlY" value="Yes">
            <label for="hdlY">Yes</label>
          </div>
        </div>

        <div class="field">
          <label>High LDL Cholesterol</label>
          <div class="toggle-group bin">
            <input type="radio" name="High LDL Cholesterol" id="ldlN" value="No" checked>
            <label for="ldlN">No</label>
            <input type="radio" name="High LDL Cholesterol" id="ldlY" value="Yes">
            <label for="ldlY">Yes</label>
          </div>
        </div>

      </div>
    </div>

    <!-- ── Model Selector + Submit ── -->
    <div class="card">
      <div class="card-title">Prediction Model</div>
      <div class="model-row">
        <div class="model-option">
          <input type="radio" name="model" id="mRF" value="rf" checked>
          <label for="mRF"><span class="dot"></span> Random Forest</label>
        </div>
        <div class="model-option">
          <input type="radio" name="model" id="mLR" value="lr">
          <label for="mLR"><span class="dot"></span> Logistic Regression</label>
        </div>
      </div>
      <br>
      <button type="submit" class="btn-predict" id="submitBtn">
        <span id="btnText">Analyse Risk</span>
        <div class="spinner" id="spinner"></div>
        <svg id="btnIcon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
      </button>
    </div>

  </form>

  <!-- Result Panel -->
  <div id="resultPanel">
    <div class="card-title">Prediction Result</div>
    <div id="riskHeadline" class="result-headline"></div>
    <div id="riskSub" class="result-sub"></div>

    <div class="risk-meter">
      <div class="risk-bar-track">
        <div class="risk-bar-fill" id="riskBar"></div>
      </div>
      <div class="risk-labels">
        <span>0%</span><span>Low Risk</span><span>Moderate</span><span>High Risk</span><span>100%</span>
      </div>
    </div>

    <div class="stats-row">
      <div class="stat-box">
        <div class="val" id="statProb">—</div>
        <div class="lbl">Probability Score</div>
      </div>
      <div class="stat-box">
        <div class="val" id="statPred">—</div>
        <div class="lbl">Prediction</div>
      </div>
      <div class="stat-box">
        <div class="val" id="statModel">—</div>
        <div class="lbl">Model Used</div>
      </div>
    </div>

    <div class="disclaimer">
      ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational and research purposes only.
      The output is generated by a machine learning model and does not constitute medical advice.
      Always consult a qualified healthcare professional for diagnosis and treatment.
    </div>
  </div>

  <div class="footer">
    Built with Flask + Scikit-learn &nbsp;·&nbsp; <a href="#">CardioScan AI</a>
  </div>

</div>

<script>
document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const btn = document.getElementById('submitBtn');
  const spinner = document.getElementById('spinner');
  const btnIcon = document.getElementById('btnIcon');
  const btnText = document.getElementById('btnText');

  btn.disabled = true;
  spinner.style.display = 'block';
  btnIcon.style.display = 'none';
  btnText.textContent = 'Analysing...';

  const formData = new FormData(e.target);
  const data = Object.fromEntries(formData);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data)
    });
    const result = await response.json();

    // Show result
    const panel = document.getElementById('resultPanel');
    panel.classList.add('show');
    panel.scrollIntoView({behavior:'smooth', block:'start'});

    const level = result.risk_level.toLowerCase();
    const headline = document.getElementById('riskHeadline');
    const sub = document.getElementById('riskSub');
    const bar = document.getElementById('riskBar');

    headline.className = 'result-headline ' + level;
    bar.className = 'risk-bar-fill ' + level;

    if (level === 'low') {
      headline.textContent = '✓ Low Risk';
      sub.textContent = 'The model predicts a low probability of heart disease for this patient profile.';
    } else if (level === 'moderate') {
      headline.textContent = '~ Moderate Risk';
      sub.textContent = 'Some risk factors are present. Lifestyle changes and medical consultation are advised.';
    } else {
      headline.textContent = '⚠ High Risk';
      sub.textContent = 'Significant risk factors detected. Immediate consultation with a cardiologist is strongly recommended.';
    }

    document.getElementById('statProb').textContent = result.probability + '%';
    document.getElementById('statPred').textContent = result.prediction === 'Yes' ? '🔴 Positive' : '🟢 Negative';
    document.getElementById('statModel').textContent = data.model === 'rf' ? 'Rnd Forest' : 'Log Reg';

    setTimeout(() => { bar.style.width = result.probability + '%'; }, 100);

  } catch(err) {
    alert('Prediction error: ' + err.message);
  } finally {
    btn.disabled = false;
    spinner.style.display = 'none';
    btnIcon.style.display = 'block';
    btnText.textContent = 'Analyse Risk';
  }
});
</script>
</body>
</html>
'''
    return render_template_string(html)

@app.route('/health')
def health(): return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
