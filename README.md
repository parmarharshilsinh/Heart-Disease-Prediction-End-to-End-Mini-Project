# ❤️ CardioScan AI — Heart Disease Risk Predictor

A Flask web application that predicts heart disease risk using trained ML models (Random Forest + Logistic Regression).

## 📁 Project Structure

```
heart_disease_project/
├── app.py                          # Flask server (redesigned UI)
├── heart_disease.csv               # Dataset (10,000 patients, 20 features)
├── heart_disease_prediction.ipynb  # Training notebook
├── models/                         # Generated after running the notebook
│   ├── rf_model.pkl
│   ├── lr_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── features.json
└── README.md
```

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install flask scikit-learn numpy pandas
```

### 2. Train the models
Open `heart_disease_prediction.ipynb` in Jupyter and run all cells.
This saves trained models into the `models/` folder.

### 3. Start the server
```bash
python app.py
```

Visit → **http://localhost:5000**

## 🌐 UI Features
- **Age slider** — drag or type age (18–100)
- **Low / Medium / High toggles** — for Exercise, Alcohol, Stress, Sugar
- **Yes / No toggles** — for binary medical conditions
- **Number inputs** — for Blood Pressure, Cholesterol, BMI, CRP, Homocysteine, etc.
- **Model selector** — switch between Random Forest and Logistic Regression
- **Animated risk bar** — visual probability meter (Low / Moderate / High)

## ⚠️ Disclaimer
This tool is for educational purposes only. Not a substitute for professional medical advice.
