"""
FastAPI app that serves the trained XGBoost model with a simple web form.

Run:
  pip install fastapi uvicorn pandas scikit-learn xgboost
  uvicorn web_app:app --reload --host 0.0.0.0 --port 8000
Use:
  Open http://localhost:8000 for the form
  or POST JSON to /predict.
"""
from pathlib import Path
import pickle
from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


MODEL_PATH = Path("2.训练集构建模型/xgb_model.pkl")
FEATURES = ["Female_age", "BMI", "PLT", "FSH", "TSH"]


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


model = load_model(MODEL_PATH)


class PatientFeatures(BaseModel):
    Female_age: float = Field(..., description="Female age (years)")
    BMI: float = Field(..., description="Body mass index")
    PLT: float = Field(..., description="Platelet count")
    FSH: float = Field(..., description="FSH")
    TSH: float = Field(..., description="TSH")


app = FastAPI(title="IVF/ICSI early miscarriage risk (XGBoost)")


def to_dataframe(payload: Dict[str, float]) -> pd.DataFrame:
    # keep column order consistent with training
    df = pd.DataFrame([payload], columns=FEATURES)
    return df[FEATURES]


@app.get("/", response_class=HTMLResponse)
def index():
    # Simple HTML form that posts to /predict
    return """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>IVF/ICSI early miscarriage risk (XGBoost)</title>
      <style>
        :root {
          --bg: linear-gradient(135deg, #f7f9fc 0%, #e9f2ff 50%, #f6f7ff 100%);
          --card: #ffffff;
          --primary: #1f6feb;
          --accent: #13b8a6;
          --text: #1c2a3a;
          --muted: #5c6b7a;
          --border: #dce3ed;
          --shadow: 0 20px 45px rgba(31, 111, 235, 0.12);
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
          color: var(--text);
          background: var(--bg);
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 24px;
        }
        .shell {
          width: 100%;
          max-width: 840px;
        }
        .card {
          background: var(--card);
          border: 1px solid var(--border);
          box-shadow: var(--shadow);
          border-radius: 14px;
          padding: 28px;
        }
        .title {
          margin: 0 0 8px;
          font-size: 24px;
          font-weight: 700;
        }
        .subtitle {
          margin: 0 0 20px;
          color: var(--muted);
          font-size: 14px;
        }
        .grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 16px 18px;
        }
        label {
          display: block;
          font-weight: 600;
          margin-bottom: 6px;
          color: var(--text);
        }
        .hint {
          display: block;
          color: var(--muted);
          font-size: 12px;
          margin-top: 2px;
        }
        input[type="number"] {
          width: 100%;
          padding: 10px 12px;
          border: 1px solid var(--border);
          border-radius: 10px;
          font-size: 14px;
          transition: border-color 0.2s, box-shadow 0.2s;
        }
        input[type="number"]:focus {
          border-color: var(--primary);
          box-shadow: 0 0 0 3px rgba(31, 111, 235, 0.18);
          outline: none;
        }
        button {
          margin-top: 10px;
          padding: 12px 16px;
          width: 100%;
          border: none;
          border-radius: 12px;
          background: linear-gradient(90deg, var(--primary), var(--accent));
          color: #fff;
          font-weight: 700;
          font-size: 15px;
          cursor: pointer;
          transition: transform 0.1s ease, box-shadow 0.2s;
          box-shadow: 0 12px 30px rgba(19, 184, 166, 0.25);
        }
        button:hover { transform: translateY(-1px); }
        button:active { transform: translateY(0); }
        .result {
          margin-top: 18px;
          padding: 14px 16px;
          border-radius: 10px;
          border: 1px dashed var(--border);
          background: #f8fbff;
          min-height: 48px;
          font-weight: 600;
        }
        .row {
          margin-bottom: 2px;
          font-size: 14px;
        }
        .row span { color: var(--muted); }
      </style>
    </head>
    <body>
      <div class="shell">
        <div class="card">
          <h2 class="title">IVF/ICSI early miscarriage risk (XGBoost)</h2>
          <p class="subtitle">Enter patient features to estimate probability of early miscarriage risk. Units are indicated for each feature.</p>
          <form id="form">
            <div class="grid">
              <div>
                <label>Female age <span class="hint">(years)</span></label>
                <input type="number" step="0.01" name="Female_age" required />
              </div>
              <div>
                <label>BMI <span class="hint">(kg/m^2)</span></label>
                <input type="number" step="0.01" name="BMI" required />
              </div>
              <div>
                <label>PLT <span class="hint">(10^9/L)</span></label>
                <input type="number" step="0.01" name="PLT" required />
              </div>
              <div>
                <label>FSH <span class="hint">(IU/L)</span></label>
                <input type="number" step="0.01" name="FSH" required />
              </div>
              <div>
                <label>TSH <span class="hint">(mIU/L)</span></label>
                <input type="number" step="0.01" name="TSH" required />
              </div>
            </div>
            <button type="submit">Predict</button>
          </form>
          <div class="result" id="result">Prediction will appear here.</div>
        </div>
      </div>
      <script>
        const form = document.getElementById("form");
        const resultBox = document.getElementById("result");
        form.addEventListener("submit", async (e) => {
          e.preventDefault();
          const data = Object.fromEntries(new FormData(form).entries());
          for (const k in data) data[k] = Number(data[k]);
          resultBox.textContent = "Predicting...";
          try {
            const res = await fetch("/predict", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(data),
            });
            const json = await res.json();
            if (res.ok) {
              resultBox.innerHTML = `
                <div class="row">Probability: <span>${json.probability.toFixed(4)}</span></div>
                <div class="row">Label (>=0.5 => 1): <span>${json.label}</span></div>
              `;
            } else {
              resultBox.textContent = `Error: ${json.detail}`;
            }
          } catch (err) {
            resultBox.textContent = `Error: ${err}`;
          }
        });
      </script>
    </body>
    </html>
    """


@app.post("/predict")
def predict(features: PatientFeatures):
    try:
        df = to_dataframe(features.model_dump())
        proba = float(model.predict_proba(df)[0, 1])
        label = int(proba >= 0.5)
        return {"probability": proba, "label": label}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
