# Heart Disease Prediction – End-to-End ML Deployment

An end-to-end machine learning project that predicts whether a patient is
at risk of heart disease based on clinical parameters. The model is
served through a Flask REST API and deployed as a live web service on
Render.

## 🔗 Live Deployment

**Render URL:** _`<PASTE YOUR RENDER DEPLOYMENT URL HERE AFTER DEPLOYING>`_

> Replace the line above with your actual Render URL, e.g.
> `https://heart-disease-deployment.onrender.com`

## 📁 Repository Structure

```
HeartDiseaseDeployment/
│
├── app.py                # Flask REST API
├── model.pkl              # Trained classification model
├── feature_names.pkl      # Feature order expected by the model
├── requirements.txt       # Python dependencies
├── Procfile                # Tells Render how to start the app
├── README.md
├── train_model.py         # Data prep + model training script
├── heart.csv               # Dataset
├── templates/
│   └── index.html          # Optional simple landing page
└── static/                 # (Optional) static assets
```

## 📊 Dataset

Heart Disease Prediction Dataset (clinical parameters such as age, sex,
chest pain type, resting blood pressure, cholesterol, fasting blood
sugar, ECG results, max heart rate, exercise-induced angina, ST
depression, slope, number of major vessels, and thalassemia), with a
binary `target` column (1 = heart disease present, 0 = not present).

> **Note:** `heart.csv` in this repo follows the exact same 14-column
> schema as the Kaggle "Heart Disease Dataset" (johnsmith88). If you
> have downloaded the original Kaggle CSV, you can simply replace
> `heart.csv` with it and re-run `train_model.py` — no other code
> changes are required.

## ⚙️ Setup — Run Locally

```bash
# 1. Clone the repository
git clone <YOUR_GITHUB_REPO_URL>
cd HeartDiseaseDeployment

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Re)train the model — this regenerates model.pkl and feature_names.pkl
python train_model.py

# 5. Run the Flask API
python app.py
```

The API will be available at `http://127.0.0.1:5000`.

## 🧠 Model

- **Algorithm:** Random Forest Classifier (`n_estimators=200, max_depth=6`)
- **Train/Test Split:** 80% / 20%, stratified on the target
- **Evaluation Metric:** Accuracy Score (~0.88 on the held-out test set)
- **Serialization:** `joblib.dump()` → `model.pkl`

## 🌐 API Usage

### `GET /`
Health/info endpoint — returns the expected input fields.

### `GET /health`
Simple health check for uptime monitoring.

### `POST /predict`
Accepts patient details as JSON and returns a prediction.

**Request body:**
```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

**Response:**
```json
{
  "prediction": "Heart Disease Detected",
  "probability_of_heart_disease": 0.63
}
```

**Test with curl:**
```bash
curl -X POST https://<your-render-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

## ☁️ Deploying on Render

1. Push this repository to a **public** GitHub repo.
2. Go to [render.com](https://render.com) → **New** → **Web Service**.
3. Connect your GitHub repository.
4. Configure the service:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app` (already set in the `Procfile`)
5. Click **Deploy**. Once live, copy the generated URL and paste it:
   - At the top of this README (Live Deployment section)
   - Into the Google Form submission

## 📝 Conclusion

The Random Forest classifier achieved an accuracy of approximately 88%
on the held-out test set, indicating strong performance in
distinguishing patients at risk of heart disease from those without it,
based on standard clinical parameters. Precision and recall were
balanced across both classes, suggesting the model does not
systematically favor one outcome. During deployment, the main
challenges involved ensuring the Flask API validated incoming JSON
correctly, keeping the feature order between training and inference
perfectly consistent, and configuring Render's build and start commands
so the service stayed active and reachable. This project highlighted
why MLOps practices matter: version-controlled code, reproducible
training scripts, serialized models, and automated cloud deployment
together make a machine learning solution reliable, maintainable, and
usable outside a notebook — turning a one-off experiment into a real,
continuously accessible service.

## 🎓 Learning Outcomes Covered

- Building and evaluating a machine learning classification model
- Saving and loading trained models using Joblib
- Developing a REST API using Flask
- Managing project code using GitHub
- Deploying machine learning applications on the cloud using Render
- Understanding MLOps fundamentals: packaging, version control,
  deployment, and serving predictions via an API
