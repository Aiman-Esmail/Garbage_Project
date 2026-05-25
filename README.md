# ♻️ AI Garbage Classifier

A production-ready web application that classifies garbage images into 6 categories using a deep learning model trained on Kaggle, deployed via Streamlit and Render.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)

---

## 🗂️ Project Structure

```
Garbage_Project/
├── app.py                            # Streamlit application
├── convert_model.py                  # H5 → SavedModel conversion
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Container for Render deployment
├── .env                              # Local secrets (never commit)
├── .env.example                      # Template for environment variables
├── .gitignore
├── README.md
└── model/
    ├── Garbage_Classifier_Final_95.h5
    └── garbage_classifier_saved/     # Converted SavedModel (preferred)
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/garbage-classifier.git
cd garbage-classifier
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
```bash
cp .env.example .env
# Edit .env and add your Kaggle credentials
```

### 5. Convert Model (if needed)
```bash
python convert_model.py
```

### 6. Run the App
```bash
streamlit run app.py
```

Visit `http://localhost:8501`

---

## 🧠 Model

| Property | Value |
|---|---|
| Architecture | CNN (Transfer Learning) |
| Input Size | 224 × 224 × 3 |
| Output Classes | 6 |
| Accuracy | ~95% |
| Format | SavedModel / H5 |

### Classes
| Label | Description |
|---|---|
| Cardboard | Boxes, packaging |
| Glass | Bottles, jars |
| Metal | Cans, foil |
| Paper | Newspapers, documents |
| Plastic | Bottles, bags |
| Trash | General waste |

---

## 🐳 Docker

```bash
docker build -t garbage-classifier .
docker run -p 8501:8501 garbage-classifier
```

---

## ☁️ Deploy on Render

1. Push your code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repository
4. Set environment:
   - **Runtime:** Docker
   - **Port:** 8501
5. Add environment variables from `.env`
6. Deploy 🚀

---

## 🔒 Security Notes

- Never commit `.env` or `kaggle.json` to Git
- Use Render's environment variable settings for secrets
- Model files (`.h5`) are excluded from Git — use cloud storage or Git LFS

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| Model not found | Run `python convert_model.py` |
| TF loading error | Use SavedModel format instead of H5 |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |
| Missing packages | `pip install -r requirements.txt` |
