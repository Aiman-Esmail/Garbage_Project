import streamlit as st
import os
import json
import datetime
import io
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.units import cm

load_dotenv()

st.set_page_config(
    page_title="MüllAI — Intelligente Abfallerkennung",
    page_icon="♻️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

* { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.stApp { background: #0a0f0a; color: #e8f5e8; }

.hero {
    background: linear-gradient(135deg, #0d1f0d 0%, #1a3a1a 50%, #0d2a1a 100%);
    border: 1px solid #2d5a2d;
    border-radius: 16px;
    padding: 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(34,197,94,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.3);
    color: #22c55e;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 48px; font-weight: 700;
    color: #ffffff; margin: 0 0 8px 0; line-height: 1.1;
}
.hero h1 span { color: #22c55e; }
.hero p { color: #9ca3af; font-size: 16px; margin: 0; max-width: 500px; }
.stats-bar { display: flex; gap: 32px; margin-top: 28px; }
.stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px; font-weight: 700; color: #22c55e; line-height: 1;
}
.stat-label { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }

.result-card {
    background: linear-gradient(135deg, #111811, #0d1f0d);
    border: 1px solid #2d5a2d;
    border-radius: 16px; padding: 28px; margin: 12px 0;
}
.result-label { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.result-value { font-family: 'Space Grotesk', sans-serif; font-size: 36px; font-weight: 700; color: #ffffff; margin: 0; }
.confidence-bar-bg { background: #1f2d1f; border-radius: 100px; height: 8px; margin-top: 12px; overflow: hidden; }
.confidence-bar-fill { height: 100%; background: linear-gradient(90deg, #16a34a, #22c55e); border-radius: 100px; }
.tip-box {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    border-left: 4px solid #22c55e;
    border-radius: 8px; padding: 16px 20px; margin: 16px 0;
    color: #bbf7d0; font-size: 14px;
}
.correction-box {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 12px; padding: 20px; margin: 16px 0;
}
.correction-title { color: #fbbf24; font-weight: 600; font-size: 14px; margin-bottom: 12px; }
.success-box {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 8px; padding: 16px; margin: 12px 0;
    color: #22c55e; font-size: 14px; text-align: center;
}
.category-pill {
    display: inline-block;
    background: #1a2e1a; border: 1px solid #2d5a2d;
    color: #86efac; padding: 4px 12px;
    border-radius: 100px; font-size: 12px; font-weight: 500;
    margin: 3px;
}
.footer {
    text-align: center; color: #374151; font-size: 12px;
    margin-top: 48px; padding-top: 24px;
    border-top: 1px solid #1f2d1f;
}

/* Login Page */
.login-container {
    max-width: 420px;
    margin: 80px auto;
    background: linear-gradient(135deg, #0d1f0d, #111811);
    border: 1px solid #2d5a2d;
    border-radius: 20px;
    padding: 48px 40px;
}
.login-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 36px; font-weight: 700;
    color: #22c55e; text-align: center;
    margin-bottom: 4px;
}
.login-subtitle {
    text-align: center; color: #6b7280;
    font-size: 13px; margin-bottom: 32px;
}
</style>
""", unsafe_allow_html=True)

# ── Users Database ────────────────────────────────────────────────────────────
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    # Default demo users
    return {
        "demo@muellaI.de": {"password": "demo1234", "company": "Demo GmbH", "role": "admin"},
        "test@firma.de": {"password": "test1234", "company": "Test Firma AG", "role": "user"},
    }

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def check_login(email, password):
    users = load_users()
    # Case insensitive email check
    email_lower = email.lower().strip()
    for user_email, user_data in users.items():
        if user_email.lower() == email_lower and user_data["password"] == password:
            return user_data
    return None

# ── Login Page ────────────────────────────────────────────────────────────────
def show_login():
    st.markdown("""
    <div class='login-container'>
        <div class='login-logo'>♻️ MüllAI</div>
        <div class='login-subtitle'>Intelligente Abfallklassifizierung für Unternehmen</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("#### 🔐 Anmelden")
        email = st.text_input("E-Mail", placeholder="ihre@firma.de")
        password = st.text_input("Passwort", type="password", placeholder="••••••••")

        if st.button("Anmelden →", use_container_width=True, type="primary"):
            user = check_login(email, password)
            if user:
                st.session_state["logged_in"] = True
                st.session_state["user"] = user
                st.session_state["email"] = email
                st.rerun()
            else:
                st.error("❌ E-Mail oder Passwort falsch.")

        st.divider()
        st.markdown("""
        <div style='text-align:center; font-size:12px; color:#4b5563;'>
        Demo-Zugang: demo@muellaI.de / demo1234
        </div>
        """, unsafe_allow_html=True)

# ── Auth Check ────────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    show_login()
    st.stop()

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES = [
    'Battery', 'Biological', 'Brown-Glass', 'Cardboard',
    'Clothes', 'Green-Glass', 'Metal', 'Paper',
    'Plastic', 'Shoes', 'Trash', 'White-Glass'
]
CLASSES_DE = {
    'Battery': 'Batterie', 'Biological': 'Bioabfall',
    'Brown-Glass': 'Braunglas', 'Cardboard': 'Karton',
    'Clothes': 'Kleidung', 'Green-Glass': 'Grünglas',
    'Metal': 'Metall', 'Paper': 'Papier',
    'Plastic': 'Kunststoff', 'Shoes': 'Schuhe',
    'Trash': 'Restmüll', 'White-Glass': 'Weißglas',
}
TONNE_DE = {
    'Battery': '🔋 Sondermüll / Rückgabestelle',
    'Biological': '🟤 Biotonne',
    'Brown-Glass': '🟫 Altglascontainer (Braun)',
    'Cardboard': '📦 Papiertonne / Blaue Tonne',
    'Clothes': '👕 Altkleidercontainer',
    'Green-Glass': '🟢 Altglascontainer (Grün)',
    'Metal': '🥫 Gelbe Tonne / Wertstofftonne',
    'Paper': '📄 Papiertonne / Blaue Tonne',
    'Plastic': '♻️ Gelbe Tonne / Gelber Sack',
    'Shoes': '👟 Altkleidercontainer',
    'Trash': '🗑️ Restmülltonne',
    'White-Glass': '⬜ Altglascontainer (Weiß)',
}
RECYCLING_TIPS_DE = {
    'Battery': '⚠️ Batterien gehören NICHT in den Hausmüll! Abgabe im Handel oder an Sammelstellen.',
    'Biological': '✅ In die Biotonne oder kompostieren. Kein Fleisch oder gekochte Speisen.',
    'Brown-Glass': '✅ Deckel entfernen, ausspülen und in den Braunglas-Container.',
    'Cardboard': '✅ Kartons flach falten und in die blaue Tonne oder Papiertonne.',
    'Clothes': '✅ Saubere Kleidung in Altkleidercontainer. Beschädigte Kleidung in den Restmüll.',
    'Green-Glass': '✅ Deckel entfernen, ausspülen und in den Grünglas-Container.',
    'Metal': '✅ Dosen ausspülen und in die gelbe Tonne oder den gelben Sack.',
    'Paper': '✅ Sauberes Papier in die blaue Tonne. Verschmutztes Papier in den Restmüll.',
    'Plastic': '✅ Verpackungen ausspülen und in die gelbe Tonne oder den gelben Sack.',
    'Shoes': '✅ Paarweise zusammenbinden und in den Altkleidercontainer.',
    'Trash': '⚠️ Gehört in die graue Restmülltonne. Prüfen ob Recycling möglich ist.',
    'White-Glass': '✅ Deckel entfernen, ausspülen und in den Weißglas-Container.',
}

IMG_SIZE = (128, 128)
KAGGLE_DATASET = "aimanesmail/garbage-classifier-model"
MODEL_DIR = "model/garbage_classifier_saved"
CORRECTIONS_FILE = "corrections.json"
HISTORY_FILE = "history.json"

# ── Load/Save Corrections ─────────────────────────────────────────────────────
def load_corrections():
    if os.path.exists(CORRECTIONS_FILE):
        with open(CORRECTIONS_FILE, "r") as f:
            return json.load(f)
    return []

def save_correction(predicted, correct):
    corrections = load_corrections()
    corrections.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "predicted": predicted,
        "correct": correct
    })
    with open(CORRECTIONS_FILE, "w") as f:
        json.dump(corrections, f, indent=2)

# ── Load/Save History ─────────────────────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(label, confidence):
    history = load_history()
    history.insert(0, {
        "time": datetime.datetime.now().strftime("%d.%m %H:%M"),
        "label": label,
        "label_de": CLASSES_DE[label],
        "confidence": round(confidence, 1)
    })
    history = history[:20]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# ── Generate PDF Report ───────────────────────────────────────────────────────
def generate_pdf(label, label_de, confidence, tonne, tip, filename):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    green = colors.HexColor("#16a34a")
    dark_text = colors.HexColor("#111827")
    gray_text = colors.HexColor("#4b5563")
    light_gray = colors.HexColor("#6b7280")

    title_style = ParagraphStyle('title', fontSize=28, fontName='Helvetica-Bold',
                                  textColor=green, spaceAfter=4)
    subtitle_style = ParagraphStyle('subtitle', fontSize=11, fontName='Helvetica',
                                     textColor=gray_text, spaceAfter=16)
    label_style = ParagraphStyle('label', fontSize=9, fontName='Helvetica',
                                  textColor=light_gray, spaceAfter=2, leading=14)
    value_style = ParagraphStyle('value', fontSize=15, fontName='Helvetica-Bold',
                                  textColor=dark_text, spaceAfter=10)
    tip_style = ParagraphStyle('tip', fontSize=11, fontName='Helvetica',
                                textColor=dark_text, spaceAfter=6, leading=16)
    footer_style = ParagraphStyle('footer', fontSize=8, fontName='Helvetica',
                                   textColor=light_gray, leading=12)

    story = []

    # Header
    story.append(Paragraph("MullAI", title_style))
    story.append(Paragraph("Intelligente Abfallklassifizierung — Analysebericht", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=green))
    story.append(Spacer(1, 0.5*cm))

    # Date
    now = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
    story.append(Paragraph("ANALYSEDATUM", label_style))
    story.append(Paragraph(now, value_style))
    story.append(Spacer(1, 0.3*cm))

    # Result table
    tonne_clean = tonne
    for emoji in ["🔋","🟤","🟫","📦","👕","🟢","🥫","📄","♻️","👟","🗑️","⬜"]:
        tonne_clean = tonne_clean.replace(emoji, "").strip()

    data = [
        ["Erkannter Abfall", label_de],
        ["Konfidenz", f"{confidence:.1f}%"],
        ["Entsorgung", tonne_clean],
    ]
    table = Table(data, colWidths=[5*cm, 12*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f0fdf4")),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (0, -1), gray_text),
        ('TEXTCOLOR', (1, 0), (1, -1), dark_text),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#d1fae5")),
        ('PADDING', (0, 0), (-1, -1), 12),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor("#f0fdf4"), colors.HexColor("#ffffff")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.6*cm))

    # Tip
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d1fae5")))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("RECYCLING-HINWEIS", label_style))
    tip_clean = tip
    for emoji in ["✅","⚠️"]:
        tip_clean = tip_clean.replace(emoji, "").strip()
    story.append(Paragraph(tip_clean, tip_style))

    story.append(Spacer(1, 1.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d1fae5")))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "MullAI v1.0 | Powered by MobileNetV2 | Entwickelt von Aiman Esmail<br/>"
        "Hamburg, Schleswig-Holstein, Deutschland",
        footer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ── Download Model ────────────────────────────────────────────────────────────
def download_model():
    if os.path.exists(MODEL_DIR):
        return True
    try:
        import kaggle
        os.makedirs(MODEL_DIR, exist_ok=True)
        files = [
            "saved_model.pb", "fingerprint.pb", "keras_metadata.pb",
            "variables.data-00000-of-00001", "variables.index"
        ]
        for f in files:
            try:
                kaggle.api.dataset_download_file(KAGGLE_DATASET, file_name=f, path=MODEL_DIR, force=True)
            except:
                pass
        variables_dir = os.path.join(MODEL_DIR, "variables")
        os.makedirs(variables_dir, exist_ok=True)
        for f in ["variables.data-00000-of-00001", "variables.index"]:
            src = os.path.join(MODEL_DIR, f)
            dst = os.path.join(variables_dir, f)
            if os.path.exists(src):
                os.rename(src, dst)
        return True
    except Exception as e:
        st.error(f"Modell-Download fehlgeschlagen: {e}")
        return False

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="KI-Modell wird geladen...")
def load_model():
    if not os.path.exists(MODEL_DIR):
        with st.spinner("Modell wird heruntergeladen..."):
            if not download_model():
                return None, None
    try:
        model = tf.saved_model.load(MODEL_DIR)
        infer = model.signatures["serving_default"]
        return ("saved_model", infer), MODEL_DIR
    except Exception as e:
        st.error(f"Modell konnte nicht geladen werden: {e}")
        return None, None

# ── Predict ───────────────────────────────────────────────────────────────────
def preprocess(image):
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def predict(model_tuple, image):
    tensor = preprocess(image)
    _, model = model_tuple
    input_tensor = tf.constant(tensor, dtype=tf.float32)
    result = model(input_tensor)
    preds = list(result.values())[0].numpy()[0]
    idx = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx]) * 100, preds

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Systemstatus")

    # User info
    user = st.session_state.get("user", {})
    email = st.session_state.get("email", "")
    st.markdown(f"""
    <div style='background:#111811; border:1px solid #2d5a2d; border-radius:8px; padding:12px; margin-bottom:12px;'>
        <div style='color:#22c55e; font-weight:600; font-size:13px;'>🏢 {user.get('company', '')}</div>
        <div style='color:#6b7280; font-size:11px;'>{email}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚪 Abmelden", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["user"] = {}
        st.rerun()

    st.divider()
    model_tuple, model_path = load_model()
    if model_tuple:
        st.success("✅ Modell geladen")
    else:
        st.error("❌ Modell nicht gefunden")

    st.divider()
    st.markdown("**Erkannte Kategorien**")
    st.markdown(
        "<div>" + "".join([f"<span class='category-pill'>{v}</span>" for v in CLASSES_DE.values()]) + "</div>",
        unsafe_allow_html=True
    )

    # Corrections counter
    corrections = load_corrections()
    if corrections:
        st.divider()
        st.markdown(f"**📊 Korrekturen: {len(corrections)}**")
        st.caption("Hilft das Modell zu verbessern")

    # History
    history = load_history()
    if history:
        st.divider()
        st.markdown("**🕐 Letzte Analysen**")
        for h in history[:5]:
            st.markdown(f"""
            <div style='background:#111811; border:1px solid #2d5a2d; border-radius:8px;
                        padding:8px 12px; margin:4px 0; font-size:12px;'>
                <span style='color:#22c55e; font-weight:600;'>{h['label_de']}</span>
                <span style='color:#6b7280;'> — {h['confidence']}%</span><br>
                <span style='color:#4b5563;'>{h['time']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='font-size:12px; color:#4b5563;'>
    MüllAI v1.0<br>Powered by MobileNetV2<br>Genauigkeit: 94.33%
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-badge'>♻️ KI-gestützte Abfallerkennung</div>
    <h1>Müll<span>AI</span></h1>
    <p>Intelligente Abfallklassifizierung für Unternehmen — präzise, schnell und DSGVO-konform.</p>
    <div class='stats-bar'>
        <div class='stat'><div class='stat-value'>94%</div><div class='stat-label'>Genauigkeit</div></div>
        <div class='stat'><div class='stat-value'>12</div><div class='stat-label'>Kategorien</div></div>
        <div class='stat'><div class='stat-value'>&lt;2s</div><div class='stat-label'>Analysezeit</div></div>
        <div class='stat'><div class='stat-value'>15K+</div><div class='stat-label'>Trainingsbilder</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("#### 📸 Bild hochladen")
    uploaded_file = st.file_uploader(
        "Bild auswählen",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Hochgeladenes Bild", use_column_width=True)

with col_result:
    if uploaded_file:
        st.markdown("#### 🔍 Analyseergebnis")

        if model_tuple is None:
            st.error("⚠️ Modell nicht verfügbar.")
            st.stop()

        with st.spinner("Wird analysiert..."):
            label, confidence, probs = predict(model_tuple, img)
            save_history(label, confidence)

        label_de = CLASSES_DE[label]
        tonne = TONNE_DE[label]
        tip = RECYCLING_TIPS_DE[label]

        st.markdown(f"""
        <div class='result-card'>
            <div class='result-label'>Erkannter Abfall</div>
            <div class='result-value'>{label_de}</div>
            <div class='result-label' style='margin-top:16px;'>Konfidenz</div>
            <div class='result-value' style='font-size:28px; color:#22c55e;'>{confidence:.1f}%</div>
            <div class='confidence-bar-bg'>
                <div class='confidence-bar-fill' style='width:{confidence}%;'></div>
            </div>
        </div>
        <div class='result-card'>
            <div class='result-label'>Entsorgung</div>
            <div style='font-size:20px; font-weight:600; color:#fff; margin-top:4px;'>{tonne}</div>
        </div>
        <div class='tip-box'>{tip}</div>
        """, unsafe_allow_html=True)

        # ── PDF Download ───────────────────────────────────────────────────
        pdf_buffer = generate_pdf(label, label_de, confidence, tonne, tip,
                                   f"MuellAI_{label_de}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
        st.download_button(
            label="📄 PDF-Bericht herunterladen",
            data=pdf_buffer,
            file_name=f"MuellAI_{label_de}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        # ── Correction Button ──────────────────────────────────────────────
        st.divider()
        st.markdown("⚠️ **Falsch erkannt? Bitte korrigieren:**")

        classes_de_list = [CLASSES_DE[c] for c in CLASSES]
        correct_label_de = st.selectbox(
            "Richtige Kategorie:",
            options=classes_de_list,
            index=classes_de_list.index(label_de)
        )

        if st.button("✅ Korrektur speichern", use_container_width=True):
            correct_label_en = [k for k, v in CLASSES_DE.items() if v == correct_label_de][0]
            if correct_label_en != label:
                save_correction(label, correct_label_en)
                st.success(f"Danke! Korrektur gespeichert: {label_de} → {correct_label_de}")
            else:
                st.info("Die Erkennung war bereits korrekt!")

    else:
        st.markdown("""
        <div style='height:300px; display:flex; align-items:center; justify-content:center;
                    background:#111811; border-radius:12px; border:1px solid #2d5a2d;'>
            <div style='text-align:center; color:#4b5563;'>
                <div style='font-size:48px;'>♻️</div>
                <div style='margin-top:12px; font-size:14px;'>
                    Laden Sie ein Bild hoch,<br>um die Analyse zu starten
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Chart ─────────────────────────────────────────────────────────────────────
if uploaded_file and 'probs' in dir():
    st.divider()
    st.markdown("#### 📊 Konfidenz pro Kategorie")
    prob_df = pd.DataFrame({
        "Kategorie": [CLASSES_DE[c] for c in CLASSES],
        "Konfidenz": probs
    }).sort_values("Konfidenz", ascending=False)
    st.bar_chart(prob_df.set_index("Kategorie")["Konfidenz"])

    with st.expander("🔬 Technische Details"):
        st.dataframe(prob_df.style.format({"Konfidenz": "{:.4f}"}))
        st.caption(f"Modellpfad: `{model_path}`")
        st.caption(f"Eingabegröße: {IMG_SIZE[0]}x{IMG_SIZE[1]}x3")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    MüllAI © 2026 — Entwickelt von Aiman Esmail | Hamburg, Deutschland<br>
    Powered by MobileNetV2 + TensorFlow + Streamlit
</div>
""", unsafe_allow_html=True)
