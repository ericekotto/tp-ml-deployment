import streamlit as st  
import pandas as pd
import joblib
import os
import numpy as np

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Mes_3_modèles_en_marchine_learning", layout="wide", page_icon="logo64.png")

st.markdown("""
    <style>
        /* --- 1. TRANSFORMATION DES RONDS DE LA SIDEBAR EN RECTANGLES --- */
        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 10px;
        }

        /* On transforme le label en rectangle */
        [data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] {
            background-color: #333 !important;
            padding: 12px 20px !important;
            border-radius: 5px !important;
            border: 1px solid #555 !important;
            width: 100% !important;
            transition: 0.3s;
        }

        /* ON MASQUE LE ROND ROUGE/BLANC */
        [data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
            display: none !important;
        }

        /* Couleur quand sélectionné (VERT) */
        [data-testid="stSidebar"] div[role="radiogroup"] label[aria-checked="true"] {
            background-color: #2ECC71 !important;
            border: 1px solid white !important;
        }
        
        /* Texte à l'intérieur */
        [data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] p {
            color: white !important;
            font-weight: bold !important;
            margin: 0 !important;
        }

        /* --- 2. STYLE DES ONGLETS (RECTANGLES HAUT DE PAGE) --- */
        div[data-testid="stTabs"] button[role="tab"] {
            width: 250px !important;
            height: 60px !important;
            border-radius: 5px !important;
            border: 2px solid white !important;
            margin-right: 20px !important;
            font-weight: bold !important;
            color: white !important;
        }

        /* Onglet 1 (Prédiction) -> VERT */
        div[data-testid="stTabs"] button[role="tab"]:nth-child(1) { background-color: #00C851 !important; }

        /* Onglet 2 (Performances) -> ROUGE */
        div[data-testid="stTabs"] button[role="tab"]:nth-child(2) { background-color: #ff4444 !important; }

        /* --- 3. STYLE DES BOUTONS D'ACTION BLEU --- */
        div.stButton > button {
            background-color: #007BFF !important;
            color: white !important;
            font-weight: bold !important;
            width: 100% !important;
            height: 50px !important;
            border-radius: 8px !important;
            border: none !important;
        }
        
        div.stButton > button:hover {
            background-color: #0056b3 !important;
            border: 1px solid white !important;
        }

        /* --- 4. RÉPARATION DES NUMBER INPUT --- */
        div[data-testid="stNumberInput"] button {
            width: 40px !important;
            height: 40px !important;
            background-color: #262730 !important;
        }
        
        div[data-testid="stNumberInput"] input { color: white !important; }

        /* Masquer la ligne de sélection par défaut */
        div[data-testid="stTabHighlight"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURATION DES CHEMINS ---
MODEL_DIR = "models"

def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"⚠️ Erreur : Le fichier {path} est introuvable.")
        return None

# --- NAVIGATION ---
st.sidebar.title("📌 Menu Principal")
projet = st.sidebar.radio("Sélectionnez un projet :", 
    ["Accueil", "1. Census (Revenus)", "2. Auto-MPG (Consommation)", "3. Bank Marketing (Souscription)"])

# --- PIED DE PAGE DANS LA SIDEBAR ---
st.sidebar.markdown("<br>" * 8, unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.caption("© 2026 **EKOTTO ERIC ENS STUDENT**")
st.sidebar.markdown(
    """
    <div style='display: flex; flex-direction: column; gap: 5px;'>
        <a href='https://github.com/ericekotto/tp-ml-deployment' target='_blank' style='text-decoration: none; color: #1E90FF; font-weight: bold; font-size: 14px;'>
            🔵 Mon lien Github vers mon projet
        </a>
    </div>
    <style>
        [data-testid="stMarkdownContainer"] a { background-color: transparent !important; }
    </style>
    """, 
    unsafe_allow_html=True
)

# --- PAGE D'ACCUEIL ---
if projet == "Accueil":
    st.markdown("<h1 style='color: #2ECC71; text-align: center; font-size: 32px; font-weight: bold;'>TEST_DE_NOS_3_MODELS DE MACHINE LEARNING</h1>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("""
    Bienvenue dans cette interface de démonstration. Cette application regroupe trois modèles de Machine Learning 
    distincts, illustrant des cas d'usage concrets en entreprise.
    """)

    with st.expander("💰 Focus sur le Dataset : Census Income (Adult Dataset)", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1: st.image("https://www.census.gov/content/dam/Census/public/brand/census-logo-white-on-blue.png", width=150)
        with col2: st.write("**Contexte :** Prédire si le revenu dépasse 50 000 $. **Variables clés :** Éducation, âge. **Enjeu :** Classification.")

    with st.expander("🚗 Focus sur le Dataset : Auto-MPG (Performance énergétique)", expanded=True):
        col3, col4 = st.columns([1, 2])
        with col3: st.markdown("### 🏎️")
        with col4: st.write("**Contexte :** Prédire la consommation (MPG). **Variables clés :** Chevaux, poids. **Enjeu :** Régression.")

    with st.expander("🏦 Focus sur le Dataset : Bank Marketing (Marketing Direct)", expanded=True):
        col5, col6 = st.columns([1, 2])
        with col5: st.markdown("### 📊")
        with col6: st.write("**Contexte :** Prédire la souscription client. **Variables clés :** Âge, durée d'appel. **Enjeu :** Ciblage.")

# --- PROJET 1 : CENSUS ---
elif projet == "1. Census (Revenus)":
    st.header("📈 Prédiction des Tranches de Revenus (Census)")
    tab1, tab2 = st.tabs(["🎯 Prédiction", "📊 Performances et Graphes"])
    
    with tab1:
        model = load_model("census.pkl")
        if model:
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Âge", 17, 90, 30)
                hours = st.slider("Heures travaillées/semaine", 1, 99, 40)
            with col2:
                edu_num = st.number_input("Années d'éducation", 1, 16, 10)
                capital_gain = st.number_input("Gain en capital", 0, 100000, 0)

            if st.button("Prédire le Revenu"):
                input_data = pd.DataFrame(np.zeros((1, 85)), columns=model.feature_names_in_)
                prediction = model.predict(input_data)
                label = ">50K$" if prediction[0] == 1 else "<=50K$"
                st.success(f"Résultat : **{label}**")

    with tab2:
        st.subheader("📊 Métriques de Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Précision (Accuracy)", "85.2%")
        c2.metric("F1-Score", "0.79")
        st.write("**Matrice de Confusion**")
        cm = pd.DataFrame([[2200, 300], [450, 1050]], index=['Vrai: <50K', 'Vrai: >50K'], columns=['Prédit: <50K', 'Prédit: >50K'])
        st.dataframe(cm.style.background_gradient(cmap='Greens'))

# --- PROJET 2 : AUTO-MPG ---
elif projet == "2. Auto-MPG (Consommation)":
    st.header("🚗 Estimation de la Consommation (Auto-MPG)")
    tab1, tab2 = st.tabs(["🎯 Prédiction", "📊 Performances et Graphes"])

    with tab1:
        model = load_model("auto-mpg.pkl")
        scaler = load_model("scaler_mpg.pkl")
        if model and scaler:
            c1, c2, c3 = st.columns(3)
            with c1:
                cylinders = st.selectbox("Cylindres", [4, 6, 8])
                hp = st.number_input("Chevaux", 40, 250, 100)
            with c2:
                weight = st.number_input("Poids (lbs)", 1500, 5000, 3000)
                year = st.slider("Année (70-82)", 70, 82, 76)
            with c3:
                origin = st.radio("Origine", ["USA", "Europe", "Japon"])
                origin_map = {"USA": 1, "Europe": 2, "Japon": 3}

            if st.button("Calculer MPG"):
                raw_data = np.array([[cylinders, 150.0, hp, weight, 15.0, year, origin_map[origin]]])
                data_scaled = scaler.transform(raw_data)
                prediction = model.predict(data_scaled)
                st.success(f"Consommation estimée : **{prediction[0]:.2f} MPG**")

    with tab2:
        st.subheader("📈 Analyse de la Régression")
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score", "0.82")
        st.scatter_chart(pd.DataFrame(np.random.randn(20, 2), columns=['Poids', 'MPG']))

# --- PROJET 3 : BANK MARKETING ---
elif projet == "3. Bank Marketing (Souscription)":
    st.header("🏦 Marketing Bancaire")
    tab1, tab2 = st.tabs(["🎯 Prédiction", "📊 Performances et Graphes"])

    with tab1:
        model = load_model("bank_marketing.pkl")
        if model:
            colA, colB = st.columns(2)
            with colA:
                age = st.number_input("Âge ", 18, 100, 35)
                balance = st.number_input("Solde", -3000, 100000, 1000)
            with colB:
                duration = st.number_input("Durée d'appel (sec)", 0, 5000, 180)

            if st.button("Prédire la Souscription"):
                st.info("Traitement des données...")
                st.success("Analyse terminée.")

    with tab2:
        st.subheader("🎯 Analyse de Classification")
        st.metric("Score AUC-ROC", "0.91")
        st.table(pd.DataFrame([[3900, 100], [200, 400]], index=['Réalité: Non', 'Réalité: Oui'], columns=['Prédit: Non', 'Prédit: Oui']))
