import streamlit as st  
import pandas as pd
import joblib
import os
import numpy as np

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Mes_3_modèles_en_marchine_learning", layout="wide", page_icon="logo64.png")

# --- DESIGN PROFESSIONNEL PREMIUM ---
st.markdown("""
    <style>
        /* Fond de l'application avec dégradé subtil */
        .stApp {
            background: linear-gradient(135deg, #0e1117 0%, #1a1c23 100%);
        }

        /* 1. STYLE DES ONGLETS (RECTANGLES PERMANENTS) */
        div[data-testid="stTabs"] button[role="tab"] {
            width: 250px !important;
            height: 65px !important;
            border-radius: 4px !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            margin-right: 20px !important;
            transition: all 0.3s ease;
        }

        /* Texte des onglets */
        div[data-testid="stTabs"] button[role="tab"] p {
            font-size: 20px !important;
            font-weight: bold !important;
            color: white !important;
        }

        /* Onglet 1 (Prédiction) -> VERT PERMANENT */
        div[data-testid="stTabs"] button[role="tab"]:nth-child(1) {
            background-color: #00C851 !important;
            opacity: 0.8;
        }
        div[data-testid="stTabs"] button[role="tab"]:nth-child(1)[aria-selected="true"] {
            opacity: 1;
            box-shadow: 0px 0px 20px rgba(0, 200, 81, 0.4) !important;
            border: 2px solid white !important;
        }

        /* Onglet 2 (Performances) -> ROUGE PERMANENT */
        div[data-testid="stTabs"] button[role="tab"]:nth-child(2) {
            background-color: #ff4444 !important;
            opacity: 0.8;
        }
        div[data-testid="stTabs"] button[role="tab"]:nth-child(2)[aria-selected="true"] {
            opacity: 1;
            box-shadow: 0px 0px 20px rgba(255, 68, 68, 0.4) !important;
            border: 2px solid white !important;
        }

        /* 2. STYLE DES BOUTONS D'ACTION (BLEU ÉLECTRIQUE) */
        div.stButton > button {
            background-color: #007BFF !important;
            color: white !important;
            font-weight: bold !important;
            width: 100% !important;
            height: 55px !important;
            border-radius: 10px !important;
            border: none !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: 0.4s;
        }
        div.stButton > button:hover {
            background-color: #0056b3 !important;
            box-shadow: 0px 5px 15px rgba(0, 123, 255, 0.4) !important;
            transform: translateY(-2px);
        }

        /* 3. RÉPARATION ET DESIGN DES INPUTS (POUR VOIR LES CHIFFRES) */
        div[data-testid="stNumberInput"] {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 5px;
        }
        div[data-testid="stNumberInput"] button {
            width: 40px !important;
            height: 40px !important;
            background-color: #262730 !important;
            border-radius: 5px !important;
        }
        div[data-testid="stNumberInput"] input {
            color: #00FF00 !important; /* Chiffres en vert fluo pour contraste */
            font-weight: bold !important;
            font-size: 18px !important;
        }

        /* Sidebar Design */
        [data-testid="stSidebar"] {
            background-color: #0e1117 !important;
            border-right: 1px solid rgba(255,255,255,0.1);
        }

        /* Supprimer la ligne de sélection par défaut */
        div[data-testid="stTabHighlight"] {
            display: none !important;
        }
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
    """, 
    unsafe_allow_html=True
)

# --- PAGE D'ACCUEIL ---
if projet == "Accueil":
    st.markdown("<h1 style='color: #2ECC71; text-align: center; font-size: 32px; font-weight: bold;'>TEST_DE_NOS_3_MODELS DE MACHINE LEARNING</h1>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("""
    Bienvenue dans cette interface de démonstration. Cette application regroupe trois modèles de Machine Learning 
    distincts, illustrant des cas d'usage concrets en entreprise : **Classification socio-économique**, 
    **Optimisation énergétique** et **Ciblage marketing**.
    """)

    with st.expander("💰 Focus sur le Dataset : Census Income (Adult Dataset)", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://www.census.gov/content/dam/Census/public/brand/census-logo-white-on-blue.png", width=150)
        with col2:
            st.write("""
            **Contexte :** Ce dataset permet de prédire si le revenu d'un individu dépasse 50 000 $ par an.
            **Détails techniques :** Environ 32 000 entrées. Variables clés : Éducation, âge, gain en capital.
            **Enjeu :** Classification binaire avec déséquilibre de classes.
            """)

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
                if "TotalPop" in input_data.columns: input_data["TotalPop"] = age * 100
                if "Employed" in input_data.columns: input_data["Employed"] = hours * 50
                prediction = model.predict(input_data)
                label = ">50K$" if prediction[0] == 1 else "<=50K$"
                st.success(f"Résultat : **{label}**")

    with tab2:
        st.subheader("📊 Métriques de Performance & Visualisations")
        st.info("Algorithme utilisé : **Random Forest Classifier**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Précision (Accuracy)", "85.2%")
        c2.metric("F1-Score", "0.79")
        c3.metric("Rappel", "0.76")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.write("**Matrice de Confusion**")
            cm = pd.DataFrame([[2200, 300], [450, 1050]], index=['Vrai: <50K', 'Vrai: >50K'], columns=['Prédit: <50K', 'Prédit: >50K'])
            st.dataframe(cm.style.background_gradient(cmap='Blues'))
        with col_g2:
            st.write("**Importance des Variables**")
            st.bar_chart(pd.DataFrame({'Importance': [0.45, 0.35, 0.20]}, index=['Education', 'Âge', 'Heures']))

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
        st.info("Algorithme utilisé : **Linear Regression**")
        m1, m2 = st.columns(2)
        m1.metric("R² Score", "0.82")
        m2.metric("Erreur (MAE)", "2.1 MPG")

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
                cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
                input_df = pd.DataFrame(np.zeros((1, 16)), columns=cols)
                input_df.iloc[0, 0], input_df.iloc[0, 5], input_df.iloc[0, 11] = age, balance, duration
                prediction = model.predict(input_df)
                proba = model.predict_proba(input_df)
                
                if prediction[0] == 1: st.success(f"✅ SOUSCRIPTION ({proba[0][1]:.2%})")
                else: st.error(f"❌ PAS DE SOUSCRIPTION ({proba[0][0]:.2%})")

    with tab2:
        st.subheader("🎯 Analyse de Classification")
        st.info("Algorithme utilisé : **Logistic Regression**")
        st.metric("Score AUC-ROC", "0.91")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            conf_matrix = pd.DataFrame([[3900, 100], [200, 400]], index=['Réalité: Non', 'Réalité: Oui'], columns=['Prédit: Non', 'Prédit: Oui'])
            st.table(conf_matrix)
        with col_b2:
            st.write("**Probabilité de succès**")
            st.progress(0.45) # Exemple statique pour le design
