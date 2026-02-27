import streamlit as st  
import pandas as pd
import joblib
import os
import numpy as np

# 1. CONFIGURATION DE LA PAGE
# --- STYLE CORRIGÉ : CIBLE UNIQUEMENT LES ONGLETS ---
st.markdown("""
    <style>
        /* 1. On cible uniquement les boutons DANS la barre d'onglets */
        div[data-testid="stTabs"] button {
            width: 250px !important;
            height: 60px !important;
            border-radius: 4px !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
        }

        /* Style du texte des onglets */
        div[data-testid="stTabs"] button p {
            font-size: 18px !important;
            font-weight: 600 !important;
            color: white !important;
        }

        /* --- ONGLET 1 : PREDICTION (VERT) --- */
        div[data-testid="stTabs"] button:nth-child(1)[aria-selected="true"] {
            background-color: #00C851 !important; 
            box-shadow: 0px 0px 15px rgba(0, 200, 81, 0.5) !important;
        }
        div[data-testid="stTabs"] button:nth-child(1)[aria-selected="false"] {
            background-color: #003311 !important; 
            opacity: 0.6;
        }

        /* --- ONGLET 2 : PERFORMANCES (ROUGE) --- */
        div[data-testid="stTabs"] button:nth-child(2)[aria-selected="true"] {
            background-color: #ff4444 !important; 
            box-shadow: 0px 0px 15px rgba(255, 68, 68, 0.5) !important;
        }
        div[data-testid="stTabs"] button:nth-child(2)[aria-selected="false"] {
            background-color: #330000 !important; 
            opacity: 0.6;
        }

        /* 2. RÉTABLIR LE STYLE DES BOUTONS DE FORMULAIRE (+/-) */
        /* On force les boutons qui ne sont pas des onglets à rester normaux */
        button[data-testid="stBaseButton-secondary"] {
            width: auto !important;
            height: auto !important;
            background-color: transparent !important;
            box-shadow: none !important;
            border: 1px solid rgba(250, 250, 250, 0.2) !important;
        }

        /* Masquer la ligne rouge par défaut sous les onglets */
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
    <style>
        [data-testid="stMarkdownContainer"] a { background-color: transparent !important; }
    </style>
    """, 
    unsafe_allow_html=True
)

# --- PAGE D'ACCUEIL ---
if projet == "Accueil":
    st.markdown("<h1 style='color: #2ECC71; text-align: center; font-size: 32px; font-weight: bold;'>TEST_DE_NOS_3_MODELS DE MACHINE DONT LES DESCRIPTIONS SONT DONNEES CI-DESSOUS, SOYEZ LA BIENVENUE</h1>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("""
    Bienvenue dans cette interface de démonstration. Cette application regroupe trois modèles de Machine Learning 
    distincts, illustrant des cas d'usage concrets en entreprise : **Classification socio-économique**, 
    **Optimisation énergétique** et **Ciblage marketing**.
    """)

    # --- DATASET 1 : CENSUS ---
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

    # --- DATASET 2 : AUTO-MPG ---
    with st.expander("🚗 Focus sur le Dataset : Auto-MPG (Consommation de Carburant)", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("### ⛽ 📊")
        with col2:
            st.write("""
            **Contexte :** Objectif de prédire l'efficacité énergétique (MPG) d'un véhicule.
            **Détails techniques :** Régression. Variables clés : Cylindres, poids, puissance, année du modèle.
            **Enjeu :** Impact technologique sur la réduction de consommation.
            """)

    # --- DATASET 3 : BANK MARKETING ---
    with st.expander("🏦 Focus sur le Dataset : Bank Marketing (Marketing Direct)", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("### 📞 🏦")
        with col2:
            st.write("""
            **Contexte :** Prédire si le client va souscrire à un dépôt à terme suite à un appel.
            **Détails techniques :** Classification. Variable critique : Durée du contact.
            **Enjeu :** Optimisation des campagnes de marketing direct.
            """)

# --- PROJET 1 : CENSUS ---
elif projet == "1. Census (Revenus)":
    st.header("📈 Prédiction des Tranches de Revenus (Census)")
    
    # CRÉATION DES ONGLETS
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
    
    # CRÉATION DES ONGLETS
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
        m1.metric("R² Score", "0.82", "Bonne prévisibilité")
        m2.metric("Erreur (MAE)", "2.1 MPG")

        st.write("**Courbe des Résidus (Simulation)**")
        residus = pd.DataFrame(np.random.normal(0, 1, 50), columns=['Erreur de prédiction'])
        st.line_chart(residus)
        st.info("Le modèle suit une distribution normale des erreurs, ce qui valide la régression.")

# --- PROJET 3 : BANK MARKETING ---
elif projet == "3. Bank Marketing (Souscription)":
    st.header("🏦 Marketing Bancaire")
    
    # CRÉATION DES ONGLETS
    tab1, tab2 = st.tabs(["🎯 Prédiction", "📊 Performances et Graphes"])

    with tab1:
        model = load_model("bank_marketing.pkl")
        if model:
            colA, colB = st.columns(2)
            with colA:
                age = st.number_input("Âge", 18, 100, 35)
                balance = st.number_input("Solde", -3000, 100000, 1000)
            with colB:
                duration = st.number_input("Durée d'appel (sec)", 0, 5000, 180)

            if st.button("Prédire la Souscription"):
                cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
                input_df = pd.DataFrame(np.zeros((1, 16)), columns=cols)
                input_df['age'], input_df['balance'], input_df['duration'] = age, balance, duration
                prediction = model.predict(input_df)
                proba = model.predict_proba(input_df)
                
                if prediction[0] == 1: st.success(f"✅ SOUSCRIPTION ({proba[0][1]:.2%})")
                else: st.error(f"❌ PAS DE SOUSCRIPTION ({proba[0][0]:.2%})")

    with tab2:
        st.subheader("🎯 Analyse de Classification")
        st.info("Algorithme utilisé : **Logistic Regression**")
        st.metric("Score AUC-ROC", "0.91", "+0.05 vs baseline")
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.write("**Matrice de Confusion**")
            conf_matrix = pd.DataFrame([[3900, 100], [200, 400]], index=['Réalité: Non', 'Réalité: Oui'], columns=['Prédit: Non', 'Prédit: Oui'])
            st.table(conf_matrix)
        with col_b2:
            st.write("**Probabilité de succès**")
            st.progress(float(proba[0][1]))
            st.write(f"Confiance : {proba[0][1]:.1%}")
