import streamlit as st  
import pandas as pd
import joblib
import os
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Mes_3_modèles_en_marchine_learning", layout="wide", page_icon="logo64.png")

# --- CONFIGURATION DES CHEMINS ---
# Dossier contenant les modèles
MODEL_DIR = "models"
# Dossier contenant les données (pour rappel ou affichage optionnel)
DATA_DIR = "data"

# Fonction pour charger les modèles avec gestion d'erreur
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

# --- PAGE D'ACCUEIL ---

if projet == "Accueil":
    # Titre personnalisé affiché en VERT
    st.markdown(
        """
        <h1 style='color: #2ECC71; text-align: center; font-size: 32px; font-weight: bold;'>
            TEST_DE_NOS_3_MODELS DE MACHINE DONT LES DESCRIPTIONS SONT DONNEES CI-DESSOUS, SOYEZ LA BIENVENUE
        </h1>
        """, 
        unsafe_allow_html=True
    )
    
    st.write("---")
    
    st.write("---") # Une ligne de séparation élégante
    st.markdown("""
    Bienvenue dans cette interface de démonstration. Cette application regroupe trois modèles de Machine Learning 
    distincts, illustrant des cas d'usage concrets en entreprise : **Classification socio-économique**, 
    **Optimisation énergétique** et **Ciblage marketing**.
    """)
    
    st.divider()

    # --- DATASET 1 : CENSUS ---
    with st.expander("💰 Focus sur le Dataset : Census Income (Adult Dataset)", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://www.census.gov/content/dam/Census/public/brand/census-logo-white-on-blue.png", width=150)
        with col2:
            st.write("""
            **Contexte :** Issu de la base de données de l'UCI Machine Learning, ce dataset permet de prédire si le revenu d'un individu 
            dépasse les 50 000 $ par an en fonction de données démographiques.
            
            **Détails techniques :**
            - **Taille :** Environ 32 000 entrées.
            - **Variables cibles :** `>50K` ou `<=50K`.
            - **Features clés :** Le niveau d'éducation (Education-num), l'âge, la catégorie socioprofessionnelle et le gain en capital.
            - **Enjeu :** C'est un problème classique de classification binaire avec un fort déséquilibre de classes.
            """)

    # --- DATASET 2 : AUTO-MPG ---
    with st.expander("🚗 Focus sur le Dataset : Auto-MPG (Consommation de Carburant)", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("### ⛽ 📊")
        with col2:
            st.write("""
            **Contexte :** Ce dataset historique concerne la consommation de carburant des automobiles en miles par gallon (MPG). 
            L'objectif est de prédire l'efficacité énergétique d'un véhicule à partir de ses caractéristiques physiques.
            
            **Détails techniques :**
            - **Type de modèle :** Régression linéaire ou Random Forest Regressor.
            - **Variables clés :** Nombre de cylindres, poids du véhicule (très corrélé), puissance (horsepower) et année du modèle.
            - **Enjeu :** Comprendre l'impact de l'évolution technologique des années 70-80 sur la réduction de la consommation.
            """)

    # --- DATASET 3 : BANK MARKETING ---
    with st.expander("🏦 Focus sur le Dataset : Bank Marketing (Marketing Direct)", expanded=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("### 📞 🏦")
        with col2:
            st.write("""
            **Contexte :** Données liées à des campagnes de marketing direct d'une institution bancaire portugaise, basées sur des appels téléphoniques.
            
            **Détails techniques :**
            - **Objectif :** Prédire si le client va souscrire à un dépôt à terme (variable `y`).
            - **Variable Critique :** La **durée du contact** (plus elle est longue, plus la chance de succès est élevée).
            - **Variables contextuelles :** Le solde du compte (balance), l'existence de prêts (housing/loan) et les résultats des campagnes précédentes.
            - **Enjeu :** Optimiser les ressources de la banque en ciblant uniquement les clients à fort potentiel.
            """)

    st.divider()
    st.info("💡 Sélectionnez un projet dans le menu à gauche pour effectuer des prédictions en temps réel.")
# --- PROJET 1 : CENSUS ---
elif projet == "1. Census (Revenus)":
    st.header("📈 Prédiction des Tranches de Revenus (Census)")
    model = load_model("census.pkl")
    
    if model:
        st.subheader("Paramètres d'entrée")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Âge", 17, 90, 30)
            hours = st.slider("Heures travaillées par semaine", 1, 99, 40)
        with col2:
            edu_num = st.number_input("Années d'éducation", 1, 16, 10)
            capital_gain = st.number_input("Gain en capital", 0, 100000, 0)

        if st.button("Prédire le Revenu"):
            # 1. On crée le tableau de 85 colonnes avec les noms officiels
            input_data = pd.DataFrame(np.zeros((1, 85)), columns=model.feature_names_in_)
            
            # 2. On injecte TES valeurs dans les bonnes cases du modèle
            # On cherche les colonnes qui influencent vraiment le résultat
            if "TotalPop" in input_data.columns:
                input_data["TotalPop"] = age * 100  # On simule une population cohérente
            if "IncomePerCap" in input_data.columns:
                input_data["IncomePerCap"] = capital_gain if capital_gain > 500 else 25000
            if "Employed" in input_data.columns:
                input_data["Employed"] = hours * 50
            if "Professional" in input_data.columns:
                input_data["Professional"] = edu_num * 5 # Plus d'études = plus "Pro"
            
            # 3. Prédiction
            prediction = model.predict(input_data)
            
            # On affiche la valeur brute pour voir si ça bouge (0 ou 1)
            st.write(f" Valeur brute prédite : {prediction[0]}")
            
            label = ">50K$" if prediction[0] == 1 else "<=50K$"
            st.success(f"Résultat : **{label}**")


# --- PROJET 2 : AUTO-MPG ---
# --- PROJET 2 : AUTO-MPG ---
elif projet == "2. Auto-MPG (Consommation)":
    st.header("🚗 Estimation de la Consommation (Auto-MPG)")
    model = load_model("auto-mpg.pkl")
    scaler = load_model("scaler_mpg.pkl")
    
    if model and scaler:
        # --- CETTE PARTIE DOIT ÊTRE EN DEHORS DU BOUTON POUR ÊTRE VISIBLE ---
        st.subheader("Caractéristiques du véhicule")
        c1, c2, c3 = st.columns(3)
        with c1:
            cylinders = st.selectbox("Cylindres", [4, 6, 8])
            displacement = st.number_input("Cylindrée (Displacement)", 50.0, 500.0, 150.0)
        with c2:
            hp = st.number_input("Chevaux (Horsepower)", 40, 250, 100)
            weight = st.number_input("Poids (lbs)", 1500, 5000, 3000)
        with c3:
            accel = st.number_input("Accélération", 8.0, 25.0, 15.0)
            year = st.slider("Année du modèle (70-82)", 70, 82, 76)
        
        origin = st.radio("Origine", ["USA", "Europe", "Japon"], horizontal=True)
        origin_map = {"USA": 1, "Europe": 2, "Japon": 3}

        # --- SEUL LE CALCUL EST DANS LE BOUTON ---
        if st.button("Calculer MPG"):
            try:
                raw_data = np.array([[cylinders, displacement, hp, weight, accel, year, origin_map[origin]]])
                data_scaled = scaler.transform(raw_data)
                prediction = model.predict(data_scaled)
                st.success(f"Consommation estimée : **{prediction[0]:.2f} MPG**")
            except Exception as e:
                st.error(f"Erreur : {e}")
# --- PROJET 3 : BANK MARKETING ---
elif projet == "3. Bank Marketing (Souscription)":
    st.header("🏦 Marketing Bancaire (Bank-Full)")
    model = load_model("bank_marketing.pkl")
    
    if model:
        st.subheader("Profil du Client")
        colA, colB = st.columns(2)
        with colA:
            age = st.number_input("Âge du client", 18, 100, 35)
            balance = st.number_input("Solde du compte (Balance)", -3000, 100000, 1000)
            duration = st.number_input("Durée du dernier contact (sec)", 0, 5000, 180)
        with colB:
            housing = st.selectbox("Prêt immobilier ?", ["Oui", "Non"])
            loan = st.selectbox("Prêt personnel ?", ["Oui", "Non"])
            h_val = 1 if housing == "Oui" else 0
            l_val = 1 if loan == "Oui" else 0

        if st.button("Prédire la Souscription"):
            try:
                # 1. On crée le DataFrame avec les 16 noms officiels
                cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 
                        'housing', 'loan', 'contact', 'day', 'month', 'duration', 
                        'campaign', 'pdays', 'previous', 'poutcome']
                
                input_df = pd.DataFrame(np.zeros((1, 16)), columns=cols)
                
                # 2. On injecte les valeurs de l'interface
                input_df['age'] = age
                input_df['balance'] = balance
                input_df['housing'] = h_val
                input_df['loan'] = l_val
                input_df['duration'] = duration
                
                # 3. TEST DE SENSIBILITÉ (Le "poutcome" est souvent crucial)
                # On met une valeur positive sur 'pdays' ou 'previous' pour voir si ça réagit
                # input_df['previous'] = 1 
                
                # 4. Prédiction
                prediction = model.predict(input_df)
                proba = model.predict_proba(input_df)

                # 5. Affichage dynamique
                if prediction[0] == 1:
                    st.success(f"✅ Résultat : SOUSCRIPTION (Confiance : {proba[0][1]:.2%})")
                else:
                    st.error(f"❌ Résultat : PAS DE SOUSCRIPTION (Confiance : {proba[0][0]:.2%})")
                
                # Barre de progression pour voir si le curseur bouge
                st.write("Probabilité de succès :")
                st.progress(float(proba[0][1]))

            except Exception as e:
                st.error(f"Erreur technique : {e}")

# --- PIED DE PAGE DANS LA SIDEBAR ---
st.sidebar.markdown("<br>" * 8, unsafe_allow_html=True) # Pousse le texte vers le bas
st.sidebar.divider() # Ligne de séparation propre

st.sidebar.caption("© 2026 **EKOTTO ERIC ENS STUDENT**") # Ton nom en discret

# Liens GitHub et LinkedIn en bleu
st.sidebar.markdown(
    """
    <div style='display: flex; flex-direction: column; gap: 5px;'>
        <a href='https://github.com/ericekotto/tp-ml-deployment' target='_blank' style='text-decoration: none; color: #1E90FF; font-weight: bold; font-size: 14px;'>
            🔵 Mon lien Github vers mon projet
        </a>
    </div>
    <style>
        /* Supprime les bordures/boîtes grises indésirables autour des liens HTML dans la sidebar */
        [data-testid="stMarkdownContainer"] a {
            background-color: transparent !important;
        } 
    </style>
    """,
    unsafe_allow_html=True
)
