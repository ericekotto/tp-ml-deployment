import streamlit as st  
import pandas as pd
import joblib
import os
import numpy as np

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Mes_3_modèles_en_marchine_learning", layout="wide", page_icon="logo64.png")

st.markdown("""
    <style>
        /* --- FORCE LA TRANSFORMATION DES RONDS EN RECTANGLES --- */
        
        /* 1. On cible le conteneur global du bouton radio dans la sidebar */
        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 10px;
        }

        /* 2. On transforme le label (le rectangle) */
        [data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] {
            background-color: #1E1E1E !important;
            padding: 12px 20px !important;
            border-radius: 5px !important; /* Rectangle parfait */
            border: 1px solid #444 !important;
            width: 100% !important;
            transition: 0.3s;
            display: flex;
        }

        /* 3. ON FAIT DISPARAITRE LE ROND (Le petit bouton rouge/blanc) */
        /* On cible le premier div enfant qui contient le cercle */
        [data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
            display: none !important;
        }

        /* 4. STYLE QUAND C'EST SÉLECTIONNÉ (LE RECTANGLE DEVIENT VERT) */
        [data-testid="stSidebar"] div[role="radiogroup"] label[aria-checked="true"] {
            background-color: #2ECC71 !important;
            border: 1px solid #FFFFFF !important;
            box-shadow: 0 0 10px rgba(46, 204, 113, 0.4);
        }

        /* 5. TEXTE À L'INTÉRIEUR DES RECTANGLES */
        [data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] p {
            font-weight: bold !important;
            color: white !important;
            margin: 0 !important;
            font-size: 15px !important;
        }

        /* --- STYLE DES ONGLETS (DU HAUT) --- */
        div[data-testid="stTabs"] button[role="tab"] {
            width: 250px !important;
            height: 60px !important;
            border-radius: 5px !important;
            border: 2px solid white !important;
            margin-right: 20px !important;
            font-weight: bold !important;
            color: white !important;
        }
        div[data-testid="stTabs"] button[role="tab"]:nth-child(1) { background-color: #00C851 !important; }
        div[data-testid="stTabs"] button[role="tab"]:nth-child(2) { background-color: #ff4444 !important; }

        /* --- STYLE BOUTON BLEU --- */
        div.stButton > button {
            background-color: #007BFF !important;
            color: white !important;
            width: 100% !important;
            height: 50px !important;
            border-radius: 8px !important;
        }

        /* Masquer la ligne orange */
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
# C'est ici que les ronds vont disparaître pour devenir des carrés
projet = st.sidebar.radio("Sélectionnez un projet :", 
    ["Accueil", "1. Census (Revenus)", "2. Auto-MPG (Consommation)", "3. Bank Marketing (Souscription)"])

# --- PIED DE PAGE SIDEBAR ---
st.sidebar.markdown("<br>" * 8, unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.caption("© 2026 **EKOTTO ERIC ENS STUDENT**")
st.sidebar.markdown(
    """
    <div style='display: flex; flex-direction: column; gap: 5px;'>
        <a href='https://github.com/ericekotto/tp-ml-deployment' target='_blank' style='text-decoration: none; color: #1E90FF; font-weight: bold;'>
            🔵 Mon lien Github vers mon projet
        </a>
    </div>
    """, 
    unsafe_allow_html=True
)

# --- PAGES ---
if projet == "Accueil":
    st.markdown("<h1 style='color: #2ECC71; text-align: center;'>TEST_DE_NOS_3_MODELS DE MACHINE LEARNING</h1>", unsafe_allow_html=True)
    st.divider()
    st.write("Bienvenue dans cette interface de démonstration.")
    
    with st.expander("💰 Census Income", expanded=True):
        st.write("Prédire si le revenu dépasse 50k$.")
    with st.expander("🚗 Auto-MPG", expanded=True):
        st.write("Prédire la consommation de carburant.")
    with st.expander("🏦 Bank Marketing", expanded=True):
        st.write("Prédire la souscription client.")

elif projet == "1. Census (Revenus)":
    st.header("📈 Prédiction Census")
    tab1, tab2 = st.tabs(["🎯 Prédiction", "📊 Performances"])
    with tab1:
        if st.button("Prédire le Revenu"):
            st.success("Résultat simulé")

elif projet == "2. Auto-MPG (Consommation)":
    st.header("🚗 Estimation Auto-MPG")
    tab1, tab2 = st.tabs(["🎯 Prédiction", "📊 Performances"])

elif projet == "3. Bank Marketing (Souscription)":
    st.header("🏦 Marketing Bancaire")
    tab1, tab2 = st.tabs(["🎯 Prédiction", "📊 Performances"])
