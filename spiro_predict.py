import streamlit as st
import pandas as pd
import urllib.request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


st.markdown("""
    <style>
    /* Align sidebar content to vertical center */
    [data-testid="stSidebar"] {
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100vh;
    }
    .sidebar-title {
        color: blue;
        text-align: center;
        margin: 10px 0;
        font-size: 24px;
    }
    .sidebar-header {
        color: indigo;
        text-align: center;
        margin: 10px 0;
        font-size: 20px;
    }
    .sidebar-subheader {
        color: turquoise;
        text-align: center;
        margin: 10px 0;
    }
    .gold-button {
        background-color: gold;
        color: black;
        border: none;
        padding: 10px 20px;
        margin: 5px;
        font-size: 16px;
        border-radius: 8px;
        cursor: pointer;
        transition: 0.3s;
    }
    .gold-button:hover {
        background-color: #ffd700;
    }
    .button-container {
        display: flex;
        justify-content: space-around;
        margin-top: 20px;
    }
    .center-text {
        text-align: center;
    }
    .blue-header {
        color: blue;
        text-align: center;
        font-size: 30px;
    }
    </style>
""", unsafe_allow_html=True)


# Sidebar content with updated colors and text alignment
st.sidebar.markdown("<h2 class='sidebar-title'>Ma page de Spirométrie</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 class='sidebar-header' style='color: white;'>Facilitez-vous la vie</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='sidebar-subheader'>Diagnostic des trouble performence == f1-score</p>", unsafe_allow_html=True)
# ---- Télécharger la base de données depuis github ----
@st.cache_data
def load_data():
    url = "https://github.com/guyyone/spiro/raw/main/df_spiro.csv"
    data_path = "df_spiro.csv"
    urllib.request.urlretrieve(url, data_path)
    df = pd.read_csv(data_path)
    return df


data = load_data()

# ---- Séparer les prédicteurs et la cible ----
x = data[['vems', 'cvf', 'vemscvf', 'aex']]
y = data['type_trouble']

# ---- Diviser en ensembles d'entraînement et de test ----
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ---- Normalisation des données ----
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ---- Entraîner le modèle RandomForest ----
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# ---- Évaluer le modèle ----
y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average='weighted')  # Calcul du score F1

# ---- Interface utilisateur ----
st.markdown(
    "<h1 style='text-align: center; color: blue; font-weight: bold;'>Application de Prédiction de Trouble Ventilatoire</h1>",
    unsafe_allow_html=True)
st.subheader("")
st.markdown("<h3 style='text-align: center; color: #5DE2E7; font-weight: bold;'>Sélectionnez une fonctionnalité :</h3>",
            unsafe_allow_html=True)
st.subheader("")

# ---- Créer des boutons pour la navigation ----
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("Accueil", key="home_button"):
        st.session_state.page = "Accueil"

with col2:
    if st.button("Type de Trouble", key="trouble_button"):
        st.session_state.page = "Type de Trouble"

with col3:
    if st.button("Calcul de l'IMC", key="imc_button"):
        st.session_state.page = "Calcul de l'IMC"

with col4:
    if st.button("Boxplot AEX vs Type de Trouble", key="boxplot_button"):
        st.session_state.page = "Boxplot AEX vs Type de Trouble"

with col5:
    if st.button("Matrice de Confusion", key="confusion_matrix_button"):
        st.session_state.page = "Matrice de Confusion"

# ---- Page d'Accueil ----
if 'page' not in st.session_state:
    st.session_state.page = "Accueil"

if st.session_state.page == "Accueil":
    st.subheader("")
    st.markdown(
        "<h3 style='text-align: center; color: gold;'>Trouvez votre type de trouble avec un f1-score de 0.84</h3>",
        unsafe_allow_html=True)
    st.markdown(
        "<h5 style='text-align: center;'>Cette application vous permet de prédire le type de trouble ventilatoire à partir de paramètres spirométriques.</h5>",
        unsafe_allow_html=True)
    st.subheader("")
    st.markdown(
        "<h5 style='text-align: center;'>Vous pouvez également calculer l'IMC et visualiser des boxplots.</h5>",
        unsafe_allow_html=True)

# ---- Page pour la prédiction du type de trouble ventilatoire ----
elif st.session_state.page == "Type de Trouble":
    st.subheader("")
    st.markdown(
        "<h4 style='text-align: center; color: #CBEC1B;'>Renseigner les valeurs de vems, de cvf, de vems/cvf et de l'aire</h4>",
        unsafe_allow_html=True)

    # Paramètres de saisie
    def user_input():
        vems = st.number_input('vems', min_value=0.0, max_value=5.0, value=2.5, step=0.1)
        cvf = st.number_input('cvf', min_value=0.0, max_value=6.0, value=3.0, step=0.1)
        vemscvf = st.number_input('vemscvf', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        aex = st.number_input('aex', min_value=0.0, max_value=30.0, value=15.0, step=0.1)
        inputs = {'vems': vems, 'cvf': cvf, 'vemscvf': vemscvf, 'aex': aex}
        return pd.DataFrame([inputs])

    df_input = user_input()

    # ---- Prédiction du type de trouble ----
    x_input = scaler.transform(df_input)
    prediction = model.predict(x_input)[0]

    # ---- Interprétation de la prédiction ----
    categories = {0: "Normal", 1: "Obstructif", 2: "Restrictif"}
    predicted_category = categories[prediction]

    # ---- Interprétation de la prédiction ----
    categories = {0: "Normal", 1: "Obstructif", 2: "Restrictif"}
    predicted_category = categories[prediction]

    st.subheader("")
    st.markdown("<h4 style='text-align: center; color: #50EA13;'>Résultat de la prédiction</h4>",
                unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; color: green; font-weight: bold;'>{predicted_category}</h2>",
                unsafe_allow_html=True)
    # Affichage centré du score F1 en couleur or
    st.markdown(f"<h3 style='text-align: center; color: gold;'>Score F1 du modèle : {f1:.2f}</h3>",
                unsafe_allow_html=True)

# ---- Page pour le calcul de l'IMC ----
elif st.session_state.page == "Calcul de l'IMC":
    # Titre centré pour la section Calcul de l'IMC
    st.markdown("<h3 style='text-align: center; color: gold'>Calcul de l'IMC</h3>", unsafe_allow_html=True)

    # Fonction pour calculer l'IMC
    def calculate_bmi(weight, height):
        bmi = weight / (height ** 2)
        return bmi

    # Saisie des valeurs pour le calcul de l'IMC
    weight = st.number_input("Poids (kg)", min_value=0.0, value=70.0, step=0.1)
    height = st.number_input("Taille (m)", min_value=0.0, value=1.75, step=0.01)

    # Calculer l'IMC
    if height > 0:
        bmi = calculate_bmi(weight, height)

        # Affichage centré pour "IMC :"
        st.markdown("<h5 style='text-align: center; color: gold;'>Votre IMC est de</h5>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center; color: #50EA13;'>{bmi:.2f}</h4>", unsafe_allow_html=True)

        # Interprétation de l'IMC et affichage centré pour "Catégorie IMC :"
        if bmi < 18.5:
            imc_category = "Insuffisance pondérale"
        elif 18.5 <= bmi < 24.9:
            imc_category = "Poids normal"
        elif 25 <= bmi < 29.9:
            imc_category = "Surpoids"
        else:
            imc_category = "Obésité"

        st.markdown("<h5 style='text-align: center; color: gold;'>Vous appartenez à la catégorie</h5>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center; color: #50EA13;'>{imc_category}</h4>", unsafe_allow_html=True)
    else:
        st.write("Veuillez entrer une taille valide.")


# ---- Page pour afficher le boxplot entre AEX et type_trouble ----
elif st.session_state.page == "Boxplot AEX vs Type de Trouble":
    # Titre centré et en vert pour la section Boxplot
    st.markdown("<h3 style='text-align: center; color: #1371E9;'>Boxplot entre AEX et Type de Trouble</h3>", unsafe_allow_html=True)

    # Création du boxplot avec une palette de couleurs personnalisée
    plt.figure(figsize=(10, 6))
    custom_palette = ['#98F5F9', '#CECECE', '#EAAFFB']  # Palette de couleurs personnalisée
    sns.boxplot(x='type_trouble', y='aex', data=data, palette=custom_palette)
    plt.xlabel('Type de Trouble')
    plt.ylabel('AEX')
    plt.title('Boxplot de AEX selon le Type de Trouble')
    plt.xticks([0, 1, 2], ['Normal', 'Obstructif', 'Restrictif'])  # Remplacer les étiquettes
    st.pyplot(plt)


# ---- Page pour afficher la matrice de confusion ----
elif st.session_state.page == "Matrice de Confusion":
    # Titre centré et en vert pour la section Matrice de Confusion
    st.markdown("<h3 style='text-align: center; #FFECA1: green;'>Matrice de Confusion du Modèle</h3>", unsafe_allow_html=True)

    # Calculer la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    # Affichage de la matrice de confusion
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Obstructif", "Restrictif"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice de Confusion")
    st.pyplot(plt)
