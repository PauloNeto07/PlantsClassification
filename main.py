import streamlit.components.v1 as components
import lime.lime_tabular
import streamlit
import mahotas
import pickle
import pandas
import numpy
import cv2

from sklearn.model_selection import train_test_split

def get_model():
    return pickle.load(open('etc/random_forest_model.dat', 'rb'))

def convert_byteio_image(string):
    array = numpy.frombuffer(string, numpy.uint8)
    image = cv2.imdecode(array, flags=1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

streamlit.markdown("<h1 style='text-align: center; color: white;'>Classificação de Plantas Frutíferas</h1>", unsafe_allow_html=True)

streamlit.sidebar.title('Opsans')

uploaded_images = streamlit.sidebar.file_uploader("Escolha imagem", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

model = get_model()
if uploaded_images:
    for uploaded_image in uploaded_images:
        bytes_data = uploaded_image.getvalue()
        image = convert_byteio_image(bytes_data)

        if (image.shape != (256, 256)):    
            image = cv2.resize(image, (256, 256))

        features = mahotas.features.haralick(image, compute_14th_feature=True, return_mean=True).reshape(1, 14)

        pred = model.predict(features)   
        probs = model.predict_proba(features)

        col1, col2 = streamlit.columns([5 , 5])

        with col1:
            streamlit.markdown("<h3 style='text-align: center ; color: white;'>Image</h3>", unsafe_allow_html=True)
            col1.image(image, use_column_width=True)

        if pred[0] == 1:
            pred_output = f"Probabilidades: Acerola: {probs[0][0]:.2%}"
        elif pred[0] == 2:
            pred_output = f"Probabilidades: Amora: {probs[0][1]:.2%}"
        elif pred[0] == 3:
            pred_output = f"Probabilidades: Bacuri: {probs[0][2]:.2%}"
        elif pred[0] == 4:
            pred_output = f"Probabilidades: Banana: {probs[0][3]:.2%}"
        elif pred[0] == 5:
            pred_output = f"Probabilidades: Caja: {probs[0][4]:.2%}"
        elif pred[0] == 6:
            pred_output = f"Probabilidades: Caju: {probs[0][5]:.2%}"
        elif pred[0] == 7:
            pred_output = f"Probabilidades: Goiaba: {probs[0][6]:.2%}"
        elif pred[0] == 8:
            pred_output = f"Probabilidades: Graviola: {probs[0][7]:.2%}"
        elif pred[0] == 9:
            pred_output = f"Probabilidades: Mamão: {probs[0][8]:.2%}"
        elif pred[0] == 10:
            pred_output = f"Probabilidades: Manga: {probs[0][9]:.2%}"
        elif pred[0] == 11:
            pred_output = f"Probabilidades: Maracuja: {probs[0][10]:.2%}"
        elif pred[0] == 12:
            pred_output = f"Probabilidades: Pinha: {probs[0][11]:.2%}"

        with col1:    

            streamlit.markdown("<h4 style='text-align: center; color: white;'>" + pred_output + "</h4>", unsafe_allow_html=True)

    if streamlit.sidebar.button("Explicar Predição"):

            streamlit.spinner('Analizando🤔 ...')        

            df = pandas.read_csv('etc/features_RF.csv', delimiter=';')

            X = df.drop('Label', axis=1)
            y = df['Label']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X.columns, class_names=['Acerola', 'Amora', 'Bacuri', 'Banana', 'Caja', 'Caju', 'Goiaba', 'Graviola', 'Mamão', 'Manga', 'Maracuja', 'Pinha'],
                                                           feature_selection='lasso_path', discretize_continuous=False)
        
            exp = explainer.explain_instance(features.reshape(14), model.predict_proba, num_features=14)

            with col2:

                streamlit.markdown("<h3 style='text-align: center; color: white;'>Interpretação</h3>", unsafe_allow_html=True)

                exp_html = exp.as_html(predict_proba = True)
                exp_html_with_styles = exp_html.replace('<head>', '<head><style>td {color: white;}</style>')
                components.html(exp_html_with_styles, height=800)