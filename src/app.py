import streamlit as st
import joblib

# Configuração
st.set_page_config(
    page_title="Veritas-IA Premium",
    page_icon="🧠",
    layout="wide"
)

# Carregar modelo
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Sidebar
with st.sidebar:
    st.title("📌 Sobre o Projeto")
    st.write("""
    **Veritas-IA** é um sistema inteligente capaz de analisar textos jornalísticos e identificar possíveis fake news utilizando:

    - Machine Learning
    - NLP
    - Classificação de Texto
    - Python
    """)

    st.write("---")
    st.caption("Projeto Integrador GTI")

# Título principal
st.title("🧠 Veritas-IA Premium")
st.subheader("Detector Inteligente de Notícias Falsas")

st.write("Cole uma notícia abaixo e clique em analisar.")

texto = st.text_area("", height=250)

if st.button("🚀 Analisar Agora"):

    if texto.strip() == "":
        st.warning("Digite um texto para análise.")
    else:
        texto_vec = vectorizer.transform([texto])

        resultado = model.predict(texto_vec)[0]
        prob = model.predict_proba(texto_vec).max()

        st.write("---")

        col1, col2 = st.columns(2)

        with col1:
            if resultado == 0:
                st.error("⚠️ Possível Fake News")
            else:
                st.success("✅ Notícia Confiável")

        with col2:
            st.metric("Confiança da IA", f"{prob:.2%}")

        st.progress(float(prob))

        st.write("### 🧾 Análise")

        if resultado == 0:
            st.info("""
            O sistema encontrou padrões frequentemente associados a notícias falsas:
            - Linguagem apelativa
            - Estrutura suspeita
            - Sensacionalismo textual
            """)
        else:
            st.info("""
            O sistema encontrou padrões compatíveis com notícias confiáveis:
            - Estrutura formal
            - Linguagem informativa
            - Texto consistente
            """)

st.write("---")
st.caption("Desenvolvido com Python • Streamlit • Scikit-learn")
