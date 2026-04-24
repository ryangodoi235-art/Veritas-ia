import streamlit as st
import joblib

# Configuração da página
st.set_page_config(
    page_title="Veritas-IA",
    page_icon="📰",
    layout="centered"
)

# Carregar modelo
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Título
st.title("📰 Veritas-IA")
st.subheader("Sistema Inteligente de Detecção de Fake News")

st.markdown("---")

# Caixa de texto
texto = st.text_area(
    "Cole uma notícia abaixo para análise:",
    height=220
)

# Botão
if st.button("🔍 Analisar Notícia"):

    if texto.strip() == "":
        st.warning("Digite uma notícia primeiro.")
    else:
        texto_vec = vectorizer.transform([texto])

        resultado = model.predict(texto_vec)[0]
        prob = model.predict_proba(texto_vec).max()

        st.markdown("---")

        if resultado == 0:
            st.error(f"⚠️ Possível Fake News")
        else:
            st.success(f"✅ Notícia Provavelmente Verdadeira")

        st.progress(float(prob))

        st.info(f"Confiança da IA: {prob:.2%}")

        # Explicação automática
        if resultado == 0:
            st.write(
                "A IA identificou padrões textuais comuns em notícias falsas, como linguagem exagerada ou estrutura suspeita."
            )
        else:
            st.write(
                "A IA identificou padrões compatíveis com notícias confiáveis e estrutura textual consistente."
            )

st.markdown("---")
st.caption("Projeto acadêmico desenvolvido com Machine Learning e NLP.")
