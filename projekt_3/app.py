import streamlit as st
from openai import OpenAI
from streamlit.errors import StreamlitSecretNotFoundError

from utils import DEFAULT_PROMPT, build_prompt, generate_listing, resolve_api_key

MODELS = ["gpt-6-luna", "gpt-6.1-terra", "gpt-6.2-sol", "gpt-4.1-mini"]


def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default


st.set_page_config(page_title="Generator ogłoszeń", page_icon=":camera:")
st.title("Generator ogłoszeń ze zdjęcia")
st.write("Wgraj zdjęcie przedmiotu, mieszkania lub pojazdu, a model przygotuje tytuł, opis i parametry ogłoszenia w stylu OLX / Otodom / Otomoto.")

api_key = resolve_api_key() or get_secret("OPENAI_API_KEY", "")
if not api_key:
    st.warning("Brak klucza API: ustaw OPENAI_API_KEY w pliku .env lub w Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=api_key)
model = st.selectbox("Model", MODELS)

with st.expander("Prompt"):
    prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, height=380, label_visibility="collapsed")

title = st.text_input(
    "Tytuł ogłoszenia (opcjonalnie)",
    placeholder="np. Volkswagen Golf IV 1.4 benzyna 1998",
    help="Jeśli podasz tytuł, model wykorzysta zawarte w nim fakty w opisie i parametrach.",
)

uploaded_file = st.file_uploader("Wybierz zdjęcie", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Wgrane zdjęcie", width="stretch")

    if st.button("Generuj ogłoszenie", type="primary"):
        try:
            with st.spinner("Generuję ogłoszenie..."):
                final_prompt = build_prompt(prompt.strip() or DEFAULT_PROMPT, title)
                result = generate_listing(client, uploaded_file.getvalue(), final_prompt, model)
        except Exception as e:
            st.error(f"Błąd: {e}")
            st.stop()

        listing = result["listing"]
        st.caption(
            f"{listing.get('platforma', '')} · {listing.get('kategoria', '')} · model {model} · "
            f"{result['seconds']} s · tokeny: {result['input_tokens']} we / {result['output_tokens']} wy"
        )
        st.subheader(listing.get("tytul", ""))
        st.write(listing.get("opis", ""))
        if listing.get("parametry"):
            st.markdown("**Parametry**")
            st.table({"Parametr": list(listing["parametry"].keys()),
                      "Wartość": [str(v) for v in listing["parametry"].values()]})
        if listing.get("do_uzupelnienia"):
            st.markdown("**Do uzupełnienia przez sprzedającego**")
            st.markdown("\n".join(f"- {item}" for item in listing["do_uzupelnienia"]))
        with st.expander("Odpowiedź modelu (JSON)"):
            st.json(listing)
