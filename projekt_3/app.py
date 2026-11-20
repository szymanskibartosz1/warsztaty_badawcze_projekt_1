import streamlit as st
import os
import time
from datetime import date
from streamlit.errors import StreamlitSecretNotFoundError
from openai import OpenAI
from utils import DEFAULT_PROMPT, build_prompt, generate_listing, resolve_api_key

FALLBACK_MODELS = ["gpt-6-luna", "gpt-6.1-terra", "gpt-6.2-sol", "gpt-4.1-mini"]
DEFAULT_MODEL = "gpt-6-luna"

def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default

DISABLE_MANUAL_API_KEY_INPUT = (
    str(get_secret("DISABLE_MANUAL_API_KEY_INPUT", "")).lower() in {"1", "true", "yes"}
)

def get_int_config(name, default):
    raw_value = get_secret(name, os.environ.get(name, default))
    try:
        value = int(raw_value)
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default

MAX_REQUESTS_PER_MINUTE = get_int_config("MAX_REQUESTS_PER_MINUTE", 5)
MAX_REQUESTS_PER_DAY = get_int_config("MAX_REQUESTS_PER_DAY", 100)

def fetch_available_gpt_models(client):
    models = client.models.list()
    available_model_ids = {model.id for model in models.data}
    allowed_models = [model for model in FALLBACK_MODELS if model in available_model_ids]
    return allowed_models

# Streamlit app layout
st.set_page_config(page_title="Generator ogłoszeń", page_icon=":camera:")
st.title("Generator ogłoszeń ze zdjęcia")
st.write("Wgraj zdjęcie przedmiotu, mieszkania lub pojazdu, a model przygotuje tytuł, opis i parametry ogłoszenia w stylu OLX / Otodom / Otomoto.")
st.caption(
    f"Limity: {MAX_REQUESTS_PER_MINUTE} zapytań na minutę i {MAX_REQUESTS_PER_DAY} dziennie na sesję."
)

api_key = resolve_api_key()
if not api_key:
    api_key = get_secret("OPENAI_API_KEY", "")

if not DISABLE_MANUAL_API_KEY_INPUT:
    manual_api_key = st.text_input(
        "Klucz OpenAI API (opcjonalny, jeśli ustawiono OPENAI_API_KEY)",
        type="password",
        help="Jeśli puste, aplikacja użyje OPENAI_API_KEY ze zmiennych środowiskowych lub Streamlit secrets.",
    )
    if manual_api_key:
        api_key = manual_api_key

if api_key:
    client = OpenAI(api_key=api_key)
    try:
        available_models = fetch_available_gpt_models(client)
    except Exception as e:
        available_models = FALLBACK_MODELS
        st.warning(f"Could not fetch models from API. Using fallback list. Details: {e}")

    if not available_models:
        available_models = FALLBACK_MODELS
        st.warning("No GPT models were returned by the API. Using fallback model list.")

    default_model = DEFAULT_MODEL if DEFAULT_MODEL in available_models else available_models[0]
    default_index = available_models.index(default_model)
    model_choice = st.selectbox("Model", available_models, index=default_index)
    custom_model = st.text_input(
        "Inny model (opcjonalnie)",
        value="",
        help="Jeśli podasz, zastąpi model wybrany powyżej.",
    )

    with st.expander("Prompt"):
        prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, height=380, label_visibility="collapsed")

    title = st.text_input(
        "Tytuł ogłoszenia (opcjonalnie)",
        placeholder="np. Volkswagen Golf IV 1.4 benzyna 1998",
        help="Jeśli podasz tytuł, model wykorzysta zawarte w nim fakty w opisie i parametrach.",
    )

    uploaded_file = st.file_uploader("Wybierz zdjęcie", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        try:
            st.image(uploaded_file, caption="Wgrane zdjęcie", width="stretch")

            if st.button("Generuj ogłoszenie", type="primary"):
                if "request_timestamps" not in st.session_state:
                    st.session_state["request_timestamps"] = []
                if "daily_usage" not in st.session_state:
                    st.session_state["daily_usage"] = {"date": date.today().isoformat(), "count": 0}

                now = time.time()
                recent_timestamps = [
                    ts for ts in st.session_state["request_timestamps"] if now - ts < 60
                ]
                st.session_state["request_timestamps"] = recent_timestamps

                if len(recent_timestamps) >= MAX_REQUESTS_PER_MINUTE:
                    st.error("Rate limit reached: too many requests in the last minute. Please wait.")
                    st.stop()

                today = date.today().isoformat()
                if st.session_state["daily_usage"]["date"] != today:
                    st.session_state["daily_usage"] = {"date": today, "count": 0}

                if st.session_state["daily_usage"]["count"] >= MAX_REQUESTS_PER_DAY:
                    st.error("Daily limit reached for this session. Please try again tomorrow.")
                    st.stop()

                st.session_state["request_timestamps"].append(now)
                st.session_state["daily_usage"]["count"] += 1

                with st.spinner("Generuję ogłoszenie..."):
                    final_prompt = build_prompt(prompt.strip() or DEFAULT_PROMPT, title)
                    final_model = custom_model.strip() or model_choice
                    result = generate_listing(client, uploaded_file.getvalue(), final_prompt, final_model)

                listing = result["listing"]
                st.caption(
                    f"{listing.get('platforma', '')} · {listing.get('kategoria', '')} · model {final_model} · "
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
        except Exception as e:
            st.error(f"Error: {e}")
else:
    if DISABLE_MANUAL_API_KEY_INPUT:
        st.warning("No API key configured. Set OPENAI_API_KEY in environment variables or Streamlit secrets.")
    else:
        st.warning("Please provide an API key in the field above or set OPENAI_API_KEY locally.")
