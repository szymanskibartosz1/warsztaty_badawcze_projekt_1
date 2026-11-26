from datetime import date
import json
from pathlib import Path

import joblib
from openai import OpenAI
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from utils import DEFAULT_MODELS, STRUCT_CAT, predict, resolve_api_key


MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "poznan_valuation.joblib"
CATEGORY_LABELS = {
    "building_type": "Typ budynku",
    "building_material": "Materiał budynku",
    "windows_type": "Rodzaj okien",
    "heating": "Ogrzewanie",
    "construction_status": "Stan mieszkania",
    "building_ownership": "Forma własności",
}
VALUE_LABELS = {
    "missing": "Nie podano", "other": "Inne / rzadkie",
    "block": "Blok", "apartment": "Apartamentowiec", "ribbon": "Zabudowa szeregowa",
    "apartment_building": "Apartamentowiec", "tenement": "Kamienica",
    "detached": "Wolnostojący", "brick": "Cegła", "concrete_slab": "Wielka płyta",
    "concrete_plate": "Wielka płyta", "concrete": "Beton", "breezeblock": "Pustak",
    "reinforced_concrete": "Żelbet", "silikat": "Silikat",
    "plastic": "Plastikowe", "wooden": "Drewniane", "urban": "Miejskie",
    "gas": "Gazowe", "electric": "Elektryczne", "ready_to_use": "Do zamieszkania",
    "to_renovation": "Do remontu", "to_completion": "Do wykończenia", "full_ownership": "Pełna własność",
    "co_operative_ownership": "Spółdzielcze własnościowe",
    "co_operative_ownership_with_a_land_and_mortgage_registe": "Spółdzielcze własnościowe z księgą wieczystą",
    "balcony": "balkon", "basement": "piwnica", "lift": "winda", "garage": "garaż",
    "terrace": "taras", "garden": "ogród", "internet": "internet", "parking": "parking",
    "entryphone": "domofon",
}
FLAG_GROUP_LABELS = {"media": "Media", "security": "Zabezpieczenia", "equipment": "Wyposażenie", "extras": "Dodatki"}


@st.cache_resource
def load_artifact(path, modified_ns):
    # Joblib files can execute code during loading. Only load a locally generated, trusted artifact.
    artifact = joblib.load(path)
    if artifact.get("format_version") != 1:
        raise ValueError("Nieobsługiwana wersja pliku modelu.")
    return artifact


def display_value(value):
    return VALUE_LABELS.get(value, value.replace("_", " "))


def display_flag(flag):
    group, value = flag.split("__", 1)
    return f"{FLAG_GROUP_LABELS.get(group, group)}: {display_value(value)}"


def display_feature_input(item):
    value = item["input"]
    if value is None:
        return "Nie podano"
    if item["feature"].endswith("__"):
        return "Brak wskazanych cech" if value == 1 else "Wskazano cechy"
    if "__" in item["feature"]:
        return "Wskazano" if value == 1 else "Nie wskazano"
    if isinstance(value, str):
        return display_value(value)
    return f"{value:.2f}".rstrip("0").rstrip(".")


def money(value):
    return f"{value:,.0f}".replace(",", " ") + " zł"


def get_api_key():
    key = resolve_api_key()
    if key:
        return key
    try:
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except StreamlitSecretNotFoundError:
        return ""


def explain_with_llm(result, model_id, api_key):
    payload = {
        "prognoza_zl_za_m2": round(result["price_per_m"], 2),
        "wartosc_bazowa_zl_za_m2": round(result["baseline"], 2),
        "wplywy": [
            {"cecha": item["label"], "wartosc_wejscia": item["input"], "wplyw_zl_za_m2": round(item["value"], 2)}
            for item in result["effects"]
        ],
    }
    response = OpenAI(api_key=api_key).chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": (
                    "Wyjaśniasz prognozę historycznego modelu cen mieszkań w Poznaniu. "
                    "Korzystaj wyłącznie z podanych wartości i lokalnych wkładów cech w zł/m². "
                    "Wskaż największe dodatnie i ujemne wkłady względem wartości bazowej. "
                    "Nie przedstawiaj ich jako przyczyn, aktualnych cen ani porad inwestycyjnych. "
                    "Wkłady cech są współzależne: metraż na pokój i odległość od centrum są cechami pochodnymi. "
                    "Wartość 0 dla cech wyposażenia oznacza brak zaznaczenia, nie dowód nieobecności. "
                    "Braki danych nie są potwierdzonymi cechami. TF-IDF jest zbiorczym sygnałem opisu, "
                    "nie ujawnia, które słowa wpłynęły na wynik. Nie wymyślaj danych. "
                    "Odpowiedz krótko po polsku, osobno opisując czynniki podwyższające, obniżające i ograniczenia."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        max_completion_tokens=1800,
    )
    text = response.choices[0].message.content
    if not text or not text.strip():
        raise ValueError("Model nie zwrócił wyjaśnienia.")
    return text.strip()


st.markdown(
    """
    <style>
    .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem; }
    .valuation-hero {
        padding: 2rem 2.25rem; margin-bottom: 1.5rem; border-radius: 1rem;
        background: linear-gradient(120deg, #123c44, #1a6b64); color: #fff;
    }
    .valuation-hero .eyebrow { color: #bbede0; font-size: .75rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase; }
    .valuation-hero h1 { color: #fff; font-size: 2.25rem; margin: .45rem 0; padding: 0; }
    .valuation-hero p { color: #e1f4ef; margin: 0; max-width: 640px; }
    @media (max-width: 640px) {
        .valuation-hero { padding: 1.5rem; }
        .valuation-hero h1 { font-size: 1.75rem; }
    }
    </style>
    <div class="valuation-hero">
        <div class="eyebrow">Historyczny model · Poznań</div>
        <h1>Wycena nieruchomości</h1>
        <p>Wprowadź dane mieszkania i zobacz prognozę wraz z wyjaśnieniem wpływu cech.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "Demonstracja na danych ofertowych rynku wtórnego w Poznaniu z lat 2018–2021. "
    "Wynik nie jest aktualną wyceną rynkową ani operatem szacunkowym."
)

if not MODEL_PATH.is_file():
    st.info("Model wyceny nie jest jeszcze przygotowany. W katalogu projektu uruchom:")
    st.code("uv sync\nuv run python scripts/prepare_valuation.py", language="bash")
    st.caption("Skrypt korzysta domyślnie z danych sąsiedniego projektu olx_price_modelling.")
    st.stop()

try:
    artifact = load_artifact(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
except Exception as error:
    st.error(f"Nie udało się wczytać modelu wyceny: {error}")
    st.caption("Sprawdź zależności i przygotuj model zgodnie z instrukcją w README.")
    st.stop()

min_date = date.fromisoformat(artifact["start_date"])
max_date = date.fromisoformat(artifact["end_date"])
defaults = {
    "m": 50.0, "no_rooms": 2, "map_lat": None, "map_lon": None,
    "district_lat": None, "district_lon": None, "floor": None,
    "building_floors_num": None, "build_year": None, "rent": None,
    "created_at": max_date, "description": "", "flags": [],
    **{column: "missing" for column in STRUCT_CAT},
}
for field, default in defaults.items():
    st.session_state.setdefault(f"valuation_{field}", default)

presets = {item["label"]: item["values"] for item in artifact["presets"]}


def apply_preset():
    selected = st.session_state["valuation_preset"]
    if selected not in presets:
        return
    values = {**defaults, **presets[selected]}
    values["created_at"] = date.fromisoformat(values["created_at"])
    values["flags"] = [flag for flag in values["flags"] if not flag.endswith("__")]
    for field, value in values.items():
        if field in {"no_rooms", "floor", "building_floors_num", "build_year"} and value is not None:
            value = int(value)
        st.session_state[f"valuation_{field}"] = value
    st.session_state.pop("valuation_result", None)
    st.session_state.pop("valuation_explanation", None)


st.subheader("Dane mieszkania")
st.selectbox(
    "Zacznij od przykładu", ["Własne dane", *presets],
    key="valuation_preset", on_change=apply_preset,
    help="Przykłady pochodzą ze zbioru testowego; dla ochrony danych nie zawierają oryginalnych opisów.",
)
st.caption("Przykłady odtwarzają parametry ze zbioru testowego, ale nie zawierają oryginalnych opisów.")

with st.form("valuation_form"):
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Podstawowe informacje")
        st.number_input("Powierzchnia (m²)", min_value=15.0, max_value=250.0, step=1.0, key="valuation_m")
        st.number_input("Liczba pokoi", min_value=1, max_value=8, step=1, key="valuation_no_rooms")
        st.date_input(
            "Data historycznej wyceny", min_value=min_date, max_value=max_date,
            key="valuation_created_at", help="Model korzysta z daty ogłoszenia, dlatego wybierz datę z okresu danych treningowych.",
        )
        st.text_area(
            "Opis mieszkania (opcjonalnie)", key="valuation_description", height=130, max_chars=10000,
            placeholder="Np. mieszkanie po remoncie, balkon, blisko komunikacji miejskiej",
            help="Opis wpływa na prognozę przez model TF-IDF. Nie wpisuj danych osobowych.",
        )
    with right:
        st.markdown("#### Lokalizacja w Poznaniu")
        st.caption("Wprowadź współrzędne punktu albo pozostaw oba pola puste, jeśli ich nie znasz.")
        st.number_input(
            "Szerokość geograficzna", min_value=-90.0, max_value=90.0, value=None,
            format="%.5f", step=0.0001, key="valuation_map_lat", placeholder="np. 52.40840",
        )
        st.number_input(
            "Długość geograficzna", min_value=-180.0, max_value=180.0, value=None,
            format="%.5f", step=0.0001, key="valuation_map_lon", placeholder="np. 16.93420",
        )
        st.markdown("#### Budynek")
        st.number_input("Piętro", min_value=0, max_value=11, value=None, step=1, key="valuation_floor", help="0 to parter; 11 oznacza 11. piętro lub wyższe.")
        st.number_input("Liczba pięter budynku", min_value=1, max_value=100, value=None, step=1, key="valuation_building_floors_num")
        st.number_input("Rok budowy", min_value=1850, max_value=2022, value=None, step=1, key="valuation_build_year")
        st.number_input("Czynsz miesięczny (zł)", min_value=1.0, max_value=3000.0, value=None, step=50.0, key="valuation_rent")

    with st.expander("Dodatkowe cechy i współrzędne dzielnicy"):
        category_columns = st.columns(2)
        for index, column in enumerate(STRUCT_CAT):
            options = ["missing"] + [value for value in artifact["categories"][column] if value != "missing"]
            with category_columns[index % 2]:
                st.selectbox(CATEGORY_LABELS[column], options, format_func=display_value, key=f"valuation_{column}")
        st.multiselect(
            "Znane cechy wyposażenia, mediów i zabezpieczeń",
            [flag for flag in artifact["flag_columns"] if not flag.endswith("__")],
            format_func=display_flag, key="valuation_flags",
            help="Niezaznaczone cechy są kodowane jako 0, tak jak w poprzednim projekcie; nie oznacza to, że ich brak został potwierdzony.",
        )
        district_left, district_right = st.columns(2)
        with district_left:
            st.number_input("Szerokość geograficzna dzielnicy", min_value=-90.0, max_value=90.0, value=None, format="%.5f", step=0.0001, key="valuation_district_lat")
        with district_right:
            st.number_input("Długość geograficzna dzielnicy", min_value=-180.0, max_value=180.0, value=None, format="%.5f", step=0.0001, key="valuation_district_lon")
        st.caption("Współrzędne dzielnicy są oddzielnymi cechami modelu. Jeśli ich nie znasz, pozostaw pola puste.")
    submitted = st.form_submit_button("Oblicz historyczną wycenę", type="primary", use_container_width=True)

if submitted:
    values = {field: st.session_state[f"valuation_{field}"] for field in defaults}
    pairs = [(values["map_lat"], values["map_lon"]), (values["district_lat"], values["district_lon"])]
    st.session_state.pop("valuation_result", None)
    st.session_state.pop("valuation_explanation", None)
    if any((lat is None) != (lon is None) for lat, lon in pairs):
        st.error("Podaj obie współrzędne dla lokalizacji lub dzielnicy albo pozostaw obie puste.")
    elif values["floor"] is not None and values["building_floors_num"] is not None and values["floor"] > values["building_floors_num"]:
        st.error("Piętro nie może być wyższe niż liczba pięter budynku.")
    elif any(lat is not None and not (52.2 <= lat <= 52.6 and 16.6 <= lon <= 17.2) for lat, lon in pairs):
        st.error("Wprowadź współrzędne położone w przybliżonym obszarze Poznania.")
    else:
        try:
            st.session_state["valuation_result"] = predict(artifact, values)
        except Exception as error:
            st.error(f"Nie udało się obliczyć wyceny: {error}")

result = st.session_state.get("valuation_result")
if result:
    st.divider()
    st.subheader("Wynik historycznej wyceny")
    result_left, result_right = st.columns(2)
    with result_left:
        st.metric("Prognozowana cena za m²", money(result["price_per_m"]))
    with result_right:
        st.metric("Szacunkowa cena mieszkania", money(result["total_price"]))
    st.caption(
        "Cena mieszkania = prognozowana cena za m² × powierzchnia. "
        "Wynik dotyczy ostatnio zatwierdzonego formularza i nie uwzględnia zmian cen po 2021 r."
    )
    st.markdown("#### Co wpłynęło na prognozę?")
    st.caption(
        f"Wartość bazowa modelu: {money(result['baseline'])}/m². "
        "Poniżej znajdują się największe lokalne wkłady względem tej wartości; nie dowodzą związku przyczynowego."
    )
    top_effects = result["effects"][:8]
    st.dataframe(
        [
            {"Cecha modelu": item["label"], "Wartość cechy": display_feature_input(item),
             "Wkład (zł/m²)": round(item["value"])}
            for item in top_effects
        ],
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "Cechy pochodne mogą być zależne od siebie. Niezaznaczone wyposażenie jest kodowane jako 0; "
        "sygnał TF-IDF nie wskazuje wpływu konkretnych słów."
    )

    st.markdown("#### Wyjaśnienie AI")
    st.caption(
        "Opcjonalnie wyślij wynik i wkłady cech do OpenAI, aby otrzymać opis słowny. "
        "Opis mieszkania nie jest wysyłany. Zapytanie może wiązać się z opłatą."
    )
    model_id = st.text_input("Model OpenAI do wyjaśnienia", value=DEFAULT_MODELS[0], key="valuation_explanation_model")
    if st.button("Wyjaśnij wynik z AI", disabled=not bool(model_id.strip())):
        st.session_state.pop("valuation_explanation", None)
        api_key = get_api_key()
        if not api_key:
            st.error("Ustaw OPENAI_API_KEY w pliku .env, środowisku lub sekretach Streamlit.")
        else:
            try:
                with st.spinner("Przygotowywanie wyjaśnienia..."):
                    st.session_state["valuation_explanation"] = explain_with_llm(result, model_id.strip(), api_key)
            except Exception as error:
                st.error(f"Nie udało się wygenerować wyjaśnienia: {error}")
    if st.session_state.get("valuation_explanation"):
        st.markdown(st.session_state["valuation_explanation"])
