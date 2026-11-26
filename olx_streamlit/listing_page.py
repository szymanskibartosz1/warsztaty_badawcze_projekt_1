import hashlib
from io import BytesIO
import json
import mimetypes
import os
import time
from datetime import date
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from openai import OpenAI
from streamlit.errors import StreamlitSecretNotFoundError

from utils import (
    CATEGORIES,
    DEFAULT_MODELS,
    FALLBACK_MODELS,
    SYSTEM_PROMPT,
    create_listing,
    resolve_api_key,
)


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_DIRECTORY = PROJECT_ROOT / "zdjecia"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024
MAX_PREVIEW_SIZE = (480, 360)
CUSTOM_MODEL_OPTION = "Własny identyfikator modelu"


def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default


def get_int_config(name, default):
    raw_value = get_secret(name, os.environ.get(name, default))
    try:
        value = int(raw_value)
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


MAX_REQUESTS_PER_MINUTE = get_int_config("MAX_REQUESTS_PER_MINUTE", 5)
MAX_REQUESTS_PER_DAY = get_int_config("MAX_REQUESTS_PER_DAY", 100)


def list_local_images():
    if not IMAGE_DIRECTORY.is_dir():
        return []
    return sorted(
        (
            path
            for path in IMAGE_DIRECTORY.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def make_image_preview(image_bytes):
    with Image.open(BytesIO(image_bytes)) as original_image:
        preview = ImageOps.exif_transpose(original_image)
        preview.thumbnail(MAX_PREVIEW_SIZE)
        return preview.copy()


def render_model_selector(label, default_model, key):
    options = [*FALLBACK_MODELS, CUSTOM_MODEL_OPTION]
    default_index = options.index(default_model) if default_model in options else 0
    selected_model = st.selectbox(label, options, index=default_index, key=f"{key}_choice")

    if selected_model == CUSTOM_MODEL_OPTION:
        return st.text_input(
            f"{label} — wpisz ID modelu",
            placeholder="np. identyfikator widoczny w OpenAI API",
            key=f"{key}_custom",
        ).strip()
    return selected_model


def make_run_signature(image_bytes, category, seller_title, title_mode, model_ids):
    signature_data = {
        "image": hashlib.sha256(image_bytes).hexdigest(),
        "category": category,
        "title": seller_title.strip(),
        "title_mode": title_mode,
        "models": model_ids,
    }
    encoded = json.dumps(signature_data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_usage_state():
    today = date.today().isoformat()
    if "request_timestamps" not in st.session_state:
        st.session_state["request_timestamps"] = []
    if "daily_usage" not in st.session_state or st.session_state["daily_usage"]["date"] != today:
        st.session_state["daily_usage"] = {"date": today, "count": 0}


def can_make_requests(request_count):
    prepare_usage_state()
    now = time.time()
    recent_timestamps = [
        timestamp
        for timestamp in st.session_state["request_timestamps"]
        if now - timestamp < 60
    ]
    st.session_state["request_timestamps"] = recent_timestamps

    if len(recent_timestamps) + request_count > MAX_REQUESTS_PER_MINUTE:
        st.error(
            f"Limit sesji: można wykonać maksymalnie {MAX_REQUESTS_PER_MINUTE} zapytań na minutę. "
            "Odczekaj chwilę lub zmniejsz liczbę porównywanych wariantów."
        )
        return False

    used_today = st.session_state["daily_usage"]["count"]
    if used_today + request_count > MAX_REQUESTS_PER_DAY:
        st.error(
            f"Limit sesji: można wykonać maksymalnie {MAX_REQUESTS_PER_DAY} zapytań dziennie."
        )
        return False
    return True


def register_request():
    now = time.time()
    st.session_state["request_timestamps"].append(now)
    st.session_state["daily_usage"]["count"] += 1


st.markdown(
    """
    <style>
    .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem; }
    .app-hero {
        padding: 2rem 2.25rem;
        margin-bottom: 1.5rem;
        border-radius: 1rem;
        background: linear-gradient(120deg, #123c44, #1a6b64);
        color: #ffffff;
    }
    .app-hero .eyebrow {
        color: #bbede0;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .app-hero h1 { color: #ffffff; font-size: 2.25rem; margin: 0.45rem 0; padding: 0; }
    .app-hero p { color: #e1f4ef; font-size: 1rem; margin: 0; max-width: 640px; }
    .step-label {
        color: #168074;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    @media (max-width: 640px) {
        .app-hero { padding: 1.5rem; }
        .app-hero h1 { font-size: 1.75rem; }
    }
    </style>
    <div class="app-hero">
        <div class="eyebrow">Twój asystent ogłoszeń</div>
        <h1>Od zdjęcia do ogłoszenia</h1>
        <p>Przygotuj tytuł, opis i specyfikację. Dodaj zdjęcie, uzupełnij szczegóły i porównaj propozycje modeli AI.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

api_key = resolve_api_key() or get_secret("OPENAI_API_KEY", "")

if not api_key:
    st.warning(
        "Brak klucza API. Ustaw OPENAI_API_KEY w pliku .env, "
        "zmiennej środowiskowej lub sekretach Streamlit, aby generować ogłoszenia."
    )

image_bytes = None
image_name = None
image_mime_type = None

image_column, details_column = st.columns([1, 1], gap="large")

with image_column:
    st.markdown('<div class="step-label">Krok 01</div>', unsafe_allow_html=True)
    st.subheader("Dodaj zdjęcie")
    st.caption("Wybierz zdjęcie, na którym dobrze widać przedmiot lub nieruchomość.")

    image_sources = ["Wgraj z komputera"]
    local_images = list_local_images()
    if local_images:
        image_sources.append("Wybierz z folderu zdjecia/")
    image_source = st.radio("Źródło zdjęcia", image_sources, horizontal=True)

    if image_source == "Wybierz z folderu zdjecia/":
        selected_image = st.selectbox(
            "Zdjęcie z folderu zdjecia/",
            local_images,
            format_func=lambda path: path.name,
        )
        try:
            image_bytes = selected_image.read_bytes()
            image_name = selected_image.name
            image_mime_type = mimetypes.guess_type(selected_image.name)[0] or "image/png"
        except OSError as error:
            st.error(f"Nie udało się odczytać zdjęcia: {error}")
    else:
        uploaded_file = st.file_uploader(
            "Zdjęcie przedmiotu lub nieruchomości",
            type=["jpg", "jpeg", "png", "webp"],
            help="Obsługiwane formaty: JPG, PNG i WEBP; maksymalny rozmiar: 15 MB.",
        )
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            image_name = uploaded_file.name
            image_mime_type = (
                uploaded_file.type
                or mimetypes.guess_type(uploaded_file.name)[0]
                or "image/png"
            )

    st.caption("JPG, PNG lub WEBP · maksymalnie 15 MB")
    if image_bytes is not None:
        if not image_bytes:
            st.error("Wybrany plik jest pusty. Wybierz inne zdjęcie.")
            image_bytes = None
        elif len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            st.error("Zdjęcie przekracza limit 15 MB. Wybierz mniejszy plik.")
            image_bytes = None
        else:
            try:
                image_preview = make_image_preview(image_bytes)
                st.image(
                    image_preview,
                    caption=f"Podgląd: {image_name}",
                    width=image_preview.width,
                )
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                st.error("Nie udało się odczytać obrazu. Wybierz poprawny plik JPG, PNG lub WEBP.")
                image_bytes = None

with details_column:
    st.markdown('<div class="step-label">Krok 02</div>', unsafe_allow_html=True)
    st.subheader("Opisz ogłoszenie")
    st.caption("Kategoria pomaga modelowi dobrać przydatne parametry.")
    category = st.selectbox("Rodzaj ogłoszenia", list(CATEGORIES))
    seller_title = st.text_input(
        "Tytuł od sprzedającego (opcjonalnie)",
        placeholder="Np. Jasne mieszkanie 2-pokojowe na Mokotowie",
        help="Tytuł jest deklaracją sprzedającego, a nie informacją zweryfikowaną ze zdjęcia.",
    )

    if seller_title.strip():
        title_mode = st.radio(
            "Jak wykorzystać podany tytuł?",
            [
                "Porównaj: bez tytułu i z tytułem",
                "Użyj podanego tytułu",
                "Nie używaj tytułu",
            ],
        )
    else:
        title_mode = "Bez tytułu"
        st.caption("Dodaj tytuł, aby porównać wyniki z tytułem i bez niego.")

    with st.expander("Modele i ustawienia porównania", expanded=True):
        compare_models = st.checkbox("Porównaj dwa modele OpenAI", value=True)
        first_model = render_model_selector("Model 1", DEFAULT_MODELS[0], "model_1")
        model_ids = [first_model]
        if compare_models:
            second_model = render_model_selector("Model 2", DEFAULT_MODELS[1], "model_2")
            model_ids.append(second_model)

        if compare_models and first_model and second_model and first_model == second_model:
            st.warning("Wybierz dwa różne modele, aby porównać ich wyniki.")

    if title_mode == "Porównaj: bez tytułu i z tytułem":
        title_variants = [("Bez tytułu", None), ("Z podanym tytułem", seller_title.strip())]
    elif title_mode == "Użyj podanego tytułu":
        title_variants = [("Z podanym tytułem", seller_title.strip())]
    else:
        title_variants = [("Bez tytułu", None)]

    number_of_requests = len(model_ids) * len(title_variants)
    st.caption(
        f"Zapytania do API w tym przebiegu: {number_of_requests} · "
        f"Limit sesji: {MAX_REQUESTS_PER_MINUTE}/min i {MAX_REQUESTS_PER_DAY}/dzień."
    )
    generate_button = st.button(
        "Wygeneruj ogłoszenie",
        type="primary",
        disabled=not bool(image_bytes),
        use_container_width=True,
        help="Najpierw wybierz zdjęcie." if not image_bytes else None,
    )
    st.caption(
        "Zdjęcie i dane wejściowe zostaną wysłane do OpenAI. "
        "Korzystanie z API może wiązać się z opłatami."
    )

st.divider()
st.markdown('<div class="step-label">Krok 03</div>', unsafe_allow_html=True)
st.subheader("Twoje propozycje")
st.caption("Porównaj wyniki i sprawdź ich poprawność przed publikacją.")

if generate_button:
    if not api_key:
        st.error("Brak klucza API. Ustaw OPENAI_API_KEY w pliku .env, środowisku lub sekretach Streamlit.")
    elif any(not model_id for model_id in model_ids):
        st.error("Wpisz identyfikator dla każdego wybranego modelu.")
    elif compare_models and len(set(model_ids)) != len(model_ids):
        st.error("Wybierz dwa różne modele, aby porównać ich wyniki.")
    elif not can_make_requests(number_of_requests):
        pass
    else:
        client = OpenAI(api_key=api_key)
        results = []
        progress = st.progress(0, text="Przygotowywanie zapytań...")
        for request_number, (model_id, (variant_label, title_for_prompt)) in enumerate(
            ((model_id, variant) for model_id in model_ids for variant in title_variants),
            start=1,
        ):
            progress.progress(
                (request_number - 1) / number_of_requests,
                text=f"Generowanie: {model_id} — {variant_label} ({request_number}/{number_of_requests})",
            )
            register_request()
            try:
                with st.spinner(f"Model {model_id}: {variant_label.lower()}..."):
                    listing_text = create_listing(
                        client=client,
                        image_bytes=image_bytes,
                        mime_type=image_mime_type or "image/png",
                        category=category,
                        seller_title=title_for_prompt,
                        model_id=model_id,
                    )
                results.append(
                    {
                        "model": model_id,
                        "variant": variant_label,
                        "text": listing_text,
                        "error": None,
                    }
                )
            except Exception as error:  # Keep successful results if another model call fails.
                results.append(
                    {
                        "model": model_id,
                        "variant": variant_label,
                        "text": None,
                        "error": str(error),
                    }
                )
        progress.progress(1.0, text="Gotowe.")
        run_signature = make_run_signature(
            image_bytes,
            category,
            seller_title,
            title_mode,
            model_ids,
        )
        st.session_state["listing_run"] = {
            "signature": run_signature,
            "image_name": image_name,
            "results": results,
        }

current_signature = (
    make_run_signature(image_bytes, category, seller_title, title_mode, model_ids)
    if image_bytes
    else None
)
saved_run = st.session_state.get("listing_run")
if saved_run and saved_run["signature"] == current_signature:
    st.info(
        "Wygenerowane treści są propozycją. Sprawdź w szczególności markę, "
        "parametry i stan przedmiotu — zdjęcie nie potwierdza wszystkich szczegółów."
    )
    if len(title_variants) == 2:
        st.caption("Porównując warianty, zwróć uwagę na zmiany w opisie i parametrach.")
    results = saved_run["results"]
    for row_start in range(0, len(results), 2):
        row_results = results[row_start:row_start + 2]
        result_columns = st.columns(len(row_results))
        for column, (index, result) in zip(
            result_columns, enumerate(row_results, start=row_start)
        ):
            with column:
                with st.container(border=True):
                    st.subheader(result["model"])
                    st.caption(result["variant"])
                    if result["error"]:
                        st.error(f"Nie udało się wygenerować wyniku: {result['error']}")
                    else:
                        st.markdown(result["text"])
                        download_text = (
                            f"Model: {result['model']}\n"
                            f"Wariant: {result['variant']}\n\n"
                            f"{result['text']}\n"
                        )
                        st.download_button(
                            "Pobierz wynik (.txt)",
                            data=download_text,
                            file_name=f"ogloszenie_{index + 1}.txt",
                            mime="text/plain",
                            key=f"download_result_{index}",
                            use_container_width=True,
                        )
elif saved_run:
    st.info("Ustawienia lub zdjęcie uległy zmianie. Wygeneruj nowe propozycje, aby zobaczyć wyniki.")
else:
    st.info("Tutaj pojawią się propozycje po wybraniu zdjęcia i wygenerowaniu ogłoszenia.")

with st.expander("Zobacz prompt systemowy"):
    st.code(SYSTEM_PROMPT, language="text")
