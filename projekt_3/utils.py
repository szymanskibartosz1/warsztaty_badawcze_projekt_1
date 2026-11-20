import base64
import io
import json
import os
import time

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

MAX_IMAGE_SIDE = 1024

DEFAULT_PROMPT = """Jesteś doświadczonym sprzedawcą na OLX, Otodom i Otomoto. Na podstawie zdjęcia przygotuj ogłoszenie sprzedaży.

Zasady:
1. Rozpoznaj, co jest na zdjęciu (nieruchomość, samochód, motocykl, mebel, elektronika, inny przedmiot) i dobierz platformę: nieruchomości - Otodom, pojazdy - Otomoto, pozostałe - OLX.
2. Tytuł: do 70 znaków, konkretny, z najważniejszymi cechami (np. marka i model, typ, liczba pokoi), bez wykrzykników i CAPS LOCKA.
3. Opis: 4-6 zdań po polsku, napisanych jak przez sprzedającego (np. "Sprzedam...", "Na sprzedaż...", "Oferuję..."), gotowych do wklejenia na portal: co to jest, najważniejsze zalety, wyposażenie, dla kogo. Nigdy nie wspominaj o zdjęciu ani o tym, czego nie widać (bez "na zdjęciu", "widoczne", "nie da się ocenić", "do uzupełnienia", "podam później") - takie informacje trafiają wyłącznie do "do_uzupelnienia".
4. Parametry: takie jak w formularzu danej platformy. Nieruchomość: rodzaj, liczba pokoi, metraż, piętro, stan wykończenia, umeblowanie. Pojazd: marka, model, rocznik, typ nadwozia, kolor, stan. Przedmiot: kategoria, marka, materiał, kolor, stan.
5. Podawaj tylko to, co widać na zdjęciu lub wynika z podanego tytułu. Wartość, której nie da się ustalić, oznacz "do uzupełnienia"; szacunek oznacz "ok." (np. "ok. 2000-2003").
6. Nie podawaj ceny ani danych kontaktowych.

Zwróć wyłącznie JSON:
{"kategoria": "...", "platforma": "OLX | Otodom | Otomoto", "tytul": "...", "opis": "...", "parametry": {"nazwa": "wartość"}, "do_uzupelnienia": ["informacje, które sprzedający musi dopisać"]}"""

TITLE_PROMPT = """

Sprzedający podał tytuł ogłoszenia: "{title}"
Traktuj go jako informację od sprzedającego: wykorzystaj podane w nim fakty (marka, model, rocznik, metraż, liczba pokoi) w parametrach i opisie. Jeśli tytuł przeczy temu, co widać na zdjęciu, opieraj się na zdjęciu i dopisz rozbieżność do "do_uzupelnienia". Możesz poprawić tytuł stylistycznie, ale zachowaj jego fakty. Opis nadal pisz jako sprzedający: bez odwołań do tytułu, "informacji od sprzedającego" i bez uwag, co warto uzupełnić."""


def build_prompt(prompt, title=""):
    title = title.strip()
    return prompt + TITLE_PROMPT.format(title=title) if title else prompt


def encode_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_listing(client, image_bytes, prompt, model_choice):
    """Ask the model for a listing as JSON; returns the parsed listing with timing and token usage."""
    start = time.time()
    response = client.chat.completions.create(
        model=model_choice,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_bytes)}"}},
                ],
            }
        ],
        response_format={"type": "json_object"},
        # Reasoning models count hidden reasoning tokens toward this limit, so keep it generous
        max_completion_tokens=4000,
    )
    content = response.choices[0].message.content
    try:
        listing = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        listing = {"tytul": "", "opis": content or "", "parametry": {}}
    return {
        "listing": listing,
        "seconds": round(time.time() - start, 1),
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }


def resolve_api_key():
    return os.environ.get("OPENAI_API_KEY", "")
