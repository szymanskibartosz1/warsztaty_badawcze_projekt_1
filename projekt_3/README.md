# Projekt 3: generator ogłoszeń ze zdjęcia (Streamlit + OpenAI)

Aplikacja na bazie template'u [streamlit_image_processing](https://github.com/PawelOptimizer/streamlit_image_processing):
ze zdjęcia generuje tytuł, opis i parametry ogłoszenia w stylu OLX / Otodom / Otomoto, opcjonalnie z tytułem podanym
przez użytkownika. Raport z etapu 1: [RAPORT_ETAP1.md](RAPORT_ETAP1.md).

## Uruchomienie

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo 'OPENAI_API_KEY="twoj_klucz"' > .env
streamlit run app.py
```

## Pliki

- `app.py` - interfejs Streamlit,
- `utils.py` - prompty i wywołanie modelu (wynik w formacie JSON),
- `porownanie.py` - porównanie modeli i wpływu tytułu na zdjęciach z `przyklady/` (wyniki w `wyniki/`),
- `screeny/` - zrzuty ekranu aplikacji.
