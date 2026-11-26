# Ogłoszenia i wycena nieruchomości

Aplikacja ma dwie strony: generator ogłoszeń ze zdjęcia i historyczną wycenę mieszkań w Poznaniu.

## Uruchomienie

W katalogu projektu wykonaj:

```bash
uv sync
uv run streamlit run app.py
```

Do generatora ogłoszeń oraz opcjonalnego wyjaśnienia wyceny przez AI potrzebny jest klucz OpenAI. Wpisz go w pliku `.env` w katalogu projektu:

```dotenv
OPENAI_API_KEY=twój_klucz
```

Wywołania OpenAI mogą być płatne. Generator wysyła zdjęcie do API. Zdjęcie możesz wgrać w aplikacji albo utworzyć folder `zdjecia/` i umieścić je w nim.

## Wycena mieszkań

Jeśli model nie jest jeszcze przygotowany, uruchom:

```bash
uv run python scripts/prepare_valuation.py
```

Skrypt domyślnie korzysta z danych w `../olx_price_modelling/data/real_estate_ads.csv`. Inną ścieżkę możesz podać przez `--data /ścieżka/do/pliku.csv`. Model jest zapisywany w `artifacts/` i nie wymaga klucza OpenAI. Wczytuj wyłącznie zaufane pliki modelu.

Wycena opiera się na danych z Poznania z lat 2018–2021. **Nie przedstawia aktualnej ceny rynkowej.** Na macOS LightGBM może wymagać `brew install libomp`.


