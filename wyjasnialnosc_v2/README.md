# Projekt 1 - model ratingowy z interpretowalnością, wyjaśnialnością i kalibracją do 4%

Dwa modele PD dla spółek z o.o. (`zbior_13.csv`): **karta scoringowa** (binning monotoniczny → WoE → regresja logistyczna)
i **XGBoost**, z wyjaśnieniami (SHAP, PDP / ICE / ALE, LIME, a z Lab 6: ważność grup skorelowanych cech, statystyka H Friedmana,
global surrogate), eksperymentami wierności i stabilności wyjaśnień, kalibracją
(Platt / izotoniczna / beta, potem dostosowanie do tendencji centralnej 4%), skalą ratingową, progiem z funkcji zysku i analizą kontrfaktyczną.

## Struktura

| Plik / folder | Zawartość |
|---|---|
| `projekt1_rating.ipynb` | cały pipeline: dane → EDA → cechy → walidacja → modele → wyjaśnialność → kalibracja 4% → progi / ratingi → kontrfakty → audyt i monitoring |
| `RAPORT.md` | raport techniczny z pełnymi wynikami |
| `MODEL_CARD.md` | karta modelu: cel, dane, ograniczenia, wyjaśnienia, kalibracja 4%, monitoring, użycie AI |
| `prezentacja_projekt1.pptx` | prezentacja dla osób nietechnicznych (12 slajdów) |
| `dane/` | `zbior_13.csv`, słownik zmiennych (PDF i CSV) |
| `wyniki/` | zapisany podział (`podzial.csv`), zestawy cech (`cechy_v1.json`, `cechy_karty.json`), parametry XGBoost, PD na teście, podsumowanie |
| `wykresy/` | wykresy użyte w raporcie i prezentacji (m.in. `surogat.png` - drzewo surogatu, `waterfall_graniczna_4proc.png` - SHAP modelu końcowego dla firmy granicznej przy progu operacyjnym) |

## Uruchomienie

Do uruchamiania projektu używaj Pythona w wersji zapisanej w pliku `python-version` dla reprodukowalności wyników.

```
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace projekt1_rating.ipynb     # albo Run All w Jupyterze
```

Notebook uruchamiamy z folderu projektu (ścieżki względne). Pełne uruchomienie trwa ok. 20-25 minut; najdłużej trwają statystyka H, drop-column (uczenie XGBoost od nowa dla każdej grupy cech),
SHAP w wariancie interventional, strojenie XGBoost i eksperymenty stabilności LIME.

## Replikowalność

- Ziarno `RNG_SEED = 42` dla podziału, CV, modeli i wyjaśnień.
- Podział zapisany w `wyniki/podzial.csv` (zbiór i grupa dla każdego wiersza danych).
- Zestawy cech: `wyniki/cechy_v1.json` (cechy po kontroli jakości), `wyniki/cechy_karty.json` (cechy karty). Notebook nadpisuje zapisywane pliki.
- Wersje bibliotek: `requirements.txt`.


## Użycie AI i kod z laboratoriów

Przy projekcie korzystano z asystenta AI (Claude, Anthropic) - szczegóły w `MODEL_CARD.md`. Funkcje przeniesione z laboratoriów
(Lab 1-6) są oznaczone w notebooku komentarzem `# z Lab N`.

## Git lokalnie

Projekt jest śledzony w lokalnym repozytorium Git; nie dodano zdalnego repozytorium.
Zmiany zapisuj poleceniami `git add <plik>` i `git commit -m "opis zmian"`.
