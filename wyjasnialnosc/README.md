# Projekt 1 - model ratingowy z interpretowalnością, wyjaśnialnością i kalibracją do 4%

Dwa modele PD dla spółek z o.o. (`zbior_13.csv`): **karta scoringowa** (binning monotoniczny → WoE → regresja logistyczna)
i **XGBoost**, z wyjaśnieniami (SHAP, PDP / ICE / ALE, LIME), eksperymentami wierności i stabilności wyjaśnień, kalibracją
(Platt / izotoniczna / beta, potem dostosowanie do tendencji centralnej 4%), skalą ratingową, progiem z funkcji zysku i analizą kontrfaktyczną.

## Struktura

| Plik / folder | Zawartość |
|---|---|
| `projekt1_rating.ipynb` | cały pipeline: dane → EDA → cechy → walidacja → modele → wyjaśnialność → kalibracja 4% → progi / ratingi → kontrfakty → audyt i monitoring |
| `RAPORT.md` | raport techniczny z pełnymi wynikami |
| `MODEL_CARD.md` | karta modelu: cel, dane, ograniczenia, wyjaśnienia, kalibracja 4%, monitoring, użycie AI |
| `prezentacja_projekt1.pptx` | prezentacja dla osób nietechnicznych (12 slajdów) |
| `dane/` | `zbior_13.csv`, słownik zmiennych (PDF i CSV) |
| `wyniki/` | zapisany podział (`podzial.csv`), wersje cech (`cechy_v1.json`, `cechy_karty.json`), parametry XGBoost, PD na teście, podsumowanie |
| `wykresy/` | wykresy użyte w raporcie i prezentacji |

## Uruchomienie

```
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace projekt1_rating.ipynb     # albo Run All w Jupyterze
```

Notebook uruchamiamy z folderu projektu (ścieżki względne). Pełne uruchomienie trwa ok. 5 minut (Python 3.11, laptop
z 6-rdzeniowym CPU); najdłużej trwa strojenie XGBoost i eksperymenty stabilności LIME.

## Replikowalność

- Ziarno `RNG_SEED = 42` dla podziału, CV, modeli i wyjaśnień.
- Podział zapisany w `wyniki/podzial.csv` (zbiór i grupa dla każdego wiersza danych).
- Wersje cech: `wyniki/cechy_v1.json` (cechy po kontroli jakości), `wyniki/cechy_karty.json` (cechy karty).
- Wersje bibliotek: `requirements.txt`.

## Najważniejsze wyniki (test, 599 firm, 89 defaultów)

| | Karta scoringowa | XGBoost |
|---|---|---|
| Gini / KS / PR-AUC | 0,46 / 0,35 / 0,33 | 0,49 / 0,37 / 0,34 |
| Liczba cech | 7 | 164 używane z 192 |
| δ do średniego PD = 4% | −1,49 | −1,46 |

Przewaga XGBoost nie jest statystycznie istotna (różnica Gini +0,03, 95% CI [−0,04; +0,09]). Rekomendacja: karta scoringowa
jako model ratingowy, XGBoost jako model kontrolny.

## Użycie AI i kod z laboratoriów

Przy projekcie korzystano z asystenta AI (Claude, Anthropic) - szczegóły w `MODEL_CARD.md`. Funkcje przeniesione z laboratoriów
(Lab 1-5) są oznaczone w notebooku komentarzem `# z Lab N`.
