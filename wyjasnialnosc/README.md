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
| `wyniki/` | zapisany podział (`podzial.csv`), wersje cech (`cechy_v1.json`, `cechy_karty.json`), parametry XGBoost, PD na teście, podsumowanie |
| `wykresy/` | wykresy użyte w raporcie i prezentacji (m.in. `surogat.png` - drzewo surogatu, `waterfall_graniczna_4proc.png` - SHAP modelu końcowego dla firmy granicznej przy progu operacyjnym) |

## Uruchomienie

```
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace projekt1_rating.ipynb     # albo Run All w Jupyterze
```

Notebook uruchamiamy z folderu projektu (ścieżki względne). Pełne uruchomienie trwa ok. 20-25 minut (Python 3.11, laptop
z 6-rdzeniowym CPU); najdłużej trwają statystyka H, drop-column (uczenie XGBoost od nowa dla każdej grupy cech),
SHAP w wariancie interventional, strojenie XGBoost i eksperymenty stabilności LIME.

## Replikowalność

- Ziarno `RNG_SEED = 42` dla podziału, CV, modeli i wyjaśnień.
- Podział zapisany w `wyniki/podzial.csv` (zbiór i grupa dla każdego wiersza danych).
- Wersje cech: `wyniki/cechy_v1.json` (cechy po kontroli jakości), `wyniki/cechy_karty.json` (cechy karty).
- Wersje bibliotek: `requirements.txt`.
- Wyniki XGBoost zależą od platformy: losowanie wierszy i kolumn (`subsample`, `colsample_bytree`) korzysta z rozkładów losowych
  biblioteki standardowej C++, które różnią się między systemami operacyjnymi, więc to samo ziarno może dać na innym systemie nieco
  inny black-box. Raportowane liczby pochodzą z jednego uruchomienia.

## Najważniejsze wyniki (test, 599 firm, 89 defaultów)

| | Karta scoringowa | XGBoost |
|---|---|---|
| Gini / KS / PR-AUC | 0,562 / 0,440 / 0,430 | 0,583 / 0,457 / 0,458 |
| Liczba cech | 8 (w tym `wsk_index_nowe`) | 179 używanych z 195 |
| δ do średniego PD = 4% (valid ważony do DR 4%) | −1,280 | −1,218 |
| Próg z funkcji zysku (valid ważony do DR 4%; p* = 18,2%) | 15,0% | 20,5% |

Przewaga XGBoost jest mała i niepewna: różnica Gini na teście +0,021, 95% CI [−0,048; +0,089] (nieistotna), na valid +0,072
[+0,019; +0,120] (istotna). Najważniejsze predyktory to wskaźniki branżowe `wsk_index_*` (prowadzący potwierdził brak przecieku).
Rekomendacja, oparta na interpretowalności i stabilności wyjaśnień: karta scoringowa jako model ratingowy, XGBoost jako challenger.

## Użycie AI i kod z laboratoriów

Przy projekcie korzystano z asystenta AI (Claude, Anthropic) - szczegóły w `MODEL_CARD.md`. Funkcje przeniesione z laboratoriów
(Lab 1-6) są oznaczone w notebooku komentarzem `# z Lab N`.
