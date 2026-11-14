prezentacja: https://www.overleaf.com/project/6af5ffdb3ac8ca187bcd9607/share#de691eb326a2ee8adca59a2c3844adec26c836c22266dfcd

# Projekt 2 – predykcja ceny za m² (rynek wtórny, Poznań)

Cały kod jest w notebooku **`projekt2_olx.ipynb`** (uruchamiać od góry do dołu).
Notebook liczy wszystkie modele i zapisuje tabele wyników do `results/`.

## Uruchomienie
```
pip install -r requirements.txt
# rozpakować real_estate_ads.csv.zip do folderu data/
jupyter notebook projekt2_olx.ipynb
```
Kroki korzystające z OpenAI API (embeddingi OpenAI, ekstrakcja cech przez gpt-4.1-mini, wycena przez LLM)
zostały uruchomione raz, a odpowiedzi są w `cache/` i `results/llm_direct_*.csv`. Notebook tylko je wczytuje.

## Wyniki (zbiór testowy, 1214 ogłoszeń)
- `results/results.csv` – porównanie podejść (LightGBM na różnych zestawach cech, LLM bezpośrednio)
- `results/algorithms.csv` – porównanie algorytmów (Ridge, Random Forest, HistGB, XGBoost, LightGBM)

Baseline (LightGBM, cechy strukturalne): RMSE 845 zł/m².
Najlepszy model (struktura + cechy LLM): RMSE 741 zł/m² (−12,3%).
