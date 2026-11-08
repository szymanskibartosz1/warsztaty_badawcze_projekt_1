# Raport techniczny - Projekt 1: model ratingowy z interpretowalnością, wyjaśnialnością i kalibracją do 4%

**Zbiór:** `zbior_13.csv` · **Kod:** `projekt1_rating.ipynb` (wszystkie liczby w raporcie pochodzą z tego notebooka)

## 1. Problem i dane

Budujemy dwa modele szacujące PD - prawdopodobieństwo zdarzenia ryzyka (`default`) dla spółek z o.o. - na podstawie sprawozdań
finansowych: **model interpretowalny** (karta scoringowa WoE + regresja logistyczna) i **black-box** (XGBoost). PD kalibrujemy
statystycznie, dostosowujemy do tendencji centralnej 4%, mapujemy na skalę ratingową i ustalamy próg decyzyjny.

| | Wartość |
|---|---|
| Firmy / defaulty / DR | 3 000 / 445 / 14,8% |
| Cechy surowe | 222 (bilans, RZiS, przepływy, 133 wskaźniki finansowe, forma własności, PKD, schemat sprawozdania) |
| Cechy do modelowania | 189 liczbowych + 3 kategoryczne (sekcja PKD, forma własności, wielkość jednostki) |

### 1.1 Jakość danych (Data Quality)
- ±inf w 3 wskaźnikach (zerowy mianownik) → zamienione na braki; braki występują w 10 kolumnach, w 7 z nich > 50% (przepływy pieniężne) → usunięte.
- 2 kolumny stałe (`szczegolnaFormaPrawna_Symbol` - same spółki z o.o.), 10 prawie stałych i 8 identycznych kolumn - usunięte (na podstawie train).
- 182 z 189 cech ma skośność |s| > 2, mediana udziału obserwacji poza [Q1 − 3·IQR, Q3 + 3·IQR] to 12,8%. **Polityka:** nie
  usuwamy firm (skrajny wskaźnik to często sygnał ryzyka); karta używa binningu kwantylowego (odporny na ogony), drzewa nie
  wymagają cappingu, siatki PDP ograniczamy do kwantyli 2-98% z train.

### 1.2 Przeciek i powtarzające się podmioty
- Żadna cecha nie ma AUC jednowymiarowego > 0,85. Trzy wskaźniki `wsk_index_*` nie występują w słowniku, mają ok. 3 wartości na kod PKD
  (wskaźniki branżowe dla kilku okresów) i najwyższe AUC ze wszystkich cech (IV `wsk_index_skr_zaw` = 0,75, „podejrzanie silna”).
  Nie wiadomo, za jaki okres są liczone - mogą zawierać informację o skreśleniach i zawieszeniach firm **po** dacie oceny.
  **Wykluczamy je** z obu modeli. Koszt: Gini prostego logitu na valid 0,59 → 0,40, na teście 0,53 → 0,37.
- Brak identyfikatora firmy. Par „ta sama firma rok później” (to samo PKD, forma własności, kapitał, wiek różny o pełne lata ±3 dni)
  jest 61 wobec średnio 35 po losowym przemieszaniu wieku - część firm prawdopodobnie się powtarza. Stosujemy **podział grupowy**:
  51 grup wieloelementowych (109 firm) trafia w całości do jednego zbioru i jednego foldu.

### 1.3 Schemat walidacji

```
              ┌──────────── train 60% (1 801) ─────────┐┌─ valid 20% (600) ─┐┌─ test 20% (599) ─┐
dane (3000) → │ uczenie modeli, binning / WoE,         ││ kalibracja kształtu,││ wyłącznie ocena  │
              │ selekcja cech, strojenie:              ││ δ do 4%, próg       ││ końcowa          │
              │ StratifiedGroupKFold(5) wewnątrz train ││ (populacja ref.)    ││                  │
              └─────────────────────────────────────────┘└─────────────────────┘└──────────────────┘
```

Podział `StratifiedGroupKFold` (DR 14,8% w każdym zbiorze, 0 wspólnych grup). Wszystkie przekształcenia wymagające dopasowania
liczone są na train (w CV - na foldach treningowych); test nie bierze udziału w żadnej decyzji. Podział: `wyniki/podzial.csv`.

## 2. EDA
![](wykresy/eda_rozklady.png)
![](wykresy/eda_kategorie.png)

- Najsilniejsze pojedyncze cechy (AUC 0,62-0,66): kapitał własny, podatek dochodowy, zysk netto i brutto, EBITDA, ROA.
- DR różni się między sekcjami PKD (12-22%), formami własności i wielkością jednostki (mała 11,9%, mikro 16,8%).
- Korelacje: 219 par cech ma |ρ Spearmana| > 0,9, 937 par > 0,7 - wiele wskaźników to warianty tej samej miary. Ma to znaczenie
  dla selekcji cech karty, podziału wkładów SHAP i analizy kontrfaktycznej.

![](wykresy/eda_korelacje.png)

## 3. Model interpretowalny: karta scoringowa

**Przygotowanie cech (train):** binning monotoniczny (`WoEBinner` z Lab 1: ≤ 5 przedziałów, ≥ 5% obserwacji, osobny przedział
`BRAK`), WoE = ln(%dobrych / %złych). **Selekcja:** IV ≥ 0,1 (83 cechy) → filtr korelacji |ρ| < 0,6 między WoE (11 cech) →
liczba cech wybrana 5-krotną CV (najwyższe AUC przy k = 7; reguła 1 błędu std. dałaby 3 cechy, co pomija całe obszary oceny).

| Cecha | Obszar | IV | β | VIF |
|---|---|---|---|---|
| podatek_dochodowy | RZiS (proxy zysku) | 0,366 | −0,465 | 2,19 |
| Kapital_wlasny | bilans | 0,322 | −0,356 | 1,79 |
| wsk_ogolnego_zadluzenia_1 | zadłużenie | 0,291 | −0,856 | 1,20 |
| wsk_zysk_zobowiazania | rentowność | 0,246 | −0,177 | 1,76 |
| wsk_stopa_zysku_sprzedaz | rentowność | 0,220 | −0,412 | 1,70 |
| wsk_sytuacji_fin | sytuacja finansowa | 0,176 | −0,160 | 1,27 |
| pozostale_przychody_oper | RZiS | 0,174 | −0,508 | 1,46 |

- **Znaki:** wszystkie β < 0, zgodnie z konwencją WoE (wyższe WoE = niższe ryzyko) i z intuicją ekonomiczną. **VIF** < 2,2.
- **Monotoniczność na nowych danych:** na valid 19 z 25 zmian DR między sąsiednimi przedziałami ma ten sam kierunek co na train.
- **Stabilność w CV:** Gini 0,425 ± 0,110; znaki β ujemne w 34 z 35 przypadków (wyjątek: najsłabsza `wsk_zysk_zobowiazania`).
- **Karta punktowa:** 600 pkt przy szansach 50:1, PDO = 20; wynik na teście 480-597 pkt, suma punktów zgodna z modelem (sprawdzone).

![](wykresy/karta_woe_profile.png)

## 4. Model black-box: XGBoost

Wszystkie 192 cechy (kategorie one-hot, rzadkie poziomy łączone), bez wag klas. **Strojenie:** losowe przeszukiwanie 40 konfiguracji
w 5-krotnej grupowej CV na train, kryterium log-loss. Wybrano: 300 drzew, głębokość 4, `learning_rate` 0,01, `min_child_weight` 5,
`subsample` 1,0, `colsample_bytree` 0,35, λ = 5, γ = 2. **Kontrola przeuczenia:** AUC na foldach treningowych 0,91 vs 0,72 w CV -
wybór wyłącznie według CV.

## 5. Porównanie jakości

| Test (599 firm, 89 defaultów) | Karta | XGBoost |
|---|---|---|
| AUC [95% CI bootstrap] | 0,730 [0,674; 0,785] | 0,743 [0,690; 0,800] |
| Gini | 0,460 | 0,486 |
| KS | 0,353 | 0,367 |
| PR-AUC (losowy: 0,149) | 0,331 | 0,342 |
| log-loss | 0,377 | 0,372 |
| Brier | 0,115 | 0,113 |
| Gini w CV (5 foldów) | 0,425 ± 0,110 | 0,442 ± 0,101 |
| Złożoność | 7 cech, 8 parametrów, 32 przedziały | 164 używane cechy, 300 drzew, 2 915 liści |

Różnica Gini XGBoost − karta: test +0,026 [−0,039; +0,091], valid +0,033 [−0,030; +0,104] (sparowany bootstrap). **Przedziały
obejmują zero - przewaga black-boxa jest statystycznie niepewna**, a rozrzut Gini między foldami CV (0,29-0,56) jest kilka razy
większy niż różnica między modelami.

![](wykresy/roc_pr_test.png)

## 6. Interpretowalność i wyjaśnialność

**Ustawienia:** SHAP w log-szansach. Karta - wzór zamknięty φ_j = β_j·(WoE_j − E[WoE_j]), tło: train. XGBoost - `TreeExplainer`,
`tree_path_dependent`, bez tła; kolumny one-hot sumowane do cechy źródłowej. Addytywność: maks. błąd 3·10⁻⁶ (XGBoost), 9·10⁻¹⁶ (karta).

### 6.1 Wyjaśnienia globalne - najważniejsze cechy

| # | Cecha (XGBoost, średnie \|SHAP\|) | Kierunek (ρ z default, train) | Komentarz ekonomiczny |
|---|---|---|---|
| 1 | podatek_dochodowy (w karcie) | ↓ ryzyko | podatek płacą firmy z zyskiem - miara rentowności i skali |
| 2 | pozostale_przychody_oper (w karcie) | ↓ ryzyko | dodatkowe źródła przychodu, aktywna działalność |
| 3 | Kapital_wlasny (w karcie) | ↓ ryzyko | bufor na straty |
| 4 | wsk_ROE_brutto | ↓ ryzyko | rentowność kapitału |
| 5 | wsk_kapital_do_aktywa | ↓ ryzyko | udział kapitału własnego w finansowaniu - odwrotność zadłużenia |
| 6 | Aktywa | ↓ ryzyko | skala działalności |
| 7 | wsk_zast_kapitalu_podstawowego | ↑ ryzyko | kapitał podstawowy względem własnego - wysoka wartość oznacza, że straty „zjadły” kapitał |
| 8 | wsk_ogolnego_zadluzenia_1 (w karcie) | ↑ ryzyko | zobowiązania / aktywa - skokowy wzrost ryzyka powyżej ok. 0,85 |
| 9 | wsk_udzial_zob_biez_sprzedaz_1 | ↑ ryzyko (słaby, ρ = 0,01) | zobowiązania bieżące względem sprzedaży - presja płynnościowa; zależność nieliniowa |
| 10 | wsk_liczba_dni_istnienia | ↓ ryzyko | staż firmy |

Kierunki (korelacja rang z default na train) są zgodne z intuicją ekonomiczną. **Spójność z kartą:** trzy najważniejsze cechy XGBoost są w karcie; po
zgrupowaniu skorelowanych wskaźników w tematy oba modele opierają się na tych samych obszarach (zadłużenie, rentowność, zysk,
kapitał, przychody), korelacja rang tematów 0,39. Ok. 1/3 wkładu XGBoost pochodzi z cech spoza tematów karty.
Zgodność SHAP z ważnością permutacyjną dla 192 cech jest niska (ρ = 0,23) - permutacja zaniża znaczenie skorelowanych cech.

![](wykresy/shap_waznosc.png)
![](wykresy/shap_beeswarm.png)

**PDP / ICE / ALE:** kierunki PDP obu modeli zgodne dla czterech głównych cech; black-box ma krzywe nasycające się, karta - schodki.
c-ICE: efekt podatku dochodowego jest najsilniejszy dla firm mikro (−0,43 vs −0,29 log-szans dla pozostałych). Różnica PDP - ALE to
4-12% zakresu efektu, mimo bardzo silnych korelacji (dla zadłużenia ρ = 0,996). Naruszenia monotoniczności XGBoost są pomijalne (< 4%
zmienności PDP) dla podatku, kapitału, zadłużenia i rentowności sprzedaży; większe (21-29%, ale < 0,07 log-szans) dla
`pozostale_przychody_oper` i `wsk_sytuacji_fin` - kandydaci do `monotone_constraints`.

![](wykresy/pdp.png)
![](wykresy/pdp_ale.png)

### 6.2 Wyjaśnienia lokalne - studia przypadków (test)

| Przypadek | PD XGBoost | PD karta | Default | Główne czynniki (SHAP / karta) |
|---|---|---|---|---|
| najwyższe PD | 63,9% | 51,1% | nie | bardzo mały kapitał własny (ok. 2% aktywów), zadłużenie 98% aktywów, brak zysku |
| **graniczna** (PD ≈ p* = 18,2%) | 18,2% | 14,9% | nie | ↑ brak pozostałych przychodów i zysku (zerowy podatek); ↓ wysoki kapitał własny (81% aktywów) |
| typowa przyjęta | 11,4% | 8,6% | tak | wkłady małe i znoszą się; PD 11% ≈ 1 default na 9 takich firm |
| niezgodność modeli | 17,7% | 4,8% | nie | te same cechy co w karcie, ale ocenione nieliniowo, w połączeniu z innymi wskaźnikami |

Dla karty wyjaśnienie jest bezpośrednie: punkty każdej cechy i „utracone punkty względem maksimum” (kody przyczyn). Dla XGBoost:
wykres kaskadowy SHAP, kody przyczyn, LIME i analiza „what-if”. **Wniosek operacyjny:** firmy z dużą różnicą ocen obu modeli
(przypadek 4) powinny trafiać do analityka.

![](wykresy/waterfall_2.png)
![](wykresy/lime_przypadki.png)

### 6.3 Wierność wyjaśnień (fidelity)
- **SHAP - test usuwania:** zastąpienie wartością typową najważniejszej według SHAP cechy zmienia log-szanse średnio o 0,30, losowej - o 0,006;
  dla 10 cech: 0,58 vs 0,08. Ranking SHAP wiernie wskazuje cechy, od których zależy prognoza.
- **LIME - lokalne R² i błąd aproksymacji** (20 firm, nowe otoczenie losowane jak w LIME):

| Szerokość jądra | R² LIME (mediana) | R² ważone, nowe otoczenie | MAE (pp PD) | \|PD lokalne − PD modelu\| w punkcie (pp) |
|---|---|---|---|---|
| 0,25 × domyślna | 0,02 | −4,8 | 8,4 | 0,07 |
| 0,5 × domyślna | 0,21 | 0,22 | 2,6 | 6,4 |
| 1 × domyślna | 0,22 | 0,23 | 2,6 | 8,2 |
| 2 × domyślna | 0,22 | 0,23 | 2,6 | 8,2 |

Liniowy model zastępczy wyjaśnia ok. 1/5 zmienności PD XGBoost w otoczeniu firmy, a w samym punkcie myli się średnio o ok. 8 pp.
Węższe jądro poprawia zgodność w punkcie kosztem przeuczenia. **Wyjaśnienia LIME mogą być niepełne lub mylące** - zwłaszcza blisko
progu decyzyjnego; przyczyną jest schodkowy model w 192 wymiarach. LIME i SHAP wskazują często różne cechy (Jaccard top-5 0-0,67).

### 6.4 Stabilność wyjaśnień
| Eksperyment | Miara | Wynik | Źródło zmienności |
|---|---|---|---|
| LIME, 10 ziaren, ta sama firma: 1 000 / 5 000 / 20 000 próbek | Jaccard top-5 | 0,48-0,83 / 0,73-1,00 / 0,76-1,00 | metoda (losowanie) |
| | Spearman wag | 0,44-0,85 / 0,83-0,88 / 0,90-0,95 | |
| SHAP XGBoost, model uczony na 5 foldach | Jaccard top-5 / top-10 | 0,33 / 0,32; ρ rang 0,69 | model (wybór skorelowanych cech) |
| SHAP XGBoost, grupy tematyczne | Jaccard top-3 | 0,49; ρ rang 0,65 | model |
| Karta, 5 foldów | Jaccard top-3 | 0,52; ρ rang 0,69; znaki β 34/35 | model (mało defaultów) |

LIME stabilizuje się przy ≥ 5 000 próbek, ale stabilność nie oznacza wierności. TreeSHAP jest deterministyczny - zmienność
rankingu wynika z modelu: przy 190 skorelowanych wskaźnikach boosting w każdym foldzie wybiera inne „bliźniacze” cechy.
**Ograniczenia SHAP / LIME:** podział wkładu między skorelowane cechy jest umowny (zależy od wariantu SHAP i struktury drzew),
a wyjaśnienia opisują model, nie zależności przyczynowe.

## 7. Kalibracja

### 7.1 Kalibracja statystyczna (pre-cal → post-cal)
| Valid (przed) | DR | średnie PD | Brier | log-loss | ECE | CITL a | slope b |
|---|---|---|---|---|---|---|---|
| karta | 14,8% | 14,4% | 0,1148 | 0,377 | 0,030 | 0,04 | 0,95 |
| XGBoost | 14,8% | 14,1% | 0,1118 | 0,369 | 0,035 | 0,06 | 1,16 |

Wybór metody - 5-krotna CV wewnątrz valid (log-loss): karta - bez kalibracji 0,3774, Platt 0,3812, beta 0,3818, izotoniczna 0,4053;
XGBoost - 0,3690 / 0,3709 / 0,3714 / 0,4094. **Kształt jest poprawny** (modele bez wag klas), izotoniczna przeucza się przy 89
defaultach → bez korekty kształtu. Na teście Platt i beta sprowadzają CITL do 0, ale poprawiają log-loss dopiero w 4. miejscu po przecinku.

![](wykresy/kalibracja_przed.png)
![](wykresy/kalibracja_metody.png)

### 7.2 Dostosowanie do tendencji centralnej 4%
logit(q) = logit(p) + δ, δ wyznaczone na valid (populacja referencyjna): **−1,494** (karta), **−1,463** (XGBoost).

| Test | średnie PD | DR | AUC | Brier | log-loss | CITL a | slope b |
|---|---|---|---|---|---|---|---|
| karta - przed δ | 14,4% | 14,9% | 0,730 | 0,115 | 0,377 | 0,05 | 0,93 |
| karta - po δ (4%) | 4,0% | 14,9% | 0,730 | 0,131 | 0,476 | 1,54 | 0,92 |
| XGBoost - przed δ | 14,1% | 14,9% | 0,743 | 0,114 | 0,372 | 0,07 | 1,07 |
| XGBoost - po δ (4%) | 4,0% | 14,9% | 0,743 | 0,131 | 0,470 | 1,53 | 1,07 |
| karta - po δ, test przeważony do DR 4% | 3,7% | 4,0% | - | 0,037 | 0,155 | 0,10 | 0,92 |

- Ranking i AUC się nie zmieniają; zmienia się poziom PD. Względem próby (DR 14,9%) kalibracja **pogarsza się z założenia**:
  dostosowanie średniej do 4% nie jest poprawą kalibracji statystycznej.
- Po przeważeniu testu do DR = 4% (zmiana tylko udziału defaultów) PD po δ jest dobrze skalibrowane (CITL 0,10). Dostosowanie jest
  więc poprawne, jeśli portfel docelowy różni się od próby tylko udziałem defaultów (np. próba celowo wzbogacona o upadłe firmy).
  Jeśli wysoki DR wynika z gorszego okresu, PD po δ zaniża ryzyko w podobnych okresach - to cecha PD through-the-cycle.

![](wykresy/kalibracja_4proc.png)

## 8. Progi decyzyjne i ratingi

**Skala:** AAA < 1%, AA 1-2%, A 2-5%, BBB 5-8%, BB 8-12%, B 12-20%, CCC ≥ 20%, D - stwierdzony default.
Rozkład (karta / XGBoost, test): A 33% / 38%, AA 22% / 25%, BBB 20% / 16%, AAA 17% / 11%, BB-CCC 9% / 10%; HHI 0,23 / 0,25.
**Monotoniczność:** DR karty rośnie od 3% (AAA) do 50% (CCC); w XGBoost jedno odwrócenie B / CCC przy 7 firmach w CCC.
**Backtest (Jeffreys):** przed δ wszystkie klasy zielone, po δ większość czerwona - wynika wyłącznie z kotwicy 4%.

![](wykresy/backtest.png)

**Funkcja zysku** (marża 10%, LGD 45%, jednakowa ekspozycja): przyjęty dobry +0,10, przyjęty zły −0,45, odrzucony 0;
p* = 0,10 / 0,55 = 18,2%.

| Test | Próg | Akceptacja | DR wśród przyjętych | Zysk |
|---|---|---|---|---|
| karta - próg z valid (PD po δ) | 2,5% | 46,6% | 6,8% | 17,5 |
| XGBoost - próg z valid (PD po δ) | 7,0% | 88,5% | 11,1% | 20,6 |
| p* na PD po δ | 18,2% | 97-99% | 13,9-14,4% | ≈ jak przyjęcie wszystkich |
| p* na PD bez δ | 18,2% | 69-74% | 8,6-8,9% | - |
| przyjmij wszystkich | - | 100% | 14,9% | 11,0 |

p* wymaga PD skalibrowanego do portfela decyzji - na PD po δ przyjmuje prawie wszystkich; próg empiryczny z valid kompensuje
zaniżenie. Krzywe zysku są płaskie wokół optimum - w praktyce próg warto ustalić według docelowej stopy akceptacji.
**Porównanie decyzji:** przy tej samej stopie akceptacji (88,5% / 46,6%) modele zgadzają się dla 94% / 87% firm, a zysk różni się
o 1-2 jednostki; ratingi różnią się najwyżej o jedną klasę dla 97% firm.

![](wykresy/prog_zysk.png)

## 9. Analiza kontrfaktyczna

Firma z testu w klasie **BB** (PD po δ 9,9% w XGBoost, 8,9% w karcie): kapitał własny 0,3% aktywów, zadłużenie ogólne 99,7%.
**Zależności między cechami:** zamiast zmieniać jeden wskaźnik, przesuwamy **dźwignie** - grupy cech skorelowanych |ρ| ≥ 0,7
z cechą karty - o tę samą liczbę percentyli rozkładu z train; sekcja PKD, forma własności, wielkość i wiek są niezmienne.

- **XGBoost:** najmniejsza zmiana to dźwignia zadłużenia o 10 percentyli → PD 6,5%, rating **BBB**. W kwotach: spłata albo
  dokapitalizowanie ok. 1,36 mln zł (zobowiązania 2,55 → 1,19 mln zł; kapitał własny z 0,3% do ok. 15% aktywów).
  Inne pojedyncze dźwignie nie wystarczają (zysk / zobowiązania - dopiero przy 45 percentylach).
- **Karta:** brakuje 3,6 pkt; przejście zadłużenia ogólnego do sąsiedniego przedziału (z ≥ 0,85 do 0,71-0,85) daje +22 pkt.
- **Porównanie z wyjaśnieniami:** dźwignia zadłużenia ma największą sumę SHAP tej firmy (+0,79 log-szans) - zgodnie z kontrfaktem.
  LIME przypisuje tym cechom wagi bliskie zera (niska wierność LIME).
- **Realność i ograniczenia:** kontrfakt jest dalej od rzeczywistych firm niż typowa firma testowa (odległość 0,72 vs mediana 0,40) -
  inne wskaźniki zostają jak u firmy bez kapitału. Kontrfakt opisuje model, nie przyczynowość; nie uwzględnia kosztu i wykonalności zmiany.

## 10. Audyt podgrup i monitoring

| Segment (test) | n | DR | DR / PD karta | DR / PD XGBoost | AUC XGBoost |
|---|---|---|---|---|---|
| mikro | 174 | 13,2% | 0,75 | 0,79 | 0,80 |
| mała | 100 | 14,0% | 1,09 | 1,15 | 0,76 |
| pozostałe | 325 | 16,0% | 1,23 | 1,21 | 0,74 |
| forma F215 / F216 | 58 / 58 | 20,7% / 22,4% | 1,29 / 1,80 | 1,31 / 1,78 | 0,76 / 0,62 |
| sekcja C (przetwórstwo) | 82 | 17,1% | 1,52 | 1,57 | 0,68 |

Ryzyko firm mikro jest zawyżane, a formy F215 / F216 i sekcji C - zaniżane; przy 12-23 defaultach w grupie różnice mogą być
przypadkowe. Kandydat do korekty segmentowej na większej próbie. Dane nie zawierają cech osób fizycznych. PSI train → test < 0,05
dla wyniku i wszystkich cech karty. Plan monitoringu - w `MODEL_CARD.md`.

## 11. Porównanie modeli i rekomendacja

| Kryterium | Karta scoringowa | XGBoost |
|---|---|---|
| Skuteczność | Gini 0,46 | Gini 0,49 (różnica statystycznie niepewna) |
| Złożoność | 7 cech, 8 parametrów | 164 cechy, 300 drzew |
| Interpretowalność | bezpośrednia (punkty, WoE, znaki β) | pośrednia (SHAP, PDP / ALE, LIME) |
| Wiarygodność wyjaśnień | dokładne, stabilne (Jaccard top-3 0,52) | SHAP wierny, ale niestabilny między foldami (0,33); LIME mało wierny |
| Kalibracja | dobra przed δ; δ = −1,49 | dobra przed δ; δ = −1,46 |
| Decyzje | zysk 17,5 (próg 2,5%) | zysk 20,6 (próg 7%); przy równej akceptacji - podobne |

**Czy poprawa jakości black-boxa uzasadnia utratę bezpośredniej interpretowalności? Nie.** Przewaga XGBoost (+0,03 Gini) mieści
się w niepewności statystycznej, oba modele opierają się na tych samych obszarach finansowych, a wyjaśnienia black-boxa są mniej
stabilne i - w przypadku LIME - mało wierne. **Rekomendacja:** karta scoringowa jako model ratingowy, XGBoost jako model kontrolny
do monitoringu i wskazywania firm o rozbieżnych ocenach.

**Główne ograniczenia:** wykluczenie `wsk_index_*` (do potwierdzenia z autorem danych), heurystyczny podział grupowy, brak ekspozycji
i daty oceny, kotwica 4% przy DR próby 14,8%, 89 defaultów w zbiorach valid i test.
