# Raport techniczny - Projekt 1: model ratingowy z interpretowalnością, wyjaśnialnością i kalibracją do 4%

**Zbiór:** `zbior_13.csv` · **Kod:** `projekt1_rating.ipynb` (wszystkie liczby w raporcie pochodzą z tego notebooka)

## 1. Problem i dane

Budujemy dwa modele szacujące PD - prawdopodobieństwo zdarzenia ryzyka (`default`) dla spółek z o.o. - na podstawie sprawozdań
finansowych: **model interpretowalny** (karta scoringowa WoE + regresja logistyczna) i **black-box** (XGBoost). PD kalibrujemy
statystycznie, dostosowujemy do tendencji centralnej 4%, mapujemy na skalę ratingową i ustalamy próg decyzyjny.

| | Wartość |
|---|---|
| Firmy / defaulty / DR | 3 000 / 445 / 14,8% |
| Cechy surowe | 222 (bilans, RZiS, przepływy, wskaźniki finansowe, 3 wskaźniki branżowe `wsk_index_*`, forma własności, PKD, schemat sprawozdania) |
| Cechy do modelowania | 192 liczbowe + 3 kategoryczne (sekcja PKD, forma własności, wielkość jednostki) = 195 |

### 1.1 Jakość danych (Data Quality)
- ±inf w 3 wskaźnikach (zerowy mianownik) → zamienione na braki; braki występują w 10 kolumnach, w 7 z nich > 50% (przepływy pieniężne) → usunięte.
- 2 kolumny stałe (`szczegolnaFormaPrawna_Symbol` - same spółki z o.o.), 10 prawie stałych i 8 identycznych kolumn - usunięte (na podstawie train).
- Typ sprawozdania: `schemat_wsk_bilans` → cecha kategoryczna `wielkosc` (mikro / mała / pozostałe); `schemat_wsk_rzis` pominięty -
  jest zgodny z `schemat_wsk_bilans` w 97,6% obserwacji. Kod PKD zastąpiony sekcją PKD.
- 182 z 192 cech liczbowych ma skośność |s| > 2, mediana udziału obserwacji poza [Q1 − 3·IQR, Q3 + 3·IQR] to 12,7%. **Polityka:** nie
  usuwamy firm (skrajny wskaźnik to często sygnał ryzyka); karta używa binningu kwantylowego (odporny na ogony), drzewa nie
  wymagają cappingu, siatki PDP ograniczamy do kwantyli 2-98% z train.

### 1.2 Przeciek, wskaźniki branżowe i powtarzające się podmioty
- Żadna cecha nie ma AUC jednowymiarowego > 0,85. Trzy wskaźniki `wsk_index_*` (skreślenia / zawieszenia, nowe i aktywne firmy)
  nie występują w słowniku i mają ok. 3 wartości na kod PKD (średnio 2,98) - to **wskaźniki branżowe liczone dla kilku okresów**.
  Mają najwyższe AUC jednowymiarowe (`wsk_index_nowe` 0,718, `wsk_index_skr_zaw` 0,708) i IV w przedziale „podejrzanie silna”
  (0,61 i 0,58). **Prowadzący potwierdził, że nie zawierają informacji z przyszłości**, więc używamy ich w obu modelach.
  Wnoszą dużo: Gini prostego logitu na valid 0,397 → 0,586, na teście 0,373 → 0,534. Firma nie ma na nie wpływu, a po wdrożeniu
  trzeba je aktualizować dla bieżącego okresu.
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

- Najsilniejsze pojedyncze cechy: wskaźniki branżowe `wsk_index_nowe` i `wsk_index_skr_zaw` (AUC 0,71-0,72), dalej kapitał własny,
  podatek dochodowy, zysk netto i brutto, EBITDA, ROA (AUC 0,63-0,66).
- DR różni się między sekcjami PKD (12-22%), formami własności i wielkością jednostki (mała 11,9%, mikro 16,8%).
- Korelacje: 219 par cech ma |ρ Spearmana| > 0,9, 937 par > 0,7 - wiele wskaźników to warianty tej samej miary. Ma to znaczenie
  dla selekcji cech karty, podziału wkładów SHAP, ważności permutacyjnej i analizy kontrfaktycznej.

![](wykresy/eda_korelacje.png)

## 3. Model interpretowalny: karta scoringowa

**Przygotowanie cech (train):** binning monotoniczny (`WoEBinner` z Lab 1: ≤ 5 przedziałów, ≥ 5% obserwacji, osobny przedział
`BRAK`), WoE = ln(%dobrych / %złych). **Selekcja:** IV ≥ 0,1 (85 cech) → filtr korelacji |ρ| < 0,6 między WoE (12 cech) →
liczba cech wybrana 5-krotną CV (najwyższe AUC przy k = 8; reguła 1 błędu std. dałaby 3 cechy, co pomija całe obszary oceny).
`wsk_index_skr_zaw` nie wchodzi do karty - jest skorelowany z `wsk_index_nowe` (|ρ| = 0,65).

| Cecha | Obszar | IV | β | p-value | VIF |
|---|---|---|---|---|---|
| wsk_index_nowe | wskaźnik branżowy | 0,610 | −0,938 | < 0,001 | 1,03 |
| podatek_dochodowy | RZiS (proxy zysku) | 0,366 | −0,512 | 0,004 | 2,19 |
| Kapital_wlasny | bilans | 0,322 | −0,434 | 0,009 | 1,79 |
| wsk_ogolnego_zadluzenia_1 | zadłużenie | 0,291 | −0,725 | < 0,001 | 1,22 |
| wsk_zysk_zobowiazania | rentowność | 0,246 | −0,147 | 0,449 | 1,76 |
| wsk_stopa_zysku_sprzedaz | rentowność | 0,220 | −0,403 | 0,040 | 1,71 |
| wsk_sytuacji_fin | sytuacja finansowa | 0,176 | −0,185 | 0,324 | 1,27 |
| pozostale_przychody_oper | RZiS | 0,174 | −0,248 | 0,237 | 1,48 |

- **Znaki:** wszystkie β < 0, zgodnie z konwencją WoE (wyższe WoE = niższe ryzyko) i z intuicją ekonomiczną. **VIF** < 2,2.
- **Istotność β:** pięć cech ma p < 0,05; trzy najsłabsze (`wsk_zysk_zobowiazania`, `wsk_sytuacji_fin`, `pozostale_przychody_oper`)
  są nieistotne pojedynczo przy skorelowanych sąsiadach, ale liczbę cech wybrano w CV, a ich znaki są poprawne i stabilne.
- **Monotoniczność na nowych danych:** na valid 23 z 29 zmian DR między sąsiednimi przedziałami ma ten sam kierunek co na train.
- **Stabilność w CV:** Gini 0,546 ± 0,105; znaki β ujemne w 39 z 40 przypadków (wyjątek: najsłabsza `wsk_zysk_zobowiazania`, 4 / 5).
- **Karta punktowa:** 600 pkt przy szansach 50:1, PDO = 20; wynik na teście 463-623 pkt (mediana 554), suma punktów zgodna z modelem (sprawdzone).

![](wykresy/karta_woe_profile.png)

## 4. Model black-box: XGBoost

Wszystkie 195 cech (kategorie one-hot, rzadkie poziomy łączone), bez wag klas. **Strojenie:** losowe przeszukiwanie 40 konfiguracji
w 5-krotnej grupowej CV na train, kryterium log-loss. Wybrano: 500 drzew, głębokość 5, `learning_rate` 0,02, `min_child_weight` 20,
`subsample` 0,6, `colsample_bytree` 0,2, λ = 5, γ = 0 (`wyniki/xgboost_parametry.json`). **Kontrola przeuczenia:** AUC na foldach
treningowych 0,886 vs 0,791 w CV - najmniejsza luka wśród najlepszych konfiguracji; wybór wyłącznie według CV.

## 5. Porównanie jakości

| Test (599 firm, 89 defaultów) | Karta | XGBoost |
|---|---|---|
| AUC [95% CI bootstrap] | 0,781 [0,724; 0,833] | 0,793 [0,740; 0,842] |
| Gini | 0,562 | 0,586 |
| KS | 0,440 | 0,462 |
| PR-AUC (losowy: 0,149) | 0,430 | 0,468 |
| log-loss | 0,350 | 0,341 |
| Brier | 0,105 | 0,102 |
| Gini w CV (5 foldów) | 0,546 ± 0,105 | 0,581 ± 0,091 |
| Złożoność | 8 cech, 9 parametrów, 37 przedziałów | 158 używanych cech, 500 drzew, 2 305 liści |

Różnica Gini XGBoost − karta (sparowany bootstrap): test **+0,024 [−0,032; +0,083]**, valid **+0,055 [+0,008; +0,099]**.
Przewaga black-boxa jest **umiarkowana**: istotna na valid, ale na teście przedział obejmuje zero, a rozrzut Gini między foldami CV
(karta 0,43-0,66, XGBoost 0,48-0,67) jest kilka razy większy niż różnica między modelami.

![](wykresy/roc_pr_test.png)

## 6. Interpretowalność i wyjaśnialność

**Ustawienia:** SHAP w log-szansach. Karta - wzór zamknięty φ_j = β_j·(WoE_j − E[WoE_j]), tło: train. XGBoost - `TreeExplainer`,
wariant **interventional** z tłem 300 firm z train - ta sama (interwencyjna) funkcja wartości co wzór zamknięty karty (wykład 4);
przy 219 parach cech z |ρ| > 0,9 `tree_path_dependent` przybliżałby funkcję warunkową z dużym błędem. Wariant path-dependent
stosujemy tylko do wartości interakcji SHAP (6.4). Kolumny one-hot sumowane do cechy źródłowej. Addytywność: maks. błąd 4·10⁻⁶
(XGBoost), 9·10⁻¹⁶ (karta). Wartość bazowa φ₀: −2,374 (XGBoost), −2,215 (karta) - σ(φ₀) nie jest średnim PD (14,8%).

### 6.1 Wyjaśnienia globalne - najważniejsze cechy

| # | Cecha (XGBoost, średnie \|SHAP\|) | Kierunek (ρ z default, train) | Komentarz ekonomiczny |
|---|---|---|---|
| 1 | wsk_index_nowe (w karcie) | ↑ ryzyko | napływ nowych firm w branży - wysoka rotacja i konkurencja |
| 2 | wsk_index_skr_zaw | ↑ ryzyko | odsetek skreśleń i zawieszeń w branży - kondycja branży w danym okresie |
| 3 | pozostale_przychody_oper (w karcie) | ↓ ryzyko | dodatkowe źródła przychodu, aktywna działalność |
| 4 | podatek_dochodowy (w karcie) | ↓ ryzyko | podatek płacą firmy z zyskiem - miara rentowności i skali |
| 5 | Kapital_wlasny (w karcie) | ↓ ryzyko | bufor na straty |
| 6 | wsk_zast_kapitalu_podstawowego | ↑ ryzyko | kapitał podstawowy względem własnego - wysoka wartość oznacza, że straty „zjadły” kapitał |
| 7 | wsk_liczba_dni_istnienia | ↓ ryzyko (słaby) | staż firmy |
| 8 | Aktywa | ↓ ryzyko | skala działalności |
| 9 | wsk_stopa_zysku_sprzedaz (w karcie) | ↓ ryzyko | rentowność sprzedaży |
| 10 | wsk_struktura_kap_wlasnego_s_2 | ↓ ryzyko | udział kapitału własnego w finansowaniu |

Kierunki (korelacja rang z default na train) są zgodne z intuicją ekonomiczną. **Spójność z kartą:** dwie najważniejsze cechy
XGBoost to wskaźniki branżowe, a 4 z 5 pierwszych cech są w karcie. Po zgrupowaniu skorelowanych wskaźników w tematy oba modele
opierają się na tych samych obszarach (branża, zadłużenie, kapitał, rentowność, zysk, przychody) - korelacja rang tematów 0,57.
Ok. połowa (49%) wkładu XGBoost pochodzi z cech spoza tematów karty, w tym z `wsk_index_skr_zaw`, którego karta nie zawiera.
Zgodność rankingu SHAP z ważnością permutacyjną dla 195 cech jest niska (ρ = 0,15) - permutacja pojedynczej cechy zaniża
znaczenie cech mających skorelowanych „bliźniaków” (np. `Aktywa`, `wsk_stopa_zysku_sprzedaz`: wysoki SHAP, ujemna ważność permutacyjna).

![](wykresy/shap_waznosc.png)
![](wykresy/shap_beeswarm.png)

### 6.2 Ważność grup skorelowanych cech (Lab 6)

Tematy z 6.1 (grupy cech skorelowanych |ρ| ≥ 0,7 z cechą karty) oceniamy dwiema metodami z Lab 6: **permutacja grupowa** (cała grupa
mieszana tą samą permutacją, valid, 5 powtórzeń) i **drop-column** (XGBoost uczony od nowa bez grupy). MDI pomijamy - jest liczone
na danych uczących i premiuje cechy ciągłe. AUC black-boxa na valid: 0,824.

| Grupa (liczba cech) | Permutacja grupy (spadek AUC) | Suma permutacji pojedynczych | Drop-column grupy | Suma \|SHAP\| |
|---|---|---|---|---|
| pozostałe cechy (129) | 0,071 | 0,052 | 0,024 | 1,67 |
| wsk_index_nowe (1) | 0,051 | 0,041 | 0,014 | 0,39 |
| zadłużenie ogólne (17) | 0,034 | 0,001 | 0,004 | 0,40 |
| kapitał własny (8) | 0,009 | 0,004 | −0,003 | 0,33 |
| podatek dochodowy (2) | 0,004 | 0,003 | 0,002 | 0,13 |
| zysk / zobowiązania (30) | 0,002 | 0,002 | −0,005 | 0,24 |

- **Efekt bliźniaków:** grupa zadłużenia ma permutację grupową 0,034 przy sumie permutacji pojedynczych 0,001 - model korzysta
  z niej mocno, ale pojedyncza permutacja tego nie widzi, bo informację przenoszą pozostałe cechy grupy.
- **Drop-column** jest bliski zera lub ujemny dla większości grup: po usunięciu grupy model odtwarza informację z innych cech
  (permutacja mierzy „czy model korzysta”, drop-column - „czy informację da się zastąpić”). Tylko wskaźnik branżowy i reszta cech
  są nie do zastąpienia (wyraźnie dodatni drop-column).
- Zgodność rang grup: permutacja grupowa vs suma |SHAP| ρ = 0,90, permutacja vs drop-column ρ = 0,68. Na poziomie grup SHAP
  i permutacja są zgodne - rozbieżność z 6.1 wynika z korelacji cech, nie z błędu metod.

### 6.3 PDP / ICE / ALE

![](wykresy/pdp.png)
![](wykresy/pdp_ale.png)

- Kierunki PDP obu modeli są zgodne dla czterech głównych cech (`wsk_index_nowe`, `wsk_index_skr_zaw`, podatek, zadłużenie);
  black-box ma krzywe nasycające się, karta - schodki.
- **c-ICE:** wzrost `wsk_index_nowe` z 0,126 do 0,505 podnosi log-szanse średnio o +1,05 (10% firm < +0,79, 10% > +1,25); efekt jest
  podobny w grupach wielkości (mikro +1,01, mała +1,06, pozostałe +1,07) - brak wyraźnej interakcji z wielkością.
- **ALE** (K = 20 przedziałów kwantylowych, min. 20-28 firm w przedziale): różnica PDP - ALE to 2-5% zakresu efektu, mimo silnych
  korelacji (dla zadłużenia ρ = 0,996 z innym wskaźnikiem). PDP nie jest więc tu istotnie zniekształcone przez korelacje.
- **Monotoniczność XGBoost:** brak naruszeń dla `wsk_index_*`, podatku i zysku / zobowiązania; pomijalne dla kapitału (2,5% zmienności
  PDP) i rentowności sprzedaży (5%); większe dla zadłużenia (13%), `wsk_sytuacji_fin` (25%) i `pozostale_przychody_oper` (26%, maks.
  0,085 log-szans) - kandydaci do `monotone_constraints`. Znaki β karty są zgodne z oczekiwanymi kierunkami dla wszystkich 8 cech.

### 6.4 Interakcje: statystyka H Friedmana (Lab 6)

Kandydatów wybieramy z wartości interakcji SHAP (path-dependent, 300 firm z testu) - 4 pary o największej średniej |interakcji|.
Każdą parę weryfikujemy **statystyką H Friedmana** (Lab 6) w log-szansach, na **dwóch niezależnych próbach** po 150 firm, razem
z **parą kontrolną** (dwie ważne cechy o najmniejszej interakcji SHAP) wyznaczającą poziom szumu metody.

| Para | Interakcja SHAP | H (próba 1 / 2) | RMS log-szans (próba 1 / 2) |
|---|---|---|---|
| wsk_index_nowe × pozostale_przychody_oper | 0,021 | 0,047 / 0,049 | 0,022 / 0,022 |
| wsk_index_skr_zaw × wsk_koszty_przychody | 0,011 | 0,030 / 0,029 | 0,013 / 0,012 |
| wsk_index_skr_zaw × wielkosc | 0,011 | 0,025 / 0,025 | 0,011 / 0,011 |
| wsk_index_skr_zaw × wsk_index_nowe | 0,013 | 0,015 / 0,018 | 0,012 / 0,013 |
| kontrolna: pozostale_przychody_oper × Kapital_wlasny | 0,000 | 0,000 / 0,000 | 0,000 / 0,000 |

Interakcje są **rzeczywiste** (powtarzalne na obu próbach, wyraźnie powyżej pary kontrolnej), ale **słabe**: H < 0,05, czyli
interakcja odpowiada za kilka procent wspólnej zmienności pary, a RMS ≤ 0,022 log-szans. Model jest w praktyce prawie addytywny -
to wspiera wybór addytywnej karty scoringowej.

### 6.5 Global surrogate: drzewo naśladujące black-box (Lab 6)

Drzewo regresyjne uczone na log-szansach XGBoost w train (min. 50 firm w liściu), PD drzewa przechodzi ten sam łańcuch kalibracji
i δ. Wierność na teście (37 firm odrzuconych przez XGBoost przy progu operacyjnym z sekcji 8):

| Głębokość | Liście | Wierność R² | Zgodność decyzji | Odrzuceni przez XGBoost odrzuceni też przez drzewo | Zgodność klas | Gini drzewa |
|---|---|---|---|---|---|---|
| 2 | 4 | 0,559 | 93,8% | 0% | 41,2% | 0,370 |
| 3 | 8 | 0,669 | 95,3% | 59,5% | 42,2% | 0,482 |
| 4 | 15 | 0,760 | 96,3% | 43,2% | 52,1% | 0,529 |
| 5 | 21 | 0,795 | 96,3% | 43,2% | 54,4% | 0,548 |
| 8 | 29 | 0,805 | 96,3% | 43,2% | 54,4% | 0,556 |

![](wykresy/surogat.png)

- Drzewo o głębokości 3 dzieli najpierw według `wsk_index_nowe` (≤ 0,282), potem według podatku dochodowego / zysku brutto,
  `wsk_index_skr_zaw` i struktury kapitału obcego; PD modelu końcowego w liściach 0,4-16,5%.
- Wysoka **zgodność decyzji ogółem (94-96%) jest myląca** - większość firm jest przyjmowana. Drzewo odtwarza tylko 43-60%
  odrzuceń black-boxa, a klasy ratingowe zgadzają się dla ok. połowy firm. Surogat nadaje się do komunikacji (reguły dla komitetu),
  nie do decyzji ani jako wyjaśnienie pojedynczej odmowy. Nawet drzewo o 29 liściach (Gini 0,556) nie dorównuje karcie (0,562).

### 6.6 Wyjaśnienia lokalne - studia przypadków (test)

| Przypadek | PD XGBoost | PD karta | Default | Główne czynniki (SHAP / karta) |
|---|---|---|---|---|
| najwyższe PD | 71,8% | 68,2% | tak | wysokie wskaźniki branżowe (`wsk_index_nowe` 0,46, `wsk_index_skr_zaw` 0,48), kapitał własny 12,6 tys. zł, zadłużenie 94,5% aktywów, brak zysku (zerowy podatek) |
| **graniczna** (PD ≈ p* = 18,2%) | 18,2% | 22,6% | nie | ↑ `wsk_index_nowe`, `wsk_zast_kapitalu_podstawowego` = 60, zerowy podatek; ↓ niski `wsk_index_skr_zaw` (0,20) |
| typowa przyjęta | 8,3% | 8,6% | nie | ↑ wskaźniki branżowe, strata na sprzedaży; ↓ podatek, pozostałe przychody, kapitał (1,86 mln zł) |
| niezgodność modeli | 16,3% | 3,4% | tak | XGBoost: wysoki `wsk_index_skr_zaw` (+0,52 log-szans; brak w karcie) i mały kapitał; karta: niski `wsk_index_nowe` daje 94,9 pkt (prawie maksimum) |

Dla karty wyjaśnienie jest bezpośrednie: punkty każdej cechy i „utracone punkty względem maksimum” (kody przyczyn). Dla XGBoost:
wykres kaskadowy SHAP, kody przyczyn, LIME i analiza „what-if”. **Wniosek operacyjny:** firmy z dużą różnicą ocen obu modeli
(przypadek 4 - firma, u której faktycznie wystąpił default) powinny trafiać do analityka.

![](wykresy/waterfall_2.png)
![](wykresy/lime_przypadki.png)

### 6.7 Wierność wyjaśnień (fidelity)
- **SHAP - test usuwania:** zastąpienie wartością typową najważniejszej według SHAP cechy zmienia log-szanse średnio o 0,56, losowej -
  o 0,015; dla 10 cech: 0,87 vs 0,10. Ranking SHAP wiernie wskazuje cechy, od których zależy prognoza.
- **LIME - lokalne R² i błąd aproksymacji** (20 firm, nowe otoczenie losowane jak w LIME):

| Szerokość jądra | R² LIME (mediana) | R² ważone, nowe otoczenie | MAE (pp PD) | \|PD lokalne − PD modelu\| w punkcie (pp) |
|---|---|---|---|---|
| 0,25 × domyślna | 0,02 | −1,46 | 9,1 | 0,06 |
| 0,5 × domyślna | 0,54 | 0,54 | 4,3 | 6,5 |
| 1 × domyślna | 0,56 | 0,56 | 4,3 | 7,3 |
| 2 × domyślna | 0,56 | 0,56 | 4,3 | 7,4 |

Liniowy model zastępczy wyjaśnia ok. połowy zmienności PD XGBoost w otoczeniu firmy, ale w samym punkcie myli się średnio o ok. 7 pp.
Węższe jądro poprawia zgodność w punkcie kosztem przeuczenia. **Wyjaśnienia LIME mogą być niepełne lub mylące** - zwłaszcza blisko
progu decyzyjnego; przyczyną jest schodkowy model w 195 wymiarach. LIME i SHAP wskazują częściowo różne cechy (Jaccard top-5 0,43-1,0;
wśród wspólnych cech kierunek zgodny tylko w 3-4 przypadkach z 6-7).

### 6.8 Stabilność wyjaśnień
| Eksperyment | Miara | Wynik | Źródło zmienności |
|---|---|---|---|
| LIME, 10 ziaren, ta sama firma: 1 000 / 5 000 / 20 000 próbek | Jaccard top-5 | 0,56-0,79 / 0,70-1,00 / 0,77-1,00 | metoda (losowanie) |
| | Spearman wag | 0,71-0,77 / 0,84-0,94 / 0,95-0,98 | |
| SHAP XGBoost, model uczony na 5 foldach | Jaccard top-5 / top-10 | 0,46 / 0,57; ρ rang 0,77 | model (wybór skorelowanych cech) |
| SHAP XGBoost, grupy tematyczne | Jaccard top-3 | 1,00; ρ rang 0,91 | model |
| Karta, 5 foldów | Jaccard top-3 | 0,60; ρ rang 0,83; znaki β 39/40 | model (mało defaultów) |

LIME stabilizuje się przy ≥ 5 000 próbek, ale stabilność nie oznacza wierności. TreeSHAP jest deterministyczny - zmienność
rankingu wynika z modelu: przy 190 skorelowanych wskaźnikach boosting w każdym foldzie wybiera inne „bliźniacze” cechy, ale
ranking **grup** jest w pełni stabilny. Dlatego monitoring wyjaśnień prowadzimy na poziomie grup (sekcja 10).
**Ograniczenia SHAP / LIME:** podział wkładu między skorelowane cechy jest umowny (zależy od wariantu SHAP i struktury drzew),
a wyjaśnienia opisują model, nie zależności przyczynowe.

## 7. Kalibracja

### 7.1 Kalibracja statystyczna (pre-cal → post-cal)
| Valid (przed) | DR | średnie PD | Brier | log-loss | ECE | CITL a | slope b |
|---|---|---|---|---|---|---|---|
| karta | 14,8% | 13,5% | 0,1031 | 0,340 | 0,029 | 0,13 | 1,06 |
| XGBoost | 14,8% | 13,4% | 0,1009 | 0,328 | 0,017 | 0,14 | 1,14 |

Wybór metody - 5-krotna CV wewnątrz valid (log-loss): karta - bez kalibracji 0,3402, Platt 0,3444, beta 0,3447, izotoniczna 0,4002;
XGBoost - 0,3275 / 0,3306 / 0,3318 / 0,3741. **Kształt jest poprawny** (modele bez wag klas), izotoniczna przeucza się przy 89
defaultach → bez korekty kształtu. Na teście Platt i beta przestrzeliwują średnią (PD 16,2-16,4% przy DR 14,9%, CITL −0,13 do −0,15)
i pogarszają log-loss - potwierdzenie decyzji.

![](wykresy/kalibracja_przed.png)
![](wykresy/kalibracja_metody.png)

### 7.2 Dostosowanie do tendencji centralnej 4%
logit(q) = logit(p) + δ, δ wyznaczone na valid (populacja referencyjna): **−1,486** (karta), **−1,488** (XGBoost).

| Test | średnie PD | DR | AUC | Brier | log-loss | CITL a | slope b |
|---|---|---|---|---|---|---|---|
| karta - przed δ | 14,7% | 14,9% | 0,781 | 0,105 | 0,350 | 0,02 | 0,95 |
| karta - po δ (4%) | 4,4% | 14,9% | 0,781 | 0,124 | 0,441 | 1,50 | 0,95 |
| XGBoost - przed δ | 14,7% | 14,9% | 0,793 | 0,102 | 0,341 | 0,01 | 0,95 |
| XGBoost - po δ (4%) | 4,6% | 14,9% | 0,793 | 0,121 | 0,430 | 1,50 | 0,95 |
| karta - po δ, test przeważony do DR 4% | 3,7% | 4,0% | - | 0,036 | 0,145 | 0,07 | 0,97 |
| XGBoost - po δ, test przeważony do DR 4% | 3,8% | 4,0% | - | 0,035 | 0,142 | 0,07 | 0,98 |

- Ranking i AUC się nie zmieniają; zmienia się poziom PD. Względem próby (DR 14,9%) kalibracja **pogarsza się z założenia**:
  dostosowanie średniej do 4% nie jest poprawą kalibracji statystycznej.
- Po przeważeniu testu do DR = 4% (zmiana tylko udziału defaultów) PD po δ jest dobrze skalibrowane (CITL 0,07; bez δ: −1,41).
  Dostosowanie jest więc poprawne, jeśli portfel docelowy różni się od próby tylko udziałem defaultów (np. próba celowo wzbogacona
  o upadłe firmy). Jeśli wysoki DR wynika z gorszego okresu, PD po δ zaniża ryzyko w podobnych okresach - to cecha PD through-the-cycle.

![](wykresy/kalibracja_4proc.png)

## 8. Progi decyzyjne i ratingi

**Skala:** AAA < 1%, AA 1-2%, A 2-5%, BBB 5-8%, BB 8-12%, B 12-20%, CCC ≥ 20%, D - stwierdzony default.
Rozkład (karta / XGBoost, test): AAA 27,5% / 30,6%, AA 20,0% / 19,5%, A 25,4% / 22,9%, BBB 8,5% / 10,5%, BB-CCC 18,5% / 16,5%; HHI 0,20 / 0,20.
**Monotoniczność:** DR karty rośnie od 4,8% (AAA) do 68,8% (CCC); w XGBoost jedno odwrócenie A / BBB (16,1% vs 11,1%, 63 firmy w BBB).
**Backtest (test Jeffreysa i dokładny test dwumianowy):** przed δ - karta: 6 klas zielonych, AAA żółta; XGBoost: wszystkie zielone.
Po δ 6 z 7 klas czerwonych w obu modelach; test dwumianowy daje nieco wyższe p-value (np. BBB karty 0,046 vs 0,028), ale te same
wnioski. Czerwone światła wynikają wyłącznie z kotwicy 4% wobec DR testu 14,9%.

![](wykresy/backtest.png)

**Funkcja zysku** (marża 10%, LGD 45%, jednakowa ekspozycja): przyjęty dobry +0,10, przyjęty zły −0,45, odrzucony 0;
p* = 0,10 / 0,55 = 18,2%. **Próg empiryczny wyznaczamy na valid przeważonym do DR = 4%** (wykład 3: najpierw δ, potem próg) - tylko
wtedy PD po δ jest skalibrowane względem portfela, na którym liczymy zysk. Wynik: **16,5% (karta), 16,0% (XGBoost)** - blisko p*.

| Test | Próg | Akceptacja | DR wśród przyjętych | Zysk |
|---|---|---|---|---|
| karta - portfel 4%, próg z valid | 16,5% | 97,9% | 3,4% | 47,5 |
| XGBoost - portfel 4%, próg z valid | 16,0% | 96,6% | 3,0% | 48,4 |
| portfel 4%, p* (karta / XGBoost) | 18,2% | 98,6% / 97,4% | 3,6% / 3,2% | 47,5 / 48,1 |
| portfel 4%, przyjmij wszystkich | - | 100% | 4,0% | 46,7 |
| portfel jak próba, p* na PD bez δ (karta / XGBoost) | 18,2% | 72,5% / 71,3% | 7,8% / 7,5% | 24,7 / 25,1 |
| portfel jak próba, przyjmij wszystkich | - | 100% | 14,9% | 11,0 |

Wyniki dla portfela 4% są ważone (test przeważony do DR 4%, suma wag 599). Próg empiryczny i p* dają prawie ten sam zysk -
przy portfelu o DR 4% model odrzuca tylko 2-3,5% najgorszych firm, a krzywe zysku są płaskie wokół optimum.
**Tabela decyzyjna (XGBoost, test bez wag):** obniżanie progu z 20% do 3% zmniejsza akceptację z 96,0% do 59,3%, a DR wśród przyjętych
z 12,7% do 5,1%; liczba odrzuconych dobrych firm rośnie szybciej niż unikniętych złych (jeden uniknięty default „kosztuje” 4,5 dobrej firmy).
**Porównanie decyzji (test, progi z valid):** zgodność decyzji 96,8%, ten sam rating 54,4%, różnica ≤ 1 klasa 91,8%; z 19 firm
o różnej decyzji 17 odrzuca tylko XGBoost. Przy tej samej stopie akceptacji (93,8% / 96,3%) decyzje zgadzają się w 94,5% / 98,0%,
a zysk różni się o 1,8 / 0 jednostek.

![](wykresy/prog_zysk.png)

### 8.1 Firma graniczna przy progu operacyjnym: SHAP modelu końcowego

Decyzję podejmujemy na PD po δ i progu 16,0%, więc firmę graniczną wybieramy właśnie tam (XGBoost). Kształtu nie kalibrujemy, a δ jest
stałym przesunięciem logitu, więc **SHAP modelu końcowego = te same wkłady φ, wartość bazowa φ₀ + δ** (wykład 4); kontrola:
φ₀ + δ + Σφ = −1,665 = logit PD końcowego.

Firma (wiersz testu 431): PD końcowe 15,9% (tuż poniżej progu - przyjęta), rating B, bez defaultu. Ryzyko podnoszą przede wszystkim
wskaźniki branżowe (`wsk_index_nowe` +0,71, `wsk_index_skr_zaw` +0,53 log-szans), dalej mały kapitał własny (31 tys. zł, +0,21),
`wsk_zast_kapitalu_podstawowego` = 50 (+0,12), pozostałe przychody (+0,12) i zerowy podatek (+0,11). Decyzja zależy więc w dużej mierze
od kondycji branży, na którą firma nie ma wpływu - ważna informacja dla analityka.

![](wykresy/waterfall_graniczna_4proc.png)

## 9. Analiza kontrfaktyczna

Firma z testu w klasie **BB w obu modelach** (jedna z 14 takich; wiersz 582, PD po δ 9,7% w XGBoost, 11,9% w karcie, bez defaultu):
kapitał własny 7,6 tys. zł, zadłużenie ogólne 81%, zerowy podatek, ujemna rentowność sprzedaży.
**Zależności między cechami:** zamiast zmieniać jeden wskaźnik, przesuwamy **dźwignie** - grupy cech skorelowanych |ρ| ≥ 0,7
z cechą karty - o tę samą liczbę percentyli rozkładu z train. Niezmienne: sekcja PKD, forma własności, wielkość, wiek oraz
wskaźniki branżowe `wsk_index_*` (firma nie ma na nie wpływu).

- **XGBoost:** najtańsze pojedyncze dźwignie (koszt 10 percentyli) - podatek dochodowy → PD 7,8%, rating **BBB**, albo zadłużenie
  ogólne → PD 7,3%, BBB; dalej kapitał własny (15 percentyli, PD 7,6%) i rentowność sprzedaży (20 percentyli, PD 7,4%).
  Dźwignia podatku w kwotach: zysk brutto z −9,7 tys. zł do 0, podatek z 0 do ok. 2,0 tys. zł - przesunięcie percentylowe
  nie zachowuje dokładnie relacji księgowych, więc kwoty są przybliżone.
- **Karta:** brakuje 12,6 pkt (502 → 515, PD < 8%); przejście zadłużenia ogólnego do niższego przedziału (z 0,71-0,85 do < 0,71)
  daje +13,2 pkt i samo wystarcza; podatek o dwa przedziały w górę daje +16,5 pkt. **Oba modele wskazują zadłużenie** jako
  skuteczną dźwignię.
- **Porównanie z wyjaśnieniami:** największe sumy SHAP tej firmy mają dźwignie kapitału (+0,52) i zadłużenia (+0,50) - zadłużenie
  jest też jednym z dwóch najtańszych kontrfaktów; podatek ma mniejszy SHAP (+0,16), ale skok z zera jest „tani” w percentylach. LIME przypisuje
  cechom dźwigni wagi bliskie zera (−0,02 do +0,01) - niska wierność LIME.
- **Realność i ograniczenia:** odległość kontrfaktu do najbliższej firmy z train 0,586 wobec mediany 0,490 i 90. percentyla 0,653
  dla firm testowych - kontrfakt jest nietypowy, ale w zakresie rzeczywistych firm. Kontrfakt opisuje model, nie przyczynowość;
  nie uwzględnia kosztu i wykonalności zmiany.

## 10. Audyt podgrup i monitoring

| Segment (test) | n | defaulty | DR | DR / PD karta | DR / PD XGBoost | AUC XGBoost |
|---|---|---|---|---|---|---|
| mikro | 174 | 23 | 13,2% | 0,76 | 0,77 | 0,87 |
| mała | 100 | 14 | 14,0% | 1,08 | 1,22 | 0,78 |
| pozostałe | 325 | 52 | 16,0% | 1,17 | 1,11 | 0,78 |
| forma F215 / F216 | 58 / 58 | 12 / 13 | 20,7% / 22,4% | 1,10 / 1,83 | 1,12 / 1,95 | 0,80 / 0,67 |
| sekcja C (przetwórstwo) | 82 | 14 | 17,1% | 1,69 | 1,72 | 0,66 |

Ryzyko firm mikro jest zawyżane, a formy F216 i sekcji C - zaniżane (przy 13-14 defaultach różnice mogą być przypadkowe);
forma F215 jest teraz skalibrowana. Kandydat do korekty segmentowej na większej próbie. Dane nie zawierają cech osób fizycznych.
PSI train → test < 0,05 dla wyniku obu modeli, wszystkich cech karty, `wsk_index_skr_zaw` i cech kategorycznych.

**Monitoring wyjaśnień:** ranking pojedynczych cech SHAP jest niestabilny nawet bez zmiany populacji (Jaccard top-5 0,46 między
foldami), więc alarm oparty na Jaccard top-5 byłby fałszywy. Monitorujemy **względną zmianę średniego |SHAP| grup tematycznych**
(ranking grup stabilny: Jaccard top-3 1,00) względem walidacji oraz znaki β karty. Pełny plan monitoringu - w `MODEL_CARD.md`.

## 11. Porównanie modeli i rekomendacja

| Kryterium | Karta scoringowa | XGBoost |
|---|---|---|
| Skuteczność | Gini 0,562 | Gini 0,586 (różnica +0,024 [−0,032; +0,083] na teście; +0,055 [+0,008; +0,099] na valid) |
| Złożoność | 8 cech, 9 parametrów | 158 cech, 500 drzew |
| Interpretowalność | bezpośrednia (punkty, WoE, znaki β) | pośrednia (SHAP, PDP / ALE, LIME, surogat) |
| Wiarygodność wyjaśnień | dokładne, stabilne (Jaccard top-3 0,60) | SHAP wierny, ale ranking cech niestabilny (0,46; grupy stabilne); LIME mało wierny; surogat odtwarza 43-60% odrzuceń |
| Kalibracja | dobra przed δ; δ = −1,486 | dobra przed δ; δ = −1,488 |
| Decyzje (próg z valid) | zysk 47,5 (portfel 4%), 16,5 (test bez wag) | zysk 48,4 (portfel 4%), 21,0 (test bez wag); przy równej akceptacji - podobne |

**Czy poprawa jakości black-boxa uzasadnia utratę bezpośredniej interpretowalności? Nie.** Przewaga XGBoost (+0,02 Gini na teście,
+0,05 na valid) jest umiarkowana i na teście mieści się w niepewności statystycznej; oba modele opierają się na tych samych obszarach
(z wskaźnikami branżowymi na czele), model jest prawie addytywny (słabe interakcje, H < 0,05), a wyjaśnienia black-boxa na poziomie
cech są mniej stabilne i - w przypadku LIME - mało wierne. **Rekomendacja:** karta scoringowa jako model ratingowy, XGBoost jako
*challenger* do monitoringu i wskazywania firm o rozbieżnych ocenach.

**Główne ograniczenia:**
- wskaźniki branżowe `wsk_index_*` wymagają aktualizacji dla każdego okresu, a firma nie ma na nie wpływu (decyzja i rating zależą
  częściowo od kondycji branży);
- selekcja cech karty (IV, filtr korelacji) liczona na całym train - wybór liczby cech w CV jest więc nieco optymistyczny;
- brak ograniczeń monotonicznych w XGBoost (naruszenia do 26% zmienności PDP dla dwóch cech);
- ALE bez pasma bootstrap - niepewność krzywych nie jest oceniona;
- heurystyczny podział grupowy (brak identyfikatora firmy), brak ekspozycji i daty oceny;
- kotwica 4% przy DR próby 14,8% - PD po δ jest skalibrowane tylko dla portfela różniącego się udziałem defaultów;
- mało defaultów: 89 w valid i 89 w test - szerokie przedziały ufności, niepewne wnioski w podgrupach.
