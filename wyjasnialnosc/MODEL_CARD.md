# Model Card - model ratingowy PD dla spółek z o.o. (zbiór 13)

## Cel i zakres
- **Przeznaczenie:** ocena ryzyka niewypłacalności (PD) firm na podstawie sprawozdania finansowego, danych rejestrowych
  i wskaźników branżowych; przypisanie klasy ratingowej (AAA ... CCC) i wsparcie decyzji o przyjęciu klienta.
- **Populacja:** spółki z ograniczoną odpowiedzialnością (wszystkie obserwacje mają tę samą formę prawną), różne sekcje PKD,
  jednostki mikro, małe i pozostałe.
- **Horyzont / zdarzenie:** flaga `default` - zdarzenie ryzyka w horyzoncie przyjętym przez dostawcę danych.
- **Nie wolno używać:** dla innych form prawnych, osób fizycznych, firm bez sprawozdania finansowego; z nieaktualnymi (z innego
  okresu) wskaźnikami branżowymi `wsk_index_*`; do decyzji automatycznych bez przeglądu analityka dla firm z dużą różnicą ocen
  dwóch modeli; do wyceny PD w okresie kryzysu bez przeglądu kotwicy 4%.

## Dane
- 3 000 firm, 445 defaultów (DR 14,8%), 222 cechy: pozycje bilansu, RZiS, przepływów, wskaźniki finansowe, 3 wskaźniki branżowe
  `wsk_index_*`, forma własności, PKD, schemat sprawozdania.
- **Jakość danych:** ±inf w 3 wskaźnikach (dzielenie przez zero) → braki; 7 cech z > 50% braków (przepływy pieniężne), 10 prawie
  stałych i 8 zduplikowanych kolumn usunięto (decyzje na train). Brak duplikatów obserwacji. `schemat_wsk_bilans` → cecha
  `wielkosc` (mikro / mała / pozostałe); `schemat_wsk_rzis` pominięty (zgodny z bilansowym w 97,6% obserwacji).
  Do modelowania: 192 cechy liczbowe + 3 kategoryczne.
- **Wskaźniki branżowe** `wsk_index_skr_zaw`, `wsk_index_nowe`, `wsk_index_aktywne` (skreślenia / zawieszenia, nowe i aktywne firmy
  w branży; ok. 3 wartości na kod PKD - kilka okresów): spoza słownika danych, najwyższe AUC jednowymiarowe (0,71-0,72).
  Prowadzący potwierdził, że nie zawierają informacji z przyszłości - są używane w obu modelach. Podnoszą Gini prostego logitu
  na teście z 0,373 do 0,534.
- **Powtarzające się firmy:** brak identyfikatora; heurystyka (to samo PKD, forma własności i kapitał, wiek różny o pełne lata)
  znalazła 61 par wobec ok. 35 oczekiwanych losowo → podział grupowy (51 grup wieloelementowych).

## Podział i walidacja
- Stratyfikowany i grupowy (`StratifiedGroupKFold`): **train 60% / valid 20% / test 20%** (1 801 / 600 / 599 firm, po 267 / 89 / 89 defaultów).
- 5-krotna stratyfikowana grupowa CV na train: wybór liczby cech karty, strojenie XGBoost, stabilność metryk i wyjaśnień.
- Binning, WoE, selekcja cech, kodowanie - tylko train; kalibracja, δ do 4%, próg - tylko valid; test - wyłącznie ocena końcowa.
- Podział zapisany w `wyniki/podzial.csv`, wersje cech w `wyniki/cechy_v1.json` i `wyniki/cechy_karty.json`.

## Modele
| | Karta scoringowa (model końcowy) | XGBoost (challenger) |
|---|---|---|
| Cechy | 8: wskaźnik branżowy nowych firm (`wsk_index_nowe`), podatek dochodowy, kapitał własny, ogólne zadłużenie, zysk / zobowiązania, stopa zysku ze sprzedaży, wskaźnik sytuacji finansowej, pozostałe przychody operacyjne | 195 (158 faktycznie używanych) |
| Złożoność | binning monotoniczny WoE (37 przedziałów), regresja logistyczna, 9 parametrów | 500 drzew, głębokość ≤ 5, `learning_rate` 0,02, `subsample` 0,6, `colsample_bytree` 0,2, `min_child_weight` 20, λ = 5, γ = 0 |
| Strojenie | liczba cech wybrana w CV (najwyższe AUC przy k = 8) | losowe przeszukiwanie 40 konfiguracji, CV, kryterium log-loss |
| Monotoniczność | wymuszona binningiem; wszystkie β < 0 (zgodne z ekonomią); 5 z 8 β istotnych (p < 0,05) | bez ograniczeń; brak lub małe naruszenia dla głównych cech, do 26% zmienności PDP dla `wsk_sytuacji_fin` i `pozostale_przychody_oper` |

**Wybór modelu końcowego:** karta scoringowa - przewaga XGBoost (+0,024 Gini na teście) jest umiarkowana i statystycznie niepewna,
model jest prawie addytywny (słabe interakcje), a karta jest bezpośrednio interpretowalna i ma stabilniejsze wyjaśnienia.
XGBoost pozostaje modelem kontrolnym (challenger).

## Jakość (test, 599 firm, 89 defaultów)
| | Karta | XGBoost |
|---|---|---|
| AUC (95% CI) | 0,781 [0,724; 0,833] | 0,793 [0,740; 0,842] |
| Gini | 0,562 | 0,586 |
| KS | 0,440 | 0,462 |
| PR-AUC | 0,430 | 0,468 |
| log-loss / Brier (przed δ) | 0,350 / 0,105 | 0,341 / 0,102 |
| Gini w CV (średnia ± odch. std.) | 0,546 ± 0,105 | 0,581 ± 0,091 |

Różnica Gini (XGBoost − karta), 95% przedział bootstrap (sparowany): test +0,024 [−0,032; +0,083], valid +0,055 [+0,008; +0,099].

## Kalibracja
- **Przed kalibracją (valid):** średnie PD 13,5% / 13,4% przy DR 14,8%; CITL a = 0,13 / 0,14; slope 1,06 / 1,14.
- **Kształt:** Platt, izotoniczna i beta porównane w 5-krotnej CV wewnątrz valid. Najniższy log-loss: bez kalibracji
  (Platt i beta gorsze o 0,003-0,005, izotoniczna o 0,05-0,06 - przeuczona przy 89 defaultach) → kształt bez zmian.
- **Dostosowanie do 4%:** logit(q) = logit(p) + δ, δ wyznaczone na valid: −1,486 (karta), −1,488 (XGBoost). Średnie PD na teście 4,4% / 4,6%.
- **Po dostosowaniu (test, DR 14,9%):** AUC bez zmian; CITL a = 1,50, Brier 0,124, log-loss 0,441 (karta) - kalibracja względem
  próby pogarsza się z założenia. Na teście przeważonym do DR = 4%: CITL a = 0,07, slope 0,97, Brier 0,036 - PD jest skalibrowane
  dla portfela docelowego, jeśli różni się on od próby tylko udziałem defaultów.

## Skala ratingowa, backtest i próg
- Skala z Lab 3: AAA < 1%, AA 1-2%, A 2-5%, BBB 5-8%, BB 8-12%, B 12-20%, CCC ≥ 20%; D - stwierdzony default.
- Rozkład (karta, test): AAA 27,5%, AA 20,0%, A 25,4%, BBB 8,5%, BB 9,0%, B 6,8%, CCC 2,7%; HHI 0,20. DR rośnie monotonicznie (4,8% → 68,8%).
- Backtest karty (test Jeffreysa i test dwumianowy): przed δ 6 klas zielonych i 1 żółta; po δ 6 z 7 czerwonych - skutek kotwicy 4% przy DR
  próby 14,9%, nie błąd kształtu (PD through-the-cycle).
- **Próg:** marża 10%, LGD 45%, ekspozycja jednakowa (brak danych o EAD); p* = 18,2%. Próg z funkcji zysku wyznaczony na valid
  przeważonym do DR 4%: karta 16,5%, XGBoost 16,0% - blisko p*. Na teście przeważonym do DR 4%: akceptacja 97,9% / 96,6%,
  DR wśród przyjętych 3,4% / 3,0%, zysk 47,5 / 48,4 wobec 46,7 przy przyjęciu wszystkich. Decyzje modeli zgadzają się dla 96,8% firm.

## Interpretowalność
- **Globalnie:** ryzyko rośnie z wskaźnikami branżowymi (napływ nowych firm, skreślenia i zawieszenia w branży) i z zadłużeniem
  ogólnym (skokowo powyżej ok. 0,85), a maleje z zyskiem (podatek dochodowy), kapitałem własnym, rentownością sprzedaży i pozostałymi
  przychodami operacyjnymi. Dwie najważniejsze cechy XGBoost według SHAP to wskaźniki branżowe; 4 z 5 pierwszych są w karcie.
- **Ważność grup skorelowanych cech (Lab 6):** permutacja grupowa i drop-column pokazują, że np. grupa zadłużenia jest dla XGBoost
  ważna (spadek AUC 0,034), choć pojedyncze permutacje tego nie widzą; informację większości grup model potrafi zastąpić innymi cechami.
- **Interakcje (Lab 6):** statystyka H Friedmana na dwóch próbach z parą kontrolną - interakcje powtarzalne, ale słabe (H < 0,05).
- **Global surrogate (Lab 6):** drzewo o głębokości 3 odtwarza log-szanse XGBoost z R² 0,67, ale tylko 60% jego odrzuceń - służy
  do komunikacji reguł, nie do decyzji.
- **Lokalnie:** karta - dokładna dekompozycja wyniku na punkty cech; XGBoost - SHAP (wariant interventional, tło 300 firm z train,
  addytywność z błędem 4·10⁻⁶) i LIME. Dla decyzji wyjaśniamy model końcowy: te same wkłady SHAP z wartością bazową φ₀ + δ
  (firma graniczna przy progu operacyjnym).
- **Wierność:** usunięcie najważniejszej cechy według SHAP zmienia log-szanse o 0,56 wobec 0,015 dla losowej;
  LIME ma lokalne R² ≈ 0,56 i błąd w punkcie ≈ 7 pp PD - jego wyjaśnienia traktujemy jako pomocnicze.
- **Stabilność:** LIME stabilny przy ≥ 5 000 próbek (Jaccard top-5 0,70-1,0); ranking pojedynczych cech SHAP XGBoost zmienia się
  między foldami (Jaccard top-5 0,46 - wybór skorelowanych cech), ranking grup jest stabilny (Jaccard top-3 1,00);
  karta - Jaccard top-3 0,60, znaki β stabilne w 39 z 40 przypadków.
- **Kontrfakty:** dla firmy BB (w obu modelach) poprawa do BBB wymaga np. obniżenia zadłużenia - wskazują to oba modele;
  wskaźniki branżowe traktujemy jako niezmienne.

## Ograniczenia i ryzyka
- Mała liczba defaultów w valid i test (po 89) - szerokie przedziały ufności, niepewne wnioski w podgrupach.
- Wskaźniki branżowe `wsk_index_*` trzeba aktualizować dla każdego okresu; firma nie ma na nie wpływu, a mają duży wpływ na PD.
- Selekcja cech karty (IV, filtr korelacji) liczona na całym train - wybór liczby cech w CV jest nieco optymistyczny.
- Brak ograniczeń monotonicznych w XGBoost; ALE bez pasma bootstrap (niepewność krzywych nieoceniona).
- Kotwica 4% przy DR próby 14,8%: PD po δ zakłada, że portfel docelowy różni się od próby tylko udziałem defaultów.
- Podział grupowy oparty na heurystyce (brak identyfikatora firmy); dane prawdopodobnie z kilku okresów, brak daty - nie da się
  ocenić stabilności w czasie.
- Brak ekspozycji - ekonomia progu na jednostkę ekspozycji.
- Segmenty: firmy mikro - PD zawyżone (DR / PD ≈ 0,76-0,77); forma własności F216 i sekcja C - PD zaniżone (DR / PD ≈ 1,7-2,0),
  przy 13-14 defaultach w grupie.
- Wyjaśnienia (SHAP, LIME, PDP, kontrfakty) opisują model, a nie zależności przyczynowe; przy skorelowanych cechach podział wkładu jest umowny.
- Brak cech osób fizycznych (płeć, wiek) - AI Act zał. III pkt 5 lit. b nie dotyczy bezpośrednio, ale audyt segmentów firm należy powtarzać.

## Monitoring
| Obszar | Miara | Częstotliwość | Próg alarmowy |
|---|---|---|---|
| populacja | PSI wyniku, CSI cech karty, `wsk_index_*` i top-10 SHAP | kwartalnie | > 0,10 obserwuj, > 0,25 przegląd |
| dane branżowe | aktualność `wsk_index_*` (okres zgodny z datą oceny) | przy każdej aktualizacji danych | wskaźniki z innego okresu |
| dyskryminacja | Gini, KS na nowych defaultach | rocznie | spadek Gini > 0,10 |
| kalibracja | średnie PD vs DR, Brier, ECE, Jeffreys i test dwumianowy per klasa | rocznie | czerwone światło w ≥ 2 klasach |
| wyjaśnienia | względna zmiana średniego \|SHAP\| grup tematycznych względem walidacji, znaki β karty | kwartalnie | zmiana > 30% albo zmiana znaku β |
| decyzje | stopa akceptacji, DR wśród przyjętych | miesięcznie | odchylenie > 5 pp |

Monitoring wyjaśnień prowadzimy na poziomie grup, a nie Jaccard top-5 pojedynczych cech, bo ranking cech SHAP zmienia się między
foldami nawet bez zmiany populacji (Jaccard top-5 0,46), a ranking grup jest stabilny. PSI między train a test: < 0,05 dla wyniku
i wszystkich cech karty.

## Użycie AI (regulamin pkt 10.3)
Do przygotowania kodu, analizy wyników i dokumentacji użyto asystenta AI (Claude, Anthropic): streszczenie metod z laboratoriów,
napisanie kodu notebooka na ich wzór, przygotowanie raportu, Model Card i prezentacji. Aktualizacja metod według Lab 6 i wykładów
(ważność grup cech, statystyka H, global surrogate, SHAP modelu końcowego) również była wspierana tym narzędziem. Wszystkie wyniki
pochodzą z uruchomienia notebooka; zespół zweryfikował decyzje modelowe i interpretacje. Fragmenty kodu przeniesione z laboratoriów
są oznaczone w notebooku komentarzem `# z Lab N` (regulamin pkt 10.5).
