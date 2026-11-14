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
| Cechy | 8: wskaźnik branżowy nowych firm (`wsk_index_nowe`), podatek dochodowy, kapitał własny, ogólne zadłużenie, zysk / zobowiązania, stopa zysku ze sprzedaży, wskaźnik sytuacji finansowej, pozostałe przychody operacyjne | 195 (179 faktycznie używanych) |
| Złożoność | binning monotoniczny WoE (37 przedziałów), regresja logistyczna, 9 parametrów | 800 drzew, głębokość ≤ 5, `learning_rate` 0,01, `subsample` 0,6, `colsample_bytree` 0,5, `min_child_weight` 5, λ = 5, γ = 2 |
| Strojenie | liczba cech wybrana w CV (najwyższe AUC przy k = 8) | losowe przeszukiwanie 40 konfiguracji, CV, kryterium log-loss |
| Monotoniczność | wymuszona binningiem; wszystkie β < 0 (zgodne z ekonomią); 5 z 8 β istotnych (p < 0,05) | bez ograniczeń; brak lub małe naruszenia dla głównych cech, do 28% zmienności PDP dla `wsk_sytuacji_fin` i `pozostale_przychody_oper`, 83% dla wieku firmy (`wsk_liczba_dni_istnienia`) |

**Wybór modelu końcowego:** karta scoringowa - przewaga XGBoost (+0,021 Gini na teście) jest umiarkowana i statystycznie niepewna,
model jest prawie addytywny (słabe interakcje), a karta jest bezpośrednio interpretowalna i ma stabilniejsze wyjaśnienia.
XGBoost pozostaje modelem kontrolnym (challenger).

## Jakość (test, 599 firm, 89 defaultów)
| | Karta | XGBoost |
|---|---|---|
| AUC (95% CI) | 0,781 [0,724; 0,833] | 0,792 [0,737; 0,842] |
| Gini | 0,562 | 0,583 |
| KS | 0,440 | 0,457 |
| PR-AUC | 0,430 | 0,458 |
| log-loss / Brier (przed δ) | 0,350 / 0,105 | 0,347 / 0,104 |
| Gini w CV (średnia ± odch. std.) | 0,546 ± 0,105 | 0,581 ± 0,083 |

Różnica Gini (XGBoost − karta), 95% przedział bootstrap (sparowany): test +0,021 [−0,048; +0,089], valid +0,072 [+0,019; +0,120].

## Kalibracja
- **Przed kalibracją (valid):** średnie PD 13,5% / 12,8% przy DR 14,8%; CITL a = 0,13 / 0,21; slope 1,06 / 1,04.
- **Kształt:** Platt, izotoniczna i beta porównane w 5-krotnej CV wewnątrz valid. Najniższy log-loss: bez kalibracji
  (Platt i beta gorsze o 0,003-0,005, izotoniczna o 0,03-0,06 - przeuczona przy 89 defaultach) → kształt bez zmian.
- **Dostosowanie do 4%:** logit(q) = logit(p) + δ, δ wyznaczone na valid przeważonym do DR 4%: −1,280 (karta), −1,218 (XGBoost). Średnie PD na teście przeważonym do DR 4%: 4,5% / 4,6%.
- **Po dostosowaniu (test, DR 14,9%):** AUC bez zmian; CITL a = 1,30, Brier 0,121, log-loss 0,420 (karta) - kalibracja względem
  próby pogarsza się z założenia. Na teście przeważonym do DR = 4%: CITL a = −0,13, slope 0,97, Brier 0,036 - PD jest skalibrowane
  dla portfela docelowego, jeśli różni się on od próby tylko udziałem defaultów.

## Skala ratingowa, backtest i próg
- Skala z Lab 3: AAA < 1%, AA 1-2%, A 2-5%, BBB 5-8%, BB 8-12%, B 12-20%, CCC ≥ 20%; D - stwierdzony default.
- Rozkład (karta, test przeważony do DR 4%): AAA 23,5%, AA 20,9%, A 25,9%, BBB 11,7%, BB 8,6%, B 7,4%, CCC 1,9%; HHI 0,19. DR rośnie monotonicznie (0,8% → 32,3%).
- Backtest karty (test Jeffreysa) na teście przeważonym do DR 4% z PD po δ: wszystkie 7 klas zielone; na próbie bez wag z PD bez δ
  również 7 zielonych. XGBoost: na portfelu 4% 7 zielonych, ale DR nie rośnie monotonicznie (A 4,6%, BBB 3,7%, BB 3,3%).
- **Próg:** marża 10%, LGD 45%, ekspozycja jednakowa (brak danych o EAD); p* = 18,2%. Próg z funkcji zysku wyznaczony na valid
  przeważonym do DR 4%: karta 15,0%, XGBoost 20,5% - po obu stronach p*. Na teście przeważonym do DR 4%: akceptacja 93,9% / 96,5%,
  DR wśród przyjętych 2,8% / 2,9%, zysk 47,5 / 48,6 wobec 46,7 przy przyjęciu wszystkich. Decyzje modeli zgadzają się dla 92,3% firm
  (46 firm z różną decyzją), ten sam rating dla 45,9%.

## Interpretowalność
- **Globalnie:** ryzyko rośnie z wskaźnikami branżowymi (napływ nowych firm, skreślenia i zawieszenia w branży) i z zadłużeniem
  ogólnym (skokowo powyżej ok. 0,85), a maleje z zyskiem (podatek dochodowy), kapitałem własnym, rentownością sprzedaży i pozostałymi
  przychodami operacyjnymi. Dwie najważniejsze cechy XGBoost według SHAP to wskaźniki branżowe; 4 z 5 pierwszych są w karcie.
- **Ważność grup skorelowanych cech (Lab 6):** permutacja grupowa i drop-column pokazują, że np. grupa zadłużenia jest dla XGBoost
  ważna (spadek AUC 0,022), choć pojedyncze permutacje tego nie widzą; informację większości grup model potrafi zastąpić innymi cechami.
- **Interakcje (Lab 6):** statystyka H Friedmana na dwóch próbach z parą kontrolną - interakcje powtarzalne, ale słabe (H ≤ 0,06, czyli H² < 0,5% zmienności).
- **Global surrogate (Lab 6):** drzewo o głębokości 3 odtwarza log-szanse XGBoost z R² 0,63, ale tylko 36% jego odrzuceń - służy
  do komunikacji reguł, nie do decyzji.
- **Lokalnie:** karta - dokładna dekompozycja wyniku na punkty cech; XGBoost - SHAP (wariant interventional, tło 300 firm z train,
  addytywność z błędem 5·10⁻⁶) i LIME (wkład = współczynnik × standaryzowana wartość cechy, porównywalny z SHAP). Dla decyzji
  wyjaśniamy model końcowy, czyli kartę: wkłady SHAP karty z wartością bazową φ₀ + δ (firma graniczna przy progu operacyjnym 15,0%:
  PD 14,99%, rating B; największy wkład ma `wsk_index_nowe`, dalej zerowy podatek dochodowy i niski kapitał własny).
- **Wierność:** usunięcie najważniejszej cechy według SHAP zmienia log-szanse o 0,65 wobec 0,020 dla losowej;
  LIME ma lokalne R² ≈ 0,49, średni błąd ≈ 3,9 pp i błąd w punkcie ≈ 8,6 pp PD - jego wyjaśnienia traktujemy jako pomocnicze.
- **Stabilność:** LIME stabilny dopiero przy 20 000 próbek (Jaccard top-5 0,80-1,0; przy 5 000: 0,56-0,83); ranking pojedynczych cech SHAP XGBoost zmienia się
  między foldami (Jaccard top-5 0,58 - wybór skorelowanych cech), ranking grup jest stabilniejszy (Jaccard top-3 0,70);
  karta - Jaccard top-3 0,60, znaki β stabilne w 39 z 40 przypadków.
- **Kontrfakty:** dla firmy BB (w obu modelach, PD 10,0% / 8,45%) najtańsza poprawa do BBB to wyższy zysk względem zobowiązań -
  w XGBoost przesunięcie tej grupy wskaźników o 10 percentyli (PD 7,7%; zysk brutto ok. 20 tys. zł, podatek rośnie razem z nim),
  w karcie przejście `wsk_zysk_zobowiazania` do wyższego przedziału (+1,8 pkt przy brakujących 1,7 pkt); wskaźniki branżowe
  traktujemy jako niezmienne, a podatek - jako skutek zysku, nie dźwignię.

## Ograniczenia i ryzyka
- Mała liczba defaultów w valid i test (po 89) - szerokie przedziały ufności, niepewne wnioski w podgrupach.
- Wskaźniki branżowe `wsk_index_*` trzeba aktualizować dla każdego okresu; firma nie ma na nie wpływu, a mają duży wpływ na PD.
- Selekcja cech karty (IV, filtr korelacji) liczona na całym train - wybór liczby cech w CV jest nieco optymistyczny.
- Brak ograniczeń monotonicznych w XGBoost; ALE bez pasma bootstrap (niepewność krzywych nieoceniona).
- Kotwica 4% przy DR próby 14,8%: PD po δ zakłada, że portfel docelowy różni się od próby tylko udziałem defaultów.
- Podział grupowy oparty na heurystyce (brak identyfikatora firmy); dane prawdopodobnie z kilku okresów, brak daty - nie da się
  ocenić stabilności w czasie.
- Brak ekspozycji - ekonomia progu na jednostkę ekspozycji.
- Segmenty: firmy mikro - PD zawyżone (DR / PD ≈ 0,76-0,80); forma własności F216 i sekcja C - PD zaniżone (DR / PD ≈ 1,7-2,0),
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
foldami nawet bez zmiany populacji (Jaccard top-5 0,58), a ranking grup jest stabilniejszy (Jaccard top-3 0,70). PSI między train a test: < 0,05 dla wyniku
i wszystkich cech karty.

## Użycie AI (regulamin pkt 10.3)
Do przygotowania kodu, analizy wyników i dokumentacji użyto asystenta AI (Claude, Anthropic): streszczenie metod z laboratoriów,
napisanie kodu notebooka na ich wzór, przygotowanie raportu, Model Card i prezentacji. Aktualizacja metod według Lab 6 i wykładów
(ważność grup cech, statystyka H, global surrogate, SHAP modelu końcowego) również była wspierana tym narzędziem. Wszystkie wyniki
pochodzą z uruchomienia notebooka; zespół zweryfikował decyzje modelowe i interpretacje. Fragmenty kodu przeniesione z laboratoriów
są oznaczone w notebooku komentarzem `# z Lab N` (regulamin pkt 10.5).
