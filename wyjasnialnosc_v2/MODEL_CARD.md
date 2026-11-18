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
- 3 000 obserwacji, 445 defaultów (DR 14,8%), 222 cechy: pozycje bilansu, RZiS, przepływów, wskaźniki finansowe, 3 wskaźniki branżowe
  `wsk_index_*`, forma własności, PKD, schemat sprawozdania.
- **Jakość danych:** ±inf w 3 wskaźnikach (np. wskutek dzielenia przez zero) → braki; 7 cech z > 50% braków związanych z przepływami
  pieniężnymi, 10 prawie
  stałych i 8 zduplikowanych kolumn pominięto w modelowaniu (decyzje na train; kategorie mogą się nakładać). Brak identycznych wierszy.
  `schemat_wsk_bilans` → cecha
  `wielkosc` (mikro / mała / pozostałe); `schemat_wsk_rzis` pominięty (zgodny z bilansowym w 97,6% obserwacji).
  Do modelowania: 192 cechy liczbowe + 3 kategoryczne.
- **Wskaźniki branżowe** `wsk_index_skr_zaw`, `wsk_index_nowe`, `wsk_index_aktywne` (skreślenia / zawieszenia, nowe i aktywne firmy
  w branży; ok. 3 wartości na kod PKD): spoza słownika danych. `wsk_index_nowe` i `wsk_index_skr_zaw` mają najwyższe AUC
  jednowymiarowe na train (odpowiednio 0,718 i 0,708). Prowadzący potwierdził, że wskaźniki nie zawierają informacji z przyszłości.
  Dodanie trzech wskaźników podnosi Gini prostego logitu na valid z 0,397 do 0,586.
- **Powtarzające się firmy:** brak identyfikatora; heurystyka (to samo PKD, forma własności i kapitał, wiek różny o pełne lata)
  znalazła 61 potencjalnych par wobec średnio 35 po przemieszaniu wieku → podział grupowy (51 grup wieloelementowych).

## Podział i walidacja
- Stratyfikowany i grupowy (`StratifiedGroupKFold`): **train 60% / valid 20% / test 20%** (1 801 / 600 / 599 obserwacji,
  odpowiednio 267 / 89 / 89 defaultów). Brak daty oceny i identyfikatora firmy uniemożliwia podział czasowy i grupowanie według ID;
  grupy wyznaczono heurystycznie.
- 5-krotna stratyfikowana grupowa CV na train: wybór liczby cech karty, strojenie XGBoost, stabilność metryk i wyjaśnień.
- Binning i WoE karty oraz kodowanie kategorii XGBoost są dopasowywane osobno na części treningowej każdego foldu.
  Lista cech do modelowania oraz ranking IV i filtr korelacji karty powstają na całym train przed CV; wyniki CV mogą być
  optymistyczne. Kalibrację, δ i próg dobieramy na valid, a test służy do oceny.
- Podział zapisany w `wyniki/podzial.csv`, zestawy cech w `wyniki/cechy_v1.json` i `wyniki/cechy_karty.json`.

## Modele
| | Karta scoringowa (model końcowy) | XGBoost (challenger) |
|---|---|---|
| Cechy | 8: wskaźnik branżowy nowych firm (`wsk_index_nowe`), podatek dochodowy, kapitał własny, ogólne zadłużenie, zysk / zobowiązania, stopa zysku ze sprzedaży, wskaźnik sytuacji finansowej, pozostałe przychody operacyjne | 195 wejściowych, 170 faktycznie użytych |
| Złożoność | binning i WoE (37 przedziałów), regresja logistyczna, 9 parametrów | 800 drzew, 7 583 liście, głębokość ≤ 5, `learning_rate` 0,01, `subsample` 0,6, `colsample_bytree` 0,5, `min_child_weight` 5, λ = 5, γ = 2 |
| Strojenie | liczba cech wybrana w CV (najwyższe AUC przy k = 8) | losowe przeszukiwanie 40 konfiguracji, CV, kryterium log-loss |
| Monotoniczność | binning wymusza monotoniczność DR na train w zwykłych przedziałach liczbowych, lecz nie gwarantuje monotonicznego WoE; wszystkie β < 0 zgodnie z konwencją WoE i procedurą selekcji | bez ograniczeń; udział naruszeń oczekiwanego kierunku PDP to m.in. 24,6% dla `wsk_sytuacji_fin` i 27,6% dla `pozostale_przychody_oper` |

**Wybór modelu końcowego:** rekomendujemy kartę scoringową jako prostszy model podstawowy. Przewaga XGBoost na teście
(+0,021 Gini) jest niepewna, a wynik karty można bezpośrednio wyjaśnić punktami. XGBoost pozostaje modelem kontrolnym
(challenger). Przed zastosowaniem potrzebna jest walidacja kalibracji i decyzji na danych rzeczywistego portfela.

## Jakość (nieważony test, 599 obserwacji, 89 defaultów)
| | Karta | XGBoost |
|---|---|---|
| AUC (95% CI) | 0,781 [0,724; 0,833] | 0,792 [0,737; 0,842] |
| Gini | 0,562 | 0,583 |
| KS (95% CI) | 0,440 [0,374; 0,559] | 0,457 [0,382; 0,553] |
| PR-AUC (95% CI) | 0,430 [0,336; 0,552] | 0,458 [0,364; 0,578] |
| log-loss (95% CI) / Brier, przed δ | 0,350 [0,303; 0,401] / 0,105 | 0,347 [0,295; 0,406] / 0,104 |
| Gini w CV na train (średnia ± odch. std.) | 0,546 ± 0,105 | 0,581 ± 0,083 |

Różnica Gini (XGBoost − karta), 95% przedział bootstrap (sparowany): test +0,021 [−0,048; +0,089],
valid +0,072 [+0,019; +0,120]. Przedziały dotyczą wyuczonych modeli, bez ponownego uczenia w próbkach bootstrap.

## Kalibracja
- **Przed kalibracją (valid):** średnie PD 13,46% / 12,82% przy DR 14,83%; CITL a = 0,135 / 0,213;
  slope 1,06 / 1,04 (karta / XGBoost).
- **Kształt:** Platt, izotoniczna i beta porównane w 5-krotnej CV wewnątrz valid. Najniższy log-loss i Brier
  uzyskano bez kalibracji kształtu dla obu modeli; nie oznacza to idealnej kalibracji.
- **Dostosowanie do 4%:** logit(q) = logit(p) + δ; δ = −1,486 / −1,479 dobieramy tak, by nieważone średnie PD na valid
  wynosiło 4%. Po przesunięciu średnie PD na nieważonym teście wynosi 4,44% / 4,57%.
- **Nieważony test (DR 14,86%):** po δ Brier rośnie z 0,105 do 0,124 dla karty i z 0,104 do 0,121 dla XGBoost;
  porównujemy PD dostosowane do 4% z próbą o wyższym DR. Ranking Gini pozostaje bez zmian.
- **Test przeważony do DR 4%:** po δ średnie PD wynosi 3,74% / 3,68%, CITL 0,074 / 0,095, a slope 0,97 / 0,85.
  Jest to symulacja zależna od założenia o rozkładach cech portfela, a nie dowód kalibracji rzeczywistego portfela.

## Skala ratingowa, backtest i próg
- Skala z Lab 3: AAA < 1%, AA 1-2%, A 2-5%, BBB 5-8%, BB 8-12%, B 12-20%, CCC ≥ 20%; D - stwierdzony default.
- Rozkład (karta, nieważony test): AAA 27,5%, AA 20,0%, A 25,4%, BBB 8,5%, BB 9,0%, B 6,8%, CCC 2,7%; HHI 0,201.
  DR rośnie między klasami karty od 4,8% do 68,8%. W XGBoost występuje jedno odwrócenie: DR klasy A to 18,7%, a BBB 14,3%.
- **Backtest na nieważonym teście:** według Jeffreysa oba modele mają przed δ 6 zielonych i 1 żółte światło,
  a po δ 6 czerwonych i 1 żółte (BBB). Porównanie dotyczy tych samych klas wyznaczonych po δ. Po przesunięciu
  PD odpowiada założonemu poziomowi 4%, podczas gdy DR testu wynosi 14,9%; sam wynik nie potwierdza kalibracji
  dla innego portfela. Wyniki testu dwumianowego nie zawsze wskazują ten sam poziom.
- **Próg:** marża 10%, LGD 45%, ekspozycja jednakowa (brak danych o EAD); p* = 18,2%. Próg z funkcji zysku wyznaczony na valid
  przeważonym do DR 4%: 16,5% dla obu modeli. Na teście przeważonym do DR 4% akceptacja wynosi 97,9% / 96,5%,
  DR wśród przyjętych 3,4% / 2,9%, oczekiwana strata 8,66 / 7,48, a zysk 47,52 / 48,64 wobec 46,72 przy przyjęciu
  wszystkich (EAD = 1, suma wag = 599). To symulacja, nie gwarancja wyników w rzeczywistym portfelu.
  Przy tych progach decyzje są zgodne dla 96,2% obserwacji na nieważonym teście.

## Interpretowalność
- **Globalnie:** dwie najważniejsze cechy XGBoost według SHAP i permutacji to `wsk_index_nowe` i `wsk_index_skr_zaw`;
  trzy kolejne według SHAP są w karcie. Najważniejsza cecha karty, `wsk_index_nowe`, stanowi ok. 36% sumy średnich |SHAP|.
  PDP dla wybranych cech opisuje kierunek reakcji modelu, nie zależność przyczynową.
- **Grupy skorelowanych cech (Lab 6):** na valid permutacja grupy zadłużenia daje spadek AUC o 0,0218, a suma ważności
  pojedynczych cech tej grupy wynosi 0,0022. Po ponownym uczeniu bez grupy spadek AUC wynosi 0,0001 w tej próbie.
- **Interakcje (Lab 6):** dla czterech wybranych par H wynosi 0,043–0,063 na dwóch losowaniach, które mają 39 wspólnych firm.
  Wynik dotyczy badanych par i prób, nie wszystkich interakcji modelu.
- **Global surrogate (Lab 6):** drzewo głębokości 3 przybliża log-szanse XGBoost z R² 0,631 i odtwarza 14 z 39 jego odrzuceń
  (35,9%). Jest uproszczonym opisem, nie zamiennikiem modelu.
- **Lokalnie:** karta daje dokładną dekompozycję wyniku na punkty cech; XGBoost wyjaśniamy m.in. SHAP i LIME.
  Do SHAP `interventional` przekazano 300 obserwacji train, z których SHAP domyślnie wybiera 100 jako tło.
  Maksymalny błąd addytywności na teście wynosi 5,2·10⁻⁶ log-szans. Po δ wkłady SHAP pozostają takie same,
  a wartość bazowa zmienia się o δ.
- **Wierność:** zastąpienie najważniejszej cechy według SHAP zmienia log-szanse średnio o 0,648 wobec 0,020 dla losowej cechy.
  W badaniu LIME mediana R² na jego próbkach wynosi 0,47, a mediana błędu w punkcie 8,56 pp PD; wyjaśnienia są pomocnicze.
- **Stabilność:** ranking SHAP XGBoost między foldami ma średni Jaccard top-5 0,58, a ranking grup top-3 0,70;
  dla karty top-3 wynosi 0,60. LIME zależy od losowania: więcej próbek często poprawia stabilność, ale nie zawsze.
- **Kontrfakty:** dla wybranej firmy o ratingu BB badana zmiana dźwigni zadłużenia obniża PD XGBoost z 10,07% do 7,69%.
  Wskaźniki branżowe pozostają niezmienne; analiza nie dowodzi wykonalności zmian ani związku przyczynowego.

## Założenia, ograniczenia i ryzyka
- Brak daty oceny i identyfikatora firmy wymaga podziału grupowego opartego na heurystyce; stabilność w czasie
  wymaga oceny na danych z kolejnego okresu.
- Selekcja cech karty na całym train przed CV oraz wybór liczby cech i parametrów z użyciem CV mogą powodować optymizm wyników CV.
- Wskaźniki branżowe trzeba aktualizować dla każdego okresu; nie są bezpośrednio kontrolowane przez firmę.
- Symulacja portfela 4% zależy od założenia o rozkładach cech i przyjmuje jednakową ekspozycję. Kalibrację i decyzje
  należy zweryfikować na danych portfela docelowego.
- Po 89 defaultów w valid i teście ogranicza pewność wniosków, szczególnie w małych podgrupach. Na nieważonym teście,
  przed δ, dla firm mikro DR wynosi 13,2%, a średnie PD 17,5% / 16,6%; w sekcji C DR wynosi 17,1%, a PD 10,1% / 9,2%;
  dla formy własności F216 DR to 22,4%, a PD 12,2% / 11,3% (karta / XGBoost). To sygnały do dalszej analizy,
  nie dowód trwałego błędu ani podstawa do automatycznej korekty segmentowej.
- XGBoost nie ma ograniczeń monotonicznych; dla krzywych ALE nie oszacowano pasm niepewności. Wyjaśnienia opisują
  model, nie związki przyczynowe; skorelowane cechy utrudniają interpretację ich osobnej ważności.
- Zbiór dotyczy firm i nie obejmuje cech osób fizycznych, takich jak płeć czy wiek; audyt dostępnych segmentów firm
  należy powtarzać.

## Monitoring
| Obszar | Miara | Częstotliwość | Próg alarmowy |
|---|---|---|---|
| populacja | PSI wyniku i cech karty / top-10 SHAP | kwartalnie | > 0,10 obserwuj, > 0,25 przegląd modelu |
| dyskryminacja | Gini, KS dla obserwacji z dojrzałym wynikiem | rocznie (po 12 mies.) | spadek Gini > 0,10 względem walidacji |
| kalibracja | średnie PD vs DR, Brier, ECE, test Jeffreysa per klasa | rocznie (po 12 mies.) | czerwone światło w ≥ 2 klasach; uwzględnić liczebności i niepewność |
| wyjaśnienia | średnie \|SHAP\| grup tematycznych, kontrola wersji i znaków β karty | kwartalnie | względna zmiana ważności grupy > 30% albo nieoczekiwana zmiana parametrów |
| dane branżowe | aktualność i PSI `wsk_index_*` | przy każdym nowym okresie | brak aktualizacji na bieżący okres albo PSI > 0,25 |
| decyzje | stopa akceptacji; DR wśród przyjętych po dojrzeniu obserwacji | miesięcznie; DR po 12 mies. | odchylenie > 5 pp od planu |

Monitoring wyjaśnień uwzględnia grupy tematyczne i zmienność rankingów między foldami. Parametry β wdrożonej karty pozostają
stałe do zmiany modelu; ich znaki można oceniać po ponownym dopasowaniu. Progi są propozycją operacyjną do dostosowania
do liczebności i danych portfela. PSI badanych wyników i cech na teście jest poniżej 0,05; najwyższe wynosi 0,0402 dla sekcji PKD.
Nie dowodzi to stabilności po wdrożeniu.

## Użycie AI (regulamin pkt 10.3)
Do przygotowania kodu, wstępnej analizy wyników i dokumentacji użyto asystenta AI (Claude, Anthropic): streszczenie metod z laboratoriów,
napisanie kodu notebooka na ich wzór, przygotowanie raportu, Model Card i prezentacji. Aktualizacja metod według Lab 6 i wykładów
(ważność grup cech, statystyka H, global surrogate, SHAP modelu końcowego) również była wspierana tym narzędziem. Wszystkie wyniki
pochodzą z uruchomienia notebooka. Fragmenty kodu przeniesione z laboratoriów są oznaczone w notebooku komentarzem `# z Lab N` (regulamin pkt 10.5).
