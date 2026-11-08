# Model Card - model ratingowy PD dla spółek z o.o. (zbiór 13)

## Cel i zakres
- **Przeznaczenie:** ocena ryzyka niewypłacalności (PD) firm na podstawie sprawozdania finansowego i danych rejestrowych;
  przypisanie klasy ratingowej (AAA ... CCC) i wsparcie decyzji o przyjęciu klienta.
- **Populacja:** spółki z ograniczoną odpowiedzialnością (wszystkie obserwacje mają tę samą formę prawną), różne sekcje PKD,
  jednostki mikro, małe i pozostałe.
- **Horyzont / zdarzenie:** flaga `default` - zdarzenie ryzyka w horyzoncie przyjętym przez dostawcę danych.
- **Nie wolno używać:** dla innych form prawnych, osób fizycznych, firm bez sprawozdania finansowego; do decyzji automatycznych
  bez przeglądu analityka dla firm z dużą różnicą ocen dwóch modeli; do wyceny PD w okresie kryzysu bez przeglądu kotwicy 4%.

## Dane
- 3 000 firm, 445 defaultów (DR 14,8%), 222 cechy: pozycje bilansu, RZiS, przepływów, wskaźniki finansowe, forma własności, PKD,
  schemat sprawozdania (wielkość jednostki).
- **Jakość danych:** ±inf w 3 wskaźnikach (dzielenie przez zero) → braki; 7 cech z > 50% braków (przepływy pieniężne), 10 prawie
  stałych i 8 zduplikowanych kolumn usunięto (decyzje na train). Brak duplikatów obserwacji.
- **Wykluczone cechy:** `wsk_index_skr_zaw`, `wsk_index_nowe`, `wsk_index_aktywne` - wskaźniki branżowo-okresowe spoza słownika danych,
  o nieznanym okresie pomiaru (ryzyko informacji z przyszłości); dodawały ok. 0,15-0,19 Gini prostego modelu.
- **Powtarzające się firmy:** brak identyfikatora; heurystyka (to samo PKD, forma własności i kapitał, wiek różny o pełne lata)
  znalazła 61 par wobec ok. 35 oczekiwanych losowo → podział grupowy (51 grup wieloelementowych).

## Podział i walidacja
- Stratyfikowany i grupowy (`StratifiedGroupKFold`): **train 60% / valid 20% / test 20%** (1 801 / 600 / 599 firm, po 267 / 89 / 89 defaultów).
- 5-krotna stratyfikowana grupowa CV na train: wybór liczby cech karty, strojenie XGBoost, stabilność metryk i wyjaśnień.
- Binning, WoE, selekcja cech, kodowanie - tylko train; kalibracja, δ do 4%, próg - tylko valid; test - wyłącznie ocena końcowa.
- Podział zapisany w `wyniki/podzial.csv`, wersje cech w `wyniki/cechy_v1.json` i `wyniki/cechy_karty.json`.

## Modele
| | Karta scoringowa (model końcowy) | XGBoost (model kontrolny) |
|---|---|---|
| Cechy | 7: podatek dochodowy, kapitał własny, ogólne zadłużenie, zysk / zobowiązania, stopa zysku ze sprzedaży, wskaźnik sytuacji finansowej, pozostałe przychody operacyjne | 192 (164 faktycznie używane) |
| Złożoność | binning monotoniczny WoE (32 przedziały), regresja logistyczna, 8 parametrów | 300 drzew, głębokość ≤ 4, `learning_rate` 0,01, `colsample_bytree` 0,35, γ = 2, λ = 5 |
| Strojenie | liczba cech wybrana w CV (najwyższe AUC) | losowe przeszukiwanie 40 konfiguracji, CV, kryterium log-loss |
| Monotoniczność | wymuszona binningiem; wszystkie β < 0 (zgodne z ekonomią) | bez ograniczeń; naruszenia < 4% zmienności PDP dla głównych cech |

**Wybór modelu końcowego:** karta scoringowa - przewaga XGBoost (+0,03 Gini) jest statystycznie niepewna, a karta jest
bezpośrednio interpretowalna i ma stabilniejsze wyjaśnienia.

## Jakość (test, 599 firm, 89 defaultów)
| | Karta | XGBoost |
|---|---|---|
| AUC (95% CI) | 0,730 [0,674; 0,785] | 0,743 [0,690; 0,800] |
| Gini | 0,460 | 0,486 |
| KS | 0,353 | 0,367 |
| PR-AUC | 0,331 | 0,342 |
| log-loss / Brier (przed δ) | 0,377 / 0,115 | 0,372 / 0,113 |
| Gini w CV (średnia ± odch. std.) | 0,425 ± 0,110 | 0,442 ± 0,101 |

Różnica Gini (XGBoost − karta) na teście: +0,026, 95% przedział bootstrap (sparowany): [−0,039; +0,091].

## Kalibracja
- **Przed kalibracją (valid):** średnie PD 14,4% / 14,1% przy DR 14,8%; CITL a = 0,04 / 0,06; slope 0,95 / 1,16.
- **Kształt:** Platt, izotoniczna i beta porównane w 5-krotnej CV wewnątrz valid. Najniższy log-loss: bez kalibracji
  (Platt i beta gorsze o 0,002-0,004, izotoniczna o ok. 0,03 - przeuczona przy 89 defaultach) → kształt bez zmian.
- **Dostosowanie do 4%:** logit(q) = logit(p) + δ, δ wyznaczone na valid: −1,494 (karta), −1,463 (XGBoost). Średnie PD na teście 4,0%.
- **Po dostosowaniu (test, DR 14,9%):** AUC bez zmian; CITL a = 1,54, Brier 0,131, log-loss 0,476 (karta) - kalibracja względem
  próby pogarsza się z założenia. Na teście przeważonym do DR = 4%: CITL a = 0,10, slope 0,92, Brier 0,037 - PD jest skalibrowane
  dla portfela docelowego, jeśli różni się on od próby tylko udziałem defaultów.

## Skala ratingowa, backtest i próg
- Skala z Lab 3: AAA < 1%, AA 1-2%, A 2-5%, BBB 5-8%, BB 8-12%, B 12-20%, CCC ≥ 20%; D - stwierdzony default.
- Rozkład (karta, test): AAA 16,5%, AA 21,9%, A 32,9%, BBB 19,7%, BB 3,7%, B 3,7%, CCC 1,7%; HHI 0,23. DR rośnie monotonicznie (3% → 50%).
- Backtest (test Jeffreysa): przed δ wszystkie klasy zielone; po δ większość czerwona - skutek kotwicy 4% przy DR próby 14,9%,
  nie błąd kształtu (PD through-the-cycle).
- **Próg:** marża 10%, LGD 45%, ekspozycja jednakowa (brak danych o EAD). Próg z valid na PD po δ: karta 2,5%, XGBoost 7,0%.
  Na teście: akceptacja 47% / 89%, zysk 17,5 / 20,6 wobec 11,0 przy przyjęciu wszystkich. Przy równej stopie akceptacji
  decyzje modeli zgadzają się w 87-94%.

## Interpretowalność
- **Globalnie:** ryzyko rośnie z zadłużeniem ogólnym (skokowo powyżej ok. 0,85) i maleje z zyskiem (podatek dochodowy), kapitałem
  własnym, rentownością sprzedaży i pozostałymi przychodami operacyjnymi. Trzy najważniejsze cechy XGBoost według SHAP są w karcie.
- **Lokalnie:** karta - dokładna dekompozycja wyniku na punkty cech; XGBoost - SHAP (addytywność z błędem 3·10⁻⁶) i LIME.
- **Wierność:** usunięcie najważniejszej cechy według SHAP zmienia log-szanse 50 razy bardziej niż usunięcie losowej;
  LIME ma lokalne R² ≈ 0,22 i błąd w punkcie ≈ 8 pp PD - jego wyjaśnienia traktujemy jako pomocnicze.
- **Stabilność:** LIME stabilny przy ≥ 5 000 próbek (Jaccard top-5 0,73-1,0); ranking SHAP XGBoost zmienia się między foldami
  (Jaccard top-5 0,33 - wybór skorelowanych cech), karta - Jaccard top-3 0,52, znaki β stabilne w 34 z 35 przypadków.

## Ograniczenia i ryzyka
- Mała liczba defaultów w valid i test (po 89) - szerokie przedziały ufności, niepewne wnioski w podgrupach.
- Kotwica 4% przy DR próby 14,8%: PD po δ zakłada, że portfel docelowy różni się od próby tylko udziałem defaultów.
- Wykluczenie `wsk_index_*` na podstawie zasady ostrożności - do potwierdzenia z autorem danych.
- Podział grupowy oparty na heurystyce (brak identyfikatora firmy); dane prawdopodobnie z kilku okresów, brak daty - nie da się
  ocenić stabilności w czasie.
- Brak ekspozycji - ekonomia progu na jednostkę ekspozycji.
- Segmenty: firmy mikro - PD zawyżone (DR / PD ≈ 0,75-0,79); formy własności F215 / F216 i sekcja C - PD zaniżone, przy małej liczbie defaultów.
- Wyjaśnienia (SHAP, LIME, PDP, kontrfakty) opisują model, a nie zależności przyczynowe; przy skorelowanych cechach podział wkładu jest umowny.
- Brak cech osób fizycznych (płeć, wiek) - AI Act zał. III pkt 5 lit. b nie dotyczy bezpośrednio, ale audyt segmentów firm należy powtarzać.

## Monitoring
| Obszar | Miara | Częstotliwość | Próg alarmowy |
|---|---|---|---|
| populacja | PSI wyniku, CSI cech karty i top-10 SHAP | kwartalnie | > 0,10 obserwuj, > 0,25 przegląd |
| dyskryminacja | Gini, KS na nowych defaultach | rocznie | spadek Gini > 0,10 |
| kalibracja | średnie PD vs DR, Brier, ECE, Jeffreys per klasa | rocznie | czerwone światło w ≥ 2 klasach |
| wyjaśnienia | Jaccard top-5 SHAP, znaki β | kwartalnie | Jaccard < 0,5 lub zmiana znaku |
| decyzje | stopa akceptacji, DR wśród przyjętych | miesięcznie | odchylenie > 5 pp |

PSI między train a test: < 0,05 dla wyniku i wszystkich cech karty.

## Użycie AI (regulamin pkt 10.3)
Do przygotowania kodu, analizy wyników i dokumentacji użyto asystenta AI (Claude, Anthropic): streszczenie metod z laboratoriów,
napisanie kodu notebooka na ich wzór, przygotowanie raportu, Model Card i prezentacji. Wszystkie wyniki pochodzą z uruchomienia
notebooka; zespół zweryfikował decyzje modelowe i interpretacje. Fragmenty kodu przeniesione z laboratoriów są oznaczone w notebooku
komentarzem `# z Lab N` (regulamin pkt 10.5).
