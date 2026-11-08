"""Prezentacja dla osób nietechnicznych (12 slajdów) -> prezentacja_projekt1.pptx. Uruchamiać z folderu projekt1."""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

GRANAT, SZARY, CZERWONY, CZARNY = RGBColor(0x1E, 0x3A, 0x5F), RGBColor(0x47, 0x55, 0x69), RGBColor(0xA3, 0x26, 0x2C), RGBColor(0x11, 0x18, 0x27)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
PUSTY = prs.slide_layouts[6]


def tytul(s, tekst, pod=None):
    tf = s.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(1.0)).text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = tekst; p.font.size = Pt(30); p.font.bold = True; p.font.color.rgb = GRANAT
    if pod:
        p2 = tf.add_paragraph(); p2.text = pod; p2.font.size = Pt(16); p2.font.color.rgb = SZARY


def punkty(s, lista, x, y, w, h, rozmiar=18):
    tf = s.shapes.add_textbox(x, y, w, h).text_frame
    tf.word_wrap = True
    for i, t in enumerate(lista):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        pod = t.startswith("  ")
        p.text = ("– " if pod else "• ") + t.strip()
        p.level = 1 if pod else 0
        p.font.size = Pt(rozmiar - 3 if pod else rozmiar)
        p.font.color.rgb = SZARY if pod else CZARNY
        p.space_after = Pt(8)


def wynik(s, liczba, opis, x, y, kolor=GRANAT):
    tf = s.shapes.add_textbox(x, y, Inches(3.8), Inches(1.4)).text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = liczba; p.font.size = Pt(40); p.font.bold = True; p.font.color.rgb = kolor
    p2 = tf.add_paragraph(); p2.text = opis; p2.font.size = Pt(15); p2.font.color.rgb = SZARY


def obraz(s, plik, x, y, w=None, h=None):
    s.shapes.add_picture(f"wykresy/{plik}", x, y, width=w, height=h)


# 1. tytuł
s = prs.slides.add_slide(PUSTY)
tytul(s, "Ocena ryzyka firm: model ratingowy", "Projekt 1 - interpretowalność, wyjaśnialność i kalibracja PD do 4%")
punkty(s, ["Dla kogo: analityk kredytowy i komitet kredytowy",
           "Pytanie: jak prawdopodobne jest, że firma nie spłaci zobowiązań - i dlaczego model tak uważa?",
           "Wynik: ocena ratingowa (AAA ... CCC), decyzja przyjąć / odrzucić i powody tej decyzji"],
       Inches(0.6), Inches(2.0), Inches(12), Inches(4))

# 2. problem i dane
s = prs.slides.add_slide(PUSTY)
tytul(s, "Problem i dane", "3 000 spółek z o.o., ich sprawozdania finansowe i informacja, czy doszło do zdarzenia ryzyka")
wynik(s, "3 000", "firm w danych", Inches(0.6), Inches(1.7))
wynik(s, "15%", "firm z defaultem (445)", Inches(4.6), Inches(1.7), CZERWONY)
wynik(s, "222", "informacje o każdej firmie: bilans, wyniki, wskaźniki", Inches(8.6), Inches(1.7))
punkty(s, ["Czego nie użyliśmy i dlaczego:",
           "  trzech wskaźników branżowych o nieznanym okresie pomiaru - mogły zawierać informację „z przyszłości” (to kosztuje trafność, ale chroni przed złudnie dobrym modelem)",
           "  przepływów pieniężnych - brak danych dla 99% firm",
           "Dane podzieliliśmy na trzy części: nauka modelu (60%), ustawienie kalibracji i progu (20%), niezależny sprawdzian (20%)",
           "Ta sama firma mogła występować kilka razy - takie firmy trzymaliśmy w jednej części"],
       Inches(0.6), Inches(3.4), Inches(12.2), Inches(4), rozmiar=17)

# 3. modele
s = prs.slides.add_slide(PUSTY)
tytul(s, "Dwa modele: prosty i złożony", "Porównujemy model, który da się w pełni wytłumaczyć, z modelem typu „czarna skrzynka”")
punkty(s, ["Karta scoringowa: 7 cech, za każdą firma dostaje punkty",
           "  więcej punktów = mniejsze ryzyko; każdy punkt da się wytłumaczyć klientowi",
           "Model złożony (XGBoost): 164 cechy, 300 drzew decyzyjnych",
           "  wyjaśniamy go metodami pośrednimi (SHAP, LIME)",
           "Trafność prawie taka sama: miara Gini 0,46 vs 0,49",
           "  różnica mieści się w granicach przypadku (przedział ufności obejmuje zero)"],
       Inches(0.5), Inches(1.7), Inches(6.3), Inches(5.5), rozmiar=17)
obraz(s, "roc_pr_test.png", Inches(6.9), Inches(1.9), w=Inches(6.2))

# 4. co wpływa na ocenę
s = prs.slides.add_slide(PUSTY)
tytul(s, "Co wpływa na ocenę firmy", "Oba modele patrzą na te same obszary finansowe")
punkty(s, ["↓ ryzyko: zysk (płacony podatek dochodowy), kapitał własny, rentowność sprzedaży, pozostałe przychody",
           "↑ ryzyko: zadłużenie - zwłaszcza gdy zobowiązania przekraczają 85% aktywów",
           "Kierunki zgodne z intuicją ekonomiczną w obu modelach",
           "Trzy najważniejsze cechy modelu złożonego są też w karcie scoringowej"],
       Inches(0.5), Inches(1.7), Inches(5.6), Inches(5.5), rozmiar=17)
obraz(s, "karta_woe_profile.png", Inches(6.1), Inches(1.8), w=Inches(7.0))

# 5. przykład firmy
s = prs.slides.add_slide(PUSTY)
tytul(s, "Przykład: firma na granicy decyzji", "Ryzyko 18,2% - dokładnie na progu opłacalności kredytu")
obraz(s, "waterfall_2.png", Inches(0.4), Inches(1.6), h=Inches(5.6))
punkty(s, ["Co podnosi ryzyko tej firmy:",
           "  brak pozostałych przychodów operacyjnych",
           "  brak zysku (zerowy podatek dochodowy)",
           "Co je obniża: wysoki kapitał własny (81% aktywów)",
           "Co klient może zmienić: zysk, zadłużenie, kapitał",
           "Czego nie zmieni: branża, wiek firmy",
           "Takie firmy warto przekazać analitykowi - mała zmiana wyników zmienia decyzję"],
       Inches(7.4), Inches(1.7), Inches(5.6), Inches(5.5), rozmiar=16)

# 6. czy wyjaśnieniom można ufać
s = prs.slides.add_slide(PUSTY)
tytul(s, "Czy wyjaśnieniom można ufać?", "Sprawdziliśmy, czy wyjaśnienia są wierne modelowi i czy się nie zmieniają")
wynik(s, "50×", "większa zmiana ryzyka po usunięciu cechy wskazanej przez SHAP niż losowej", Inches(0.6), Inches(1.7))
wynik(s, "~20%", "zachowania modelu opisuje LIME w otoczeniu firmy - za mało, by mu ufać", Inches(4.6), Inches(1.7), CZERWONY)
wynik(s, "100%", "dokładność wyjaśnień karty: suma punktów = ocena", Inches(8.6), Inches(1.7))
punkty(s, ["LIME daje różne wyniki przy kolejnych uruchomieniach, gdy próbek jest mało - stabilizuje się przy 5 000 próbek",
           "Ranking ważnych cech modelu złożonego zmienia się, gdy uczymy go na innych danych: wiele wskaźników mierzy to samo",
           "Wyjaśnienia opisują model, a nie przyczyny: „spadek zadłużenia o X obniży ryzyko o Y” nie jest gwarancją"],
       Inches(0.6), Inches(3.6), Inches(12.2), Inches(3.5), rozmiar=17)

# 7. kalibracja 4%
s = prs.slides.add_slide(PUSTY)
tytul(s, "Co znaczy „ryzyko 4%”?", "Średnie prawdopodobieństwo defaultu dostosowaliśmy do założenia biznesowego: 4% w docelowym portfelu")
punkty(s, ["W danych defaultów jest 15% - prawie 4 razy więcej niż w docelowym portfelu",
           "Przesuwamy wszystkie oceny o tyle samo: kolejność firm się nie zmienia",
           "Na danych przeliczonych do 4% defaultów oceny są trafne (średnio 3,7% wobec 4,0%)",
           "Jeśli wysoki odsetek defaultów to skutek złego okresu, oceny 4% będą w podobnym okresie za niskie - trzeba to monitorować"],
       Inches(0.5), Inches(1.9), Inches(5.6), Inches(5.3), rozmiar=17)
obraz(s, "kalibracja_4proc.png", Inches(6.1), Inches(2.4), w=Inches(7.1))

# 8. ratingi i próg
s = prs.slides.add_slide(PUSTY)
tytul(s, "Ratingi i decyzja o kredycie", "Kredyt opłaca się, gdy oczekiwany zysk (marża 10%) przewyższa oczekiwaną stratę (45% kwoty)")
wynik(s, "3% → 50%", "odsetek defaultów rośnie od AAA do CCC", Inches(0.6), Inches(1.7))
wynik(s, "89%", "firm przyjętych przy progu modelu złożonego", Inches(4.6), Inches(1.7))
wynik(s, "+87%", "zysku portfela vs przyjęcie wszystkich (20,6 vs 11,0)", Inches(8.6), Inches(1.7), CZERWONY)
obraz(s, "backtest.png", Inches(0.4), Inches(3.4), w=Inches(7.6))
punkty(s, ["Przy tej samej liczbie przyjętych firm oba modele podejmują te same decyzje dla 87-94% firm",
           "Próg najlepiej ustalić według planowanej stopy akceptacji - zysk zmienia się powoli wokół optimum"],
       Inches(8.2), Inches(3.6), Inches(4.9), Inches(3.5), rozmiar=15)

# 9. kontrfakt
s = prs.slides.add_slide(PUSTY)
tytul(s, "Co firma musi zmienić, by poprawić rating?", "Firma w klasie BB (ryzyko 9,9%): kapitał własny 0,3% aktywów, zobowiązania 99,7% aktywów")
punkty(s, ["Najmniejsza zmiana: zmniejszyć zadłużenie - spłata lub dokapitalizowanie ok. 1,4 mln zł",
           "  kapitał własny z 0,3% do ok. 15% aktywów → ryzyko 6,5%, rating BBB",
           "Inne pojedyncze zmiany (zysk, przychody) nie wystarczają",
           "Karta scoringowa daje tę samą odpowiedź: brakuje 4 punktów, a niższe zadłużenie daje 22",
           "Ograniczenia: zmiana jednej pozycji zmienia też inne; model nie mówi, czy to wykonalne ani czy na pewno zadziała"],
       Inches(0.6), Inches(1.9), Inches(12.2), Inches(5), rozmiar=18)

# 10. ryzyka i ograniczenia
s = prs.slides.add_slide(PUSTY)
tytul(s, "Ryzyka i ograniczenia")
punkty(s, ["Mało defaultów w zbiorze sprawdzającym (89) - wyniki są niepewne, zwłaszcza w małych grupach",
           "Pominęliśmy trzy wskaźniki branżowe; jeśli dostawca danych potwierdzi, że są z przeszłości, model można wzbogacić",
           "Nie znamy kwot kredytów ani dat oceny - zysk liczony na jednostkę kredytu, bez oceny zmian w czasie",
           "Ryzyko firm mikro jest zawyżane, części form własności i przetwórstwa - zaniżane (do sprawdzenia na większej próbie)",
           "Wyjaśnienia modelu złożonego są mniej stabilne niż karty, LIME - mało wiarygodny",
           "Założenie 4% wymaga monitorowania w gorszych okresach gospodarczych"],
       Inches(0.6), Inches(1.5), Inches(12.2), Inches(5.6), rozmiar=18)

# 11. monitoring
s = prs.slides.add_slide(PUSTY)
tytul(s, "Jak będziemy pilnować modelu", "Plan monitoringu po wdrożeniu")
punkty(s, ["Co kwartał: czy nowi klienci są podobni do tych z danych (wskaźnik PSI; alarm > 0,25)",
           "Co rok: czy model nadal odróżnia dobre firmy od złych (Gini; alarm przy spadku o 0,10)",
           "Co rok: czy odsetek defaultów w każdej klasie ratingowej zgadza się z oceną (test w każdej klasie)",
           "Co kwartał: czy najważniejsze cechy pozostają te same",
           "Co miesiąc: odsetek przyjętych firm i defaultów wśród nich"],
       Inches(0.6), Inches(1.7), Inches(12.2), Inches(5.4), rozmiar=19)

# 12. rekomendacja
s = prs.slides.add_slide(PUSTY)
tytul(s, "Rekomendacja")
punkty(s, ["Model ratingowy: karta scoringowa",
           "  prawie tak trafna jak model złożony, a każdą ocenę da się wytłumaczyć klientowi i walidatorowi",
           "Model złożony (XGBoost): jako model kontrolny",
           "  firmy, które oba modele oceniają bardzo różnie, przekazujemy analitykowi",
           "Próg decyzji: według planowanej stopy akceptacji, z oceną zysku na danych sprawdzających",
           "Odpowiedź na pytanie projektu: niewielka poprawa trafności modelu złożonego nie uzasadnia utraty przejrzystości"],
       Inches(0.6), Inches(1.5), Inches(12.2), Inches(5.6), rozmiar=19)

prs.save("prezentacja_projekt1.pptx")
print("zapisano prezentacja_projekt1.pptx,", len(prs.slides), "slajdów")
