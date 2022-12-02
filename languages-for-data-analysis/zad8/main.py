if __name__ == '__main__':
    pass

"""
utm-brama, utm-obrotnica, wraki utm 
wraki utm jest plikiem dużo mniejszym

przykladowe-dane-GRID pokazują jak zapisać dane
number cols, number rows to rozmiary macierzy
xllcenter, yllcenter to wartość pkt startu w danych metrycznych - zawsze my mamy to samo
cellsize to my ustawiamy
nodata value to u nas NaN

to co jest na czerwono jest wyżej punktowane, bez czerwonych max ocena 4
im więcej czerwonych tym lepiej. jak idziemy w czerwone to odpuszczamy niebieskie
user ustawia rozdzielczość grid, czy to 1m, 2m, 5m. to jako jedna zmienna
określenie rozmiaru okienka - albo po kwadratach albo po okienkach
minimalna liczba pkt do obliczeń
nie korzystamy z bibliotek i gotowych rozwiązań dla niebieskich, dozwolone tylko dla czerwonych. nie może być jedna funkcja, tylko samemu

jak mamy milion pomiarów można zrobić pętle, ale będzie się długo liczyć
aby było szybciej można wykorzystać kd tree
w kd tree elementy na dole są tak ułożone że na dole będą wartości leżące w pobliżu siebie
napisać to jedną funkcją

wczytać dane
wyznaczyć wartości graniczne x_min, x_max
na podstawie parametrów określić rozmiar macierzy grid
dane wczytać do kd_tree
w pętli po x, y pobierać z kd_tree wartości, wyliczać i zapisywać w konkretnym miejscu
powinno w miarę szybko się liczyć. powinno się to liczyć kilka sekund, kilkanaście sekund
z kd_tree okręgi będą prostsze niż kwadraty i są bardziej preferowane
stalamy w jakim otoczeniu szukamy punktu
kd_tree ma takie metody jak znajdź wszytkie najbliższe pkt leżące w danej odległości, znajdź najbliższe punkty dla danego punktu

jak mam prostokąt 
    480
100     250
    300
i mam rozmiar okna 0.5
to liczę grida o rozmiarze 300x360 (250 - 100)*0.5, (480-300)*0.5

dla moving average: jak kd_tree zwróci mi 5 wartości, to liczę z nich średnią
pierwszy pkt w grid odpowiada x_min, następny odpowiada (x_min + x) / x_max

na początku bawić się na wrakach
dostajemy pliki metryczne UTM (M to jest metric)
co to znaczy? 
    dx = |x2 - x1|
    dy = |y2 - y1|
    d = sqrt(dx**2 + dy**2)
"""