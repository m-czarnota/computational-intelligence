import pandas as pd


if __name__ == '__main__':
    data_train = pd.read_csv('./DataPrepared/train-features-50.txt', names=['document', 'word', 'class'], sep=' ')
    matrix = pd.DataFrame(data=0, index=data_train['document'].unique(), columns=data_train['word'].unique())
    matrix = matrix.applymap(lambda x: {0: 0, 1: 0})
    print(matrix)

    for row_index, row in data_train.iterrows():
        document = row['document']
        word = row['word']
        is_spam = row['class'] == 2

        matrix[word][document][1 if is_spam else 0] += 1



"""
nltk library
scikit-learn ma feature extraction -> count vectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
chodzi o to, żeby skonwertować dowolny text od razu na macierz, która będzie zawierała
on robi kilka rzeczy w tle. może zrobić tokenizację. ma też stop words.
ztokenizować text - co to jest token, to jest już kwestia definicji, założenia
o ngram zapomnieć
ważny parameter analyzer - nie trzeba nic zmieniać
jakbyśmy chcieli stop words wykluczyć to trzeba włączyć stop words
jak włączymy binary to będzie zliczał, czy słowo w ogóle występuje w text czy nie

fit_transform pobiera word documents i on ekstraktuje te tokeny i uczy się
fit dokonuje tokenizacji i je transformuje
get_feature_names_out() pokazuje jakie ma tokeny ze słownika w kolejności alfabetycznej
po nauczeniu on przechowuje częstości. po nauczeniu zwracamy X. jak zrobimy z niego toarray(), to zrobi z tego macierz
liczba wierszy to liczba dokumentów
liczba kolumn to liczba słów

ngram jest ciekawe, ale nie robimy

DataEmails.zip zawiera słowa, ale one są już przetworzone
nie trzeba robić split, bo już są podzielone
sklearn nie umie lematyzacji, umie to biblioteka nltk.word_lematize oraz scipy
drugi przykład z pdf jest w DataEmails.zip
do vectorizera dajemy wszytkie dokumenty tekstowe, gdzie każdy dokument to plik txt
dwie klasy: spam i nie-spam

w klasyfikacji bayessa będziemy patrzyli jak słowa korespondują z klasą
będziemy patrzyli jak często słowo pojawia się w danej klasie
i te częstości po fit muszą być przechowywane w naiwnym klasyfikatorze bayessa
we wzorze phi dla pierwszej i drugiej klasy, dla każdej z cech w dokumencie w danej klasie się jak często rozkłada
dla każdego wyrazu z dokumentu trzeba mieć 2
m to liczba maili w zbiorze treningowym, a i-ty mail zawiera n słów
te częstości trzeba sobie wyznaczyć
dla każdej klasy każdy wyraz ma być policzony jak często występuje w danym dokumencie w danej klasie
aby robić predycję to jest to napisane log. zrobić log po wszystkich. jak etykieta 1 to spam, jak 0 to nie-spam, czyli argmax rozpisany

w DataPrepared.zip dane są trójkami
identyfikator_dokumentu identyfikator_tokena częstość
czyli w train-features.txt linia pierwsza: 1 pierwszym dokumencie słowo 19 pojawiło się 2 razy
train_labels.txt są kolumna z 0 i 1. najpierw są nie-spam, potem spam

jak umieścić te dane, aby łatwo był zliczać?
tworzymy wiersz macierzy, który reprezentuje dokument
czyli wiersz to jest dokument, a w środku wiersza będą słowa
więc trzeba wiedzieć ile jest identyfikatorów słów
etykiety są liczbowe w przygotowanym zbiorze train_labels.txt
ten plik określa gdzie w danym dokumencie występuje dane słowo, a nas jeszcze interesuje w której to jest klasie
wiec trzeba dodać trzeci wymiar

macierz do count vectorizer ma być złożona z:
    wiersze to numer dokumentu
    kolumny to identyfikatory słowa
zliczać częstości, gdzie występuje swoje słowo
numer, że to słowo było tak częste i ono wpływa jakoś na prawdopodobieństwo wyjściowe

napisać metode fit
ona oblicza phi, czestości i
to już jest wygładzane

dla zbiorów testowych, dla każdego testu (dokumentu) też zobaczyć jakie były etykiety
poszukać w jednej klasie jakie były słowa 
jak wyjdzie że suma testowych jest częstsza w spamie, to częstość będzie większa

biore pierwszy dokument testowy i pobieram częstość dla danego słowa
robie to tak długo aż idektyfikator dokumentu (pierwsza kolumna) się nie zmieni

na podstawie zbioru uczącego wyciągamy pxy w klasie 1 i pxy w klasie 0
i w predict wyciągamy odpowiedź

błąd klasyfikacji możemy liczyć jako accuracy
jak wezme sobie test i przepuszcze wartość ze zbioru testowego i dostane odpowiedź, to będę miał dla każdego dokumentu odpowiedzi 0 i 1
liczę accuracy

można wykorzystać train-features-50.txt, który ma mniej słów
zrobić takie badanie, w którym mierzymy accuracy ze względu na liczbę słów
czy zmniejszanie słów do uczenia wpływa na accuracy i jak to wpływa na ostateczną decyzję
jaka liczba dokumentów jest wystarczająca?

ponieważ pracujemy z tekstem, to będzie można zobaczyć jakie słowa na to wpływają
w przypadku pkt 7 należy koniecznie użyć count vectorizer

wzory:
    1 suma jest po m, czyli wszystkich mailach w zbiorze treningowym
        2 suma jest po wszytkich wyrazach w dokumencie
            za każdym razem w danym dokumencie
                wyraz j-ty w i-tym dokumencie ma wartość k
                jeżeli to słowo wystąpi w danym dokumencie i etykieta jest 1
                    dodaje 1
                    
                w mianowniku - jest on po całym zbiorze treningowym:
                    ni to liczba słów
                    
macierz do zliczania do fit, to macierz 3 wymiarowa: dokument, słowo, klasa
"""