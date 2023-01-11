import pandas as pd
import numpy as np

from NaiveBayes import NaiveBayes


def read_data(size: int = None, data_type: str = 'train'):
    data_size = '' if size is None else f'-{size}'

    data_train = pd.read_csv(f'./DataPrepared/{data_type}-features{data_size}.txt', names=['document', 'word', 'count'], sep=' ')
    data_labels = pd.read_csv(f'./DataPrepared/{data_type}-labels{data_size}.txt', names=['label']).squeeze('columns')
    data_labels.index += 1

    matrix = pd.DataFrame(data=0, index=data_train['document'].unique(), columns=data_train['word'].unique())

    for row_index, row in data_train.iterrows():
        document = row['document']
        word = row['word']

        matrix[word][document] = row['count']

    return matrix, data_labels


def experiment_for_size(size: int = None, train_data: tuple = None, test_data: tuple = None):
    matrix_train, data_labels_train = read_data(size) if train_data is None else train_data

    naive_bayes = NaiveBayes()
    naive_bayes.fit(matrix_train, data_labels_train)

    matrix_test, data_labels_test = read_data(data_type='test') if test_data is None else test_data
    # print(matrix_train.shape, matrix_test.shape)

    # predicted = naive_bayes.predict(matrix_test)
    # print(predicted)

    return naive_bayes.score(matrix_test, data_labels_test)


def experiment_random():
    matrix_train, data_labels_train = read_data()
    rng = np.random.default_rng()
    random_indexes_to_remove = rng.choice(data_labels_train.size, size=np.random.randint(50), replace=False)

    for random_index in random_indexes_to_remove:
        matrix_train = matrix_train.drop(random_index)
        data_labels_train.drop(random_index)

    score = experiment_for_size(train_data=(matrix_train, data_labels_train))
    print(f'Accuracy for random train size {data_labels_train.size}: {score}')


def experiment_with_size():
    sizes = [50, 100, 400, None]
    matrix_test, data_labels_test = read_data(data_type='test')

    for size in sizes:
        score = experiment_for_size(size, test_data=(matrix_test, data_labels_test))
        print(f'Accuracy for train size {size if size is not None else "all"}: {score}')


if __name__ == '__main__':
    matrix_train, data_labels_train = read_data(50)
    matrix_test, data_labels_test = read_data(data_type='test')

    clf = NaiveBayes()
    clf.fit(matrix_train, data_labels_train)
    print(clf.score(matrix_test, data_labels_test))

    # experiment_with_size()
    # experiment_random()


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
