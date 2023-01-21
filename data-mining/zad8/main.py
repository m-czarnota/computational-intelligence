import pandas as pd

from Apriori import Apriori


if __name__ == '__main__':
    dataset = pd.DataFrame.from_dict({
        "t1": {'a', 'b', 'c'},
        "t2": {'b', 'c', 'd'},
        't3': {'a', 'b', 'd', 'e'},
        't4': {'a', 'c', 'd', 'e'},
        't5': {'b', 'c', 'd', 'e'},
        't6': {'b', 'd', 'c'},
        't7': {'c', 'd'},
        't8': {'a', 'b', 'c'},
        't9': {'a', 'd', 'e'},
        't10': {'b', 'd'},
    }, orient='index')
    dataset2 = pd.DataFrame.from_dict({
        "t1": {'bułka', 'parówka', 'ketchup'},
        "t2": {'bułka', 'parówka'},
        't3': {'parówka', 'pepsi', 'chipsy'},
        't4': {'pepsi', 'chipsy'},
        't5': {'chipsy', 'ketchup'},
        't6': {'parówka', 'pepsi', 'chipsy'},
    }, orient='index')

    store = pd.read_csv('Store_data.csv')

    min_supp = 0.005
    min_conf = 1.5

    apriori = Apriori(min_supp, min_conf, verbose=True, verbosity_level=2)
    print(apriori.generate_rules(store).to_markdown())


    """
    support:
    supp({'a'} => {'c'}) = 3/10
    supp({'a', 'b'} => {'c'}) = 2/10
    
    conf - czyli support całości przez support tegp przed strzałką
    conf({'a'} => {'c'}) = (3/10) / (5/10)
    conf({'a', 'b'} => {'c'}) = (2/10) / (3/10)
    
    lift - czyli support całości przez iloczyn supportu pojedynczych osobno rzeczy
    lift({'a'} => {'c'}) = ((3/10)) / ((5/10) * (7/10))
    lift({'a', 'b'} => {'c'}) = ((2/10)) / ((3/10) * (7/10))
    
    wygenerowanie zbioru o różnych częstościach
    znaleźć wszystkie unikalne elmenty w transakcjach i policzyć wszystkie częstości dla nich
    wystarczy posługiwać się częstościami
    powstaje zbiór f1, tworzy się z niego kadydatów k+1 na podstawie kandydatów o liczności k, którzy zostali zidentyfikowani i wyczyszczeni progiem support (czyli częstość występowania elementów, powyżej której jesteśmy zainteresowani wynikami - częste będą służyły do tworzenia reguł)
    ten support jest wykorzystywany jako próg i na podstawie tworzy się zbiór dwuelementowy
    czysci się zbiór dwuelementowy i na jego podstawie tworzy się zbiór trzyelementowy
    z pierszej części zwraca się suma wszystkich podzbiorów Fk
    dla k = 2, 3, aż do jakiegoś k (jak wyznaczymy sobie minimum supportów to w pewnej chwili następuje odcięcie, elemnty będą odcinane)
    załóżmy, że w zbiorze jest 10 elementów, a support wyniesie 4, to możemy tworzyc reguły które będą miały jedynie 4 elementy
    czyli co najmnej 2 podzbiory muszą być z pierwszego

    5.3 ap-genrules
    znaleźć wszystkie reguły, które przechodzą minimum konflikt
    na przykłąd mamy reguły dwu elemnetowe a i b, to wyznaczamy wszytkie zależności: z a wynika b, z b wynika a
    support dla a i b dzielony przez supporty
    z a wynika b, c; z bc wynika a; wszystkie kombinacje
    """
