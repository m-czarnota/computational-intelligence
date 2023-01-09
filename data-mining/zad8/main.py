def get_unique_items_from_dataset(dataset, with_counts: bool = True):
    uniques = {}

    for values in dataset.values():
        for value in values:
            if value in uniques.keys():
                uniques[value] += 1
            else:
                uniques[value] = 1

    if not with_counts:
        return uniques.keys()

    return uniques


def calc_min_supp_and_min_conf(dataset):
    uniques = get_unique_items_from_dataset(dataset)

    supports = []
    origin_vals = []

    for unique1 in uniques:
        origin_vals.append(unique1)
        resulting_vals = []

        for unique2 in uniques:
            if unique1 == unique2:
                continue

            resulting_vals.append(unique2)

            for dataset_values in dataset.values():
                ...


def generate_candidate_for_one_element(frequences):
    candidates = []

    for frequence1 in frequences:
        for frequence2 in frequences:
            if frequence1 != frequence2:
                candidates.append([frequence1, frequence2])

    return candidates


def generate_candidate_for_one_more_element(frequences):
    candidates = []

    for frequences_vals1 in frequences:
        for frequences_vals2 in frequences:
            is_shared_element = set(frequences_vals1).issubset(set(frequences_vals2))

            if is_shared_element:
                new_candidates = [*frequences_vals1, *frequences_vals2]
                candidates.append(new_candidates)

    return candidates


def frequent_itemset_generation_apriori(dataset):
    k = 0
    min_supp = 0.2
    f = [get_unique_items_from_dataset(dataset)]

    while True:
        k += 1

        c = generate_candidate(f[k - 1])
        c = c

        for transaction in dataset.values():
            ct =

        if len(f) != 0:
            break

    return f


if __name__ == '__main__':
    dataset = {
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
    }



    """
    support:
    supp({'a'} => {'c'}) = 3/10
    supp({'a', 'b'} => {'c'}) = 2/10
    
    conf - czyli support wewnątrz conf przez wszystkie możliwości przed strzałką
    conf({'a'} => {'c'}) = (3/10) / (5/10)
    conf({'a', 'b'} => {'c'}) = (2/10) / (3/10)
    
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
