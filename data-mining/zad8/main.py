import copy


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


def generate_candidates(frequencies: dict, dataset) -> dict:
    candidates = {}

    for frequency_item1, frequency_vals1 in frequencies.items():
        for frequency_item2, frequency_vals2 in frequencies.items():
            frequency_item1_set = set(frequency_item1)
            frequency_item2_set = set(frequency_item2)

            # for one element item set
            if len(frequency_item1_set) == 1:
                if frequency_item1_set.issubset(frequency_item2_set) is True:
                    continue

                key = tuple(sorted({*frequency_item1_set, *frequency_item2_set}))
                candidates[key] = 0

                continue

            # for multi element item set
            if len(frequency_item1_set.intersection(frequency_item2_set)) == 0 or frequency_item1_set == frequency_item2_set:
                continue

            key = tuple(sorted({*frequency_item1_set, *frequency_item2_set}))
            candidates[key] = 0

    for keys in candidates.keys():
        keys_set = set(keys)

        for transaction_items in dataset.values():
            items_set = set(transaction_items)

            if keys_set.issubset(items_set):
                candidates[keys] += 1

    return candidates


def prune_candidates(candidates: dict, min_supp: float) -> dict:
    pruned_candidates = {}

    for key, count in candidates.items():
        if count >= min_supp:
            pruned_candidates[key] = count

    return pruned_candidates


def frequent_itemset_generation_apriori(dataset):
    k = 0
    min_supp = 2
    f = get_unique_items_from_dataset(dataset)

    while True:
        support_count = 0

        c = generate_candidates(f, dataset)
        c = prune_candidates(c, min_supp)

        for transaction_items in dataset.values():
            items_set = set(transaction_items)
            ct = []

            for candidates in c.keys():
                candidates_set = set(candidates)
                candidates_in_transaction = sorted(candidates_set.intersection(items_set))
                ct = [*ct, *candidates_in_transaction]

            support_count += len(ct)

        f = copy.deepcopy(c)

        if len(f) == 0:
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

    frequent_itemset_generation_apriori(dataset)


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
