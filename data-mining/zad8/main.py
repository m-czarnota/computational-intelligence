from __future__ import annotations
import copy
import pandas as pd


def get_unique_items_from_dataset(dataset, with_counts: bool = True) -> dict | list:
    uniques = {}

    for values in dataset.values():
        for value in values:
            if value in uniques.keys():
                uniques[value] += 1
            else:
                uniques[value] = 1

    if not with_counts:
        return list(uniques.keys())

    return uniques


def generate_candidates(frequencies: dict, dataset) -> dict:
    """
    Generates all combinations of items in transactions dataset and count them.

    Example:
         {'a': 5, 'b': 7, 'c': 7}, {('a', 'b'): 3, ('b', 'c'): 5}, {('a', 'b', 'c'): 2}
    """
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
            if len(frequency_item1_set.intersection(frequency_item2_set)) == 0 \
                    or frequency_item1_set == frequency_item2_set:
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


def prune_candidates(candidates: dict) -> dict:
    pruned_candidates = {}

    for key, count in candidates.items():
        if count / len(dataset) >= min_supp:
            pruned_candidates[key] = count

    return pruned_candidates


def frequent_itemset_generation_apriori(dataset):
    frequents = []
    f = get_unique_items_from_dataset(dataset)

    while len(f) != 0:
        frequents.append(copy.deepcopy(f))

        c = generate_candidates(f, dataset)
        c = prune_candidates(c)

        f = copy.deepcopy(c)

    return frequents


def find_expression_count_in_frequents(expression: tuple, frequents: dict):
    expression = tuple(sorted(expression))
    expression = expression[0] if len(expression) == 1 else expression
    index_to_search = len(expression) - 1

    return frequents[index_to_search][expression]


def calc_lift_for_rules(local_rules: pd.DataFrame) -> list:
    lifts = []

    for rule_iter, rule in local_rules.iterrows():
        supp = rule['supp']

        lift = supp['expr'] / (supp['factors'] * supp['resulting'])
        lifts.append(lift)

    return lifts


def generate_rules_helper(frequents: dict, keys: list, local_rules: pd.DataFrame, local_word: list = []) -> pd.DataFrame:
    if len(keys) == 1:
        return local_rules

    for key_iter, key in enumerate(keys):
        keys_copy = copy.deepcopy(list(keys))  # copy to prevent modify original looped keys

        removed = keys_copy.pop(key_iter)
        keys_copy_tuple = tuple(keys_copy)  # tuple, because list is unhashable type

        # resulting element: from {} => {} the word is the second pair of brackets
        word = copy.deepcopy(local_word)
        word.append(removed)
        word = list(sorted(word))
        word_tuple = tuple(word)  # tuple, because list is unhashable type

        # calculating conf for pruning
        supp_expression = find_expression_count_in_frequents(tuple(keys), frequents) / len(dataset)
        supp_resulting = find_expression_count_in_frequents(tuple([*word]), frequents) / len(dataset)
        conf = supp_expression / supp_resulting

        # reverse pruning - add only when condition is meet
        if conf >= min_conf:
            supp_factors = find_expression_count_in_frequents(keys_copy_tuple, frequents) / len(dataset)

            local_rule = pd.Series({
                'rule': f'{keys_copy_tuple} => {word_tuple}',
                'supp': {'expr': supp_expression, 'factors': supp_factors, 'resulting': supp_resulting},
                'conf': conf})
            local_rules = pd.concat([local_rules, local_rule.to_frame().T], ignore_index=True)

        generated_rules = generate_rules_helper(frequents, keys_copy, local_rules, word)
        generated_rules = generated_rules.sort_values(by=['conf'], ascending=False).drop_duplicates(subset=['rule'])
        local_rules = pd.concat([local_rules, generated_rules], ignore_index=True)

    # sorted_rules = pd.DataFrame(columns=['rule', 'lift'])
    #
    # for rule_x, rules_y in local_rules.items():
    #     lifts_for_rule = list(sorted([calc_lift_for_rule(rule_x, rule_y, frequents) for rule_y in rules_y]))

    local_rules = local_rules.sort_values(by=['conf'], ascending=False).drop_duplicates(subset=['rule'])
    local_rules['lift'] = calc_lift_for_rules(local_rules)

    return local_rules


def generate_rules(frequents: list) -> pd.DataFrame:
    frequents_mapped = list(reversed([{key: set() for key in frequencies} for frequencies in frequents[1:]]))
    generated_rules = pd.DataFrame(columns=['rule', 'supp', 'conf'])

    for frequencies in frequents_mapped:
        for keys in frequencies.keys():
            df = pd.DataFrame(columns=['rule', 'supp', 'conf'])
            generated_rules = pd.concat([generated_rules, generate_rules_helper(frequents, keys, df)], ignore_index=True)

    return generated_rules.sort_values(by=['lift'], ascending=False)


def apriori_algorithm(dataset, min_supp: float = 0.3, min_conf: float = 0.7):
    freqs = frequent_itemset_generation_apriori(dataset)
    print(freqs)

    rules = generate_rules(freqs)
    print(rules.to_markdown())


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

    min_supp = 0.3
    min_conf = 0.7

    apriori_algorithm(dataset)


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
