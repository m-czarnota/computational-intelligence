from __future__ import annotations
import pandas as pd
import copy
from itertools import combinations_with_replacement

from VerbosityHelper import VerbosityHelper


class Apriori:
    def __init__(self, min_supp: float = 0.3, min_conf: float = 0.7, verbose: bool = True, verbosity_level: int = 1):
        self.min_supp: float = min_supp
        self.min_conf: float = min_conf

        self.frequents: list = []
        self.rules: pd.DataFrame = pd.DataFrame()

        self.__verbosity_helper = VerbosityHelper(verbose, verbosity_level)
        self.__dataset_length: int = 0

    def generate_rules(self, dataset: pd.DataFrame) -> pd.DataFrame:
        self.__dataset_length = dataset.shape[0]

        self.frequents = self.__verbosity_helper.verbose(self._generate_candidates, [dataset], message='Time of generating candidates')
        self.rules = self.__verbosity_helper.verbose(self._create_rules, message='Time of creating rules')

        return self.rules

    def _generate_candidates(self, dataset: pd.DataFrame) -> list:
        frequents: list = []
        f: dict = self.__verbosity_helper.verbose(self.__get_unique_items_from_dataset, [dataset], 2, 'Time of selecting uniques items from dataset')

        while len(f) != 0:
            frequents.append(copy.deepcopy(f))

            c = self.__generate_candidates_from_frequencies(f, dataset)
            c = self.__prune_candidates(c)

            f = copy.deepcopy(c)

        return frequents

    def _create_rules(self) -> pd.DataFrame:
        frequents_mapped = list(reversed([{key: set() for key in frequencies} for frequencies in self.frequents[1:]]))
        generated_rules = pd.DataFrame(columns=['rule', 'supp', 'conf', 'lift'])

        for frequencies in frequents_mapped:
            for keys in frequencies.keys():
                df = pd.DataFrame(columns=['rule', 'supp', 'conf', 'lift'])
                generated_rules = pd.concat([generated_rules, self.__generate_rules_helper(keys, df)],
                                            ignore_index=True)

        return generated_rules.sort_values(by=['lift'], ascending=False)

    @staticmethod
    def __get_unique_items_from_dataset(dataset: pd.DataFrame, with_counts: bool = True) -> dict | list:
        uniques = {}

        for transaction_iter, transaction_items in dataset.iterrows():
            for value in transaction_items:
                if value is None or pd.isnull(value):
                    continue

                if value in uniques.keys():
                    uniques[value] += 1
                else:
                    uniques[value] = 1

        if not with_counts:
            return list(uniques.keys())

        return uniques

    @staticmethod
    def __generate_candidates_from_frequencies(frequencies: dict, dataset: pd.DataFrame) -> dict:
        """
        Generates all combinations of items in transactions dataset and count them.

        Example:
             {'a': 5, 'b': 7, 'c': 7}, {('a', 'b'): 3, ('b', 'c'): 5}, {('a', 'b', 'c'): 2}
        """
        candidates = {}
        first_item = list(frequencies.keys())[0]

        if type(first_item) == str:  # for one element item set
            candidates = {tuple(sorted([a, b])): 0 for (a, b) in combinations_with_replacement(list(frequencies.keys()), 2) if a != b}
        else:  # for multi element item set
            for frequency_item1 in frequencies.keys():
                frequency_item1_set = set(frequency_item1)

                for frequency_item2 in frequencies.keys():
                    frequency_item2_set = set(frequency_item2)

                    if len(frequency_item1_set.intersection(frequency_item2_set)) == 0 or frequency_item1_set == frequency_item2_set:
                        continue

                    key = tuple(sorted({*frequency_item1_set, *frequency_item2_set}))
                    candidates[key] = 0

        for keys in candidates.keys():
            where_is_keys = dataset.isin(keys).sum(axis=1) >= len(keys)
            candidates[keys] = where_is_keys.sum()

        return candidates

    def __prune_candidates(self, candidates: dict) -> dict:
        pruned_candidates = {}

        for key, count in candidates.items():
            if count / self.__dataset_length >= self.min_supp:
                pruned_candidates[key] = count

        return pruned_candidates

    def __generate_rules_helper(self, keys: list, local_rules: pd.DataFrame, local_word: list = []) -> pd.DataFrame:
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
            supp_expression = self.__find_expression_count_in_frequents(tuple(keys)) / self.__dataset_length
            supp_resulting = self.__find_expression_count_in_frequents(tuple([*word])) / self.__dataset_length
            conf = supp_expression / supp_resulting

            # reverse pruning - add only when condition is meet
            if conf >= self.min_conf:
                supp_factors = self.__find_expression_count_in_frequents(keys_copy_tuple) / self.__dataset_length

                local_rule = pd.Series({
                    'rule': f'{keys_copy_tuple} => {word_tuple}',
                    'supp': {'expr': supp_expression, 'factors': supp_factors, 'resulting': supp_resulting},
                    'conf': conf})
                local_rules = pd.concat([local_rules, local_rule.to_frame().T], ignore_index=True)

            generated_rules = self.__generate_rules_helper(keys_copy, local_rules, word)
            generated_rules = generated_rules.sort_values(by=['conf'], ascending=False).drop_duplicates(subset=['rule'])
            local_rules = pd.concat([local_rules, generated_rules], ignore_index=True)

        # removing duplicated rules and calc lift for rules
        local_rules = local_rules.sort_values(by=['conf'], ascending=False).drop_duplicates(subset=['rule'])
        local_rules['lift'] = self.__calc_lift_for_rules(local_rules)

        return local_rules

    def __find_expression_count_in_frequents(self, expression: tuple) -> int:
        expression = tuple(sorted(expression))
        expression = expression[0] if len(expression) == 1 else expression
        index_to_search = len(expression) - 1 if type(expression) != str else 0

        return self.frequents[index_to_search][expression]

    @staticmethod
    def __calc_lift_for_rules(local_rules: pd.DataFrame) -> list:
        lifts = []

        for rule_iter, rule in local_rules.iterrows():
            supp = rule['supp']

            lift = supp['expr'] / (supp['factors'] * supp['resulting'])
            lifts.append(lift)

        return lifts
