import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os


class GraphModelGenerator:
    def __init__(self, save: bool = True, save_filepath: str = './images/results'):
        self.save = save
        self.save_filepath = save_filepath

        self._default_filepath_data: str = './data/results_main.csv'
        self._file = self.__read_file()

    def generate_average_measure_graph(self, filename: str = 'measure_average_infection.png'):
        measure_names = self._file['Measure'].unique()
        averages = [self._file[self._file['Measure'] == measure_name]['IF'].mean() for measure_name in measure_names]

        plt.figure(figsize=(20, 10))
        plt.bar(measure_names, averages, color='g')
        plt.title('Average infection by measure')
        plt.savefig(f'{self.save_filepath}/{filename}')

    def generate_averaged_increasing_infection_by_pp(self, filename: str = 'averaged_increasing_infection_by_pp.png'):
        networks = self._file['Network'].unique()
        filepath_to_save = f'{self.save_filepath}/{filename.replace("png", "").replace("jpg", "")}'

        if not os.path.exists(filepath_to_save):
            os.mkdir(filepath_to_save)

        for network in networks:
            network_data = self._file[self._file['Network'] == network]
            self.__generate_averaged_increasing_function_by_pp(network_data, filename)

    def generate_mean_coverage_by_network(self, filename: str = 'mean_coverage_by_network.png'):
        network_uniques = self._file['Network'].unique()
        network_uniques = np.array(list(map(lambda name: name.replace('.txt', ''), network_uniques)))
        network_uniques.sort()
        mean_coverage_by_network = self._file.groupby(['Network'], sort=True)['IF'].mean()

        plot_title = 'Mean coverage by network'
        self.__bar(network_uniques, mean_coverage_by_network, plot_title, filename)

    def generate_mean_coverage_by_pp(self, filename: str = 'mean_coverage_by_pp.png'):
        pp_uniques = self._file['PP'].unique()
        mean_coverage_by_pp = self._file.groupby(['PP'])['IF'].mean()

        plot_title = 'Mean coverage by propagation probability'
        self.__bar(pp_uniques, mean_coverage_by_pp, plot_title, filename)

    def generate_mean_coverage_by_sf(self, filename: str = 'mean_coverage_by_sf.png'):
        sf_uniques = self._file['SF'].unique()
        mean_coverage_by_sf = self._file.groupby(['SF'])['IF'].mean()

        plot_title = 'Mean coverage by seed fraction'
        self.__bar(sf_uniques, mean_coverage_by_sf, plot_title, filename)

    def generate_mean_coverage_by_measure(self, filename: str = 'mean_coverage_by_measure.png'):
        measure_uniques = self._file['Measure'].unique()
        mean_coverage_by_measure = self._file.groupby(['Measure'])['IF'].mean()

        plot_title = 'Mean coverage by measure'
        self.__bar(measure_uniques, mean_coverage_by_measure, plot_title, filename)

    def generate_mean_steps_by_network(self, filename: str = 'mean_steps_by_network.png'):
        network_uniques = self._file['Network'].unique()
        network_uniques = np.array(list(map(lambda name: name.replace('.txt', ''), network_uniques)))
        network_uniques.sort()
        mean_steps_by_network = self._file.groupby(['Network'])['Iterations count'].mean()

        plot_title = 'Mean steps by network'
        self.__bar(network_uniques, mean_steps_by_network, plot_title, filename)

    def generate_mean_steps_by_pp(self, filename: str = 'mean_steps_by_pp.png'):
        pp_uniques = self._file['PP'].unique()
        mean_steps_by_pp = self._file.groupby(['PP'])['Iterations count'].mean()

        plot_title = 'Mean steps by propagation probability'
        self.__bar(pp_uniques, mean_steps_by_pp, plot_title, filename)

    def generate_mean_steps_by_sf(self, filename: str = 'mean_steps_by_sf.png'):
        sf_uniques = self._file['SF'].unique()
        mean_steps_by_sf = self._file.groupby(['SF'])['Iterations count'].mean()

        plot_title = 'Mean steps by seed fraction'
        self.__bar(sf_uniques, mean_steps_by_sf, plot_title, filename)

    def generate_mean_steps_by_measure(self, filename: str = 'mean_steps_by_measure.png'):
        measure_uniques = self._file['Measure'].unique()
        mean_steps_by_measure = self._file.groupby(['Measure'])['Iterations count'].mean()

        plot_title = 'Mean steps by measure'
        self.__bar(measure_uniques, mean_steps_by_measure, plot_title, filename)

    def generate_max_coverage_by_method(self, filename: str = 'max_coverage_by_method.png'):
        max_coverage_by_measure = self._file.groupby(['Network', 'Measure'])['IF'].max()
        names = self._file['Network'].unique()

        plot_title = 'Max coverage by measure'
        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(20, 10))
        rects1 = ax.bar(x - width / 3, max_coverage_by_measure[::3], width, label='degree')
        rects2 = ax.bar(x + width / 3, max_coverage_by_measure[1::3], width, label='closeness')
        rects3 = ax.bar(x + width / 3, max_coverage_by_measure[2::3], width, label='random')

        ax.set_xticks(x, names)
        ax.set_ylabel('Infection coverage (%)')
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)

        ax.set_title(plot_title)
        plt.savefig(f'{self.save_filepath}/{filename}')
        plt.close(fig)

    def generate_max_coverage_by_method_and_pp(self, filename: str = 'max_coverage_by_method_and_pp.png'):
        grouped_by_network_and_column = self._file.groupby(['Network', 'Measure'])

    def __read_file(self, filepath: str = None) -> pd.DataFrame:
        return pd.read_csv(self._default_filepath_data if filepath is None else filepath)

    def __generate_averaged_increasing_function_by_pp(self, network_data: pd.DataFrame, filename: str):
        pp_uniques = network_data['PP'].unique()

    def __bar(self, names: np.array, values, title: str, filename: str):
        fig = plt.figure(figsize=(20, 10))
        plt.bar(names, values, color='g', width=0.1)
        plt.title(title)
        plt.savefig(f'{self.save_filepath}/{filename}')
        plt.close(fig)
