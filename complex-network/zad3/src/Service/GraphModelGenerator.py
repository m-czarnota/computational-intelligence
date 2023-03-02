import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


class GraphModelGenerator:
    def __init__(self, save: bool = True, save_filepath: str = './images/results'):
        self.save = save
        self.save_filepath = save_filepath

        self._default_filepath_data: str = './data/results_main.csv'
        self._file = self.__read_file()

    def generate_mean_coverage_by_network(self, filename: str = 'mean_coverage_by_network.png'):
        network_uniques = self._file['Network'].unique()
        network_uniques = np.array(list(map(lambda name: name.replace('.txt', ''), network_uniques)))
        network_uniques.sort()
        mean_coverage_by_network = self._file.groupby(['Network'], sort=True)['IF'].mean()

        plot_title = 'Mean coverage by network'
        self.__bar(network_uniques, mean_coverage_by_network, plot_title, filename, xlabel='Network', ylabel='Coverage (%)')

    def generate_mean_coverage_by_pp(self, filename: str = 'mean_coverage_by_pp.png'):
        pp_uniques = self._file['PP'].unique()
        pp_uniques = list(map(lambda name: str(name), pp_uniques))
        mean_coverage_by_pp = self._file.groupby(['PP'])['IF'].mean()

        plot_title = 'Mean coverage by propagation probability'
        self.__bar(pp_uniques, mean_coverage_by_pp, plot_title, filename, xlabel='Propagation probability', ylabel='Coverage (%)')

    def generate_mean_coverage_by_sf(self, filename: str = 'mean_coverage_by_sf.png'):
        sf_uniques = self._file['SF'].unique()
        sf_uniques = list(map(lambda name: str(name), sf_uniques))
        mean_coverage_by_sf = self._file.groupby(['SF'])['IF'].mean()

        plot_title = 'Mean coverage by seed fraction'
        self.__bar(sf_uniques, mean_coverage_by_sf, plot_title, filename, xlabel='Seed fraction', ylabel='Coverage (%)')

    def generate_mean_coverage_by_measure(self, filename: str = 'mean_coverage_by_measure.png'):
        measure_uniques = self._file['Measure'].unique()
        mean_coverage_by_measure = self._file.groupby(['Measure'])['IF'].mean()

        plot_title = 'Mean coverage by measure'
        self.__bar(measure_uniques, mean_coverage_by_measure, plot_title, filename, xlabel='Measure', ylabel='Coverage (%)')

    def generate_mean_steps_by_network(self, filename: str = 'mean_steps_by_network.png'):
        network_uniques = self._file['Network'].unique()
        network_uniques = np.array(list(map(lambda name: name.replace('.txt', ''), network_uniques)))
        network_uniques.sort()
        mean_steps_by_network = self._file.groupby(['Network'])['Iterations count'].mean()

        plot_title = 'Mean steps by network'
        self.__bar(network_uniques, mean_steps_by_network, plot_title, filename, xlabel='Network', ylabel='Steps')

    def generate_mean_steps_by_pp(self, filename: str = 'mean_steps_by_pp.png'):
        pp_uniques = self._file['PP'].unique()
        pp_uniques = list(map(lambda name: str(name), pp_uniques))
        mean_steps_by_pp = self._file.groupby(['PP'])['Iterations count'].mean()

        plot_title = 'Mean steps by propagation probability'
        self.__bar(pp_uniques, mean_steps_by_pp, plot_title, filename, xlabel='Propagation probability', ylabel='Steps')

    def generate_mean_steps_by_sf(self, filename: str = 'mean_steps_by_sf.png'):
        sf_uniques = self._file['SF'].unique()
        sf_uniques = list(map(lambda name: str(name), sf_uniques))
        mean_steps_by_sf = self._file.groupby(['SF'])['Iterations count'].mean()

        plot_title = 'Mean steps by seed fraction'
        self.__bar(sf_uniques, mean_steps_by_sf, plot_title, filename, xlabel='Seed fraction', ylabel='Steps')

    def generate_mean_steps_by_measure(self, filename: str = 'mean_steps_by_measure.png'):
        measure_uniques = self._file['Measure'].unique()
        mean_steps_by_measure = self._file.groupby(['Measure'])['Iterations count'].mean()

        plot_title = 'Mean steps by measure'
        self.__bar(measure_uniques, mean_steps_by_measure, plot_title, filename, xlabel='Measure', ylabel='Steps')

    def generate_max_coverage_by_method(self, filename: str = 'max_coverage_by_method.png'):
        max_coverage_by_measure = self._file.groupby(['Network', 'Measure'])['IF'].max()
        names = list(map(lambda name: name.replace('.txt', ''), self._file['Network'].unique()))

        plot_title = 'Max coverage by measure'
        x = np.arange(len(names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(20, 10))
        rects1 = ax.bar(x - width / 0.8, max_coverage_by_measure[::3], width, label='degree')
        rects2 = ax.bar(x, max_coverage_by_measure[1::3], width, label='closeness')
        rects3 = ax.bar(x + width / 0.8, max_coverage_by_measure[2::3], width, label='random')

        ax.set_xticks(x, names)
        ax.set_xlabel('Network Name')
        ax.set_ylabel('Infection coverage (%)')
        ax.legend(loc='lower right')

        ax.bar_label(rects1, padding=5)
        ax.bar_label(rects2, padding=5)
        ax.bar_label(rects3, padding=5)

        ax.set_title(plot_title)
        plt.savefig(f'{self.save_filepath}/{filename}') if self.save else plt.show()
        plt.close(fig)

    def generate_max_coverage_by_sf(self, filename: str = 'max_coverage_by_sf.png'):
        max_coverage_by_measure = self._file.groupby(['Network', 'SF'])['IF'].max()
        names = list(map(lambda name: name.replace('.txt', ''), self._file['Network'].unique()))

        plot_title = 'Max coverage by seed fraction'
        x = np.arange(len(names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(20, 10))
        rects1 = ax.bar(x - width / 0.8, max_coverage_by_measure[::3], width, label='5%')
        rects2 = ax.bar(x, max_coverage_by_measure[1::3], width, label='10%')
        rects3 = ax.bar(x + width / 0.8, max_coverage_by_measure[2::3], width, label='25%')

        ax.set_xticks(x, names)
        ax.set_xlabel('Network Name')
        ax.set_ylabel('Infection coverage (%)')
        ax.legend(loc='lower right')

        ax.bar_label(rects1, padding=5)
        ax.bar_label(rects2, padding=5)
        ax.bar_label(rects3, padding=5)

        ax.set_title(plot_title)
        plt.savefig(f'{self.save_filepath}/{filename}') if self.save else plt.show()
        plt.close(fig)

    def generate_max_coverage_by_method_and_pp(self, filename: str = 'max_coverage_by_method_and_pp.png'):
        grouped_by_network_and_column = self._file.groupby(['Network', 'Measure'])

    def __read_file(self, filepath: str = None) -> pd.DataFrame:
        return pd.read_csv(self._default_filepath_data if filepath is None else filepath)

    def __bar(self, names: np.array, values, title: str, filename: str, **kwargs):
        fig = plt.figure(figsize=(20, 10))
        plt.bar(names, values, color='g', width=0.3 if 'width' not in kwargs.keys() else kwargs['width'])
        plt.title(title)

        if 'xlabel' in kwargs.keys():
            plt.xlabel(kwargs['xlabel'])
        if 'ylabel' in kwargs.keys():
            plt.ylabel(kwargs['ylabel'])

        plt.savefig(f'{self.save_filepath}/{filename}') if self.save else plt.show()
        plt.close(fig)
