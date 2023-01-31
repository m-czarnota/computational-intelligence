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

        plt.figure()
        plt.bar(measure_names, averages, color='g')
        plt.title('Average infection by measure')
        plt.savefig(f'{self.save_filepath}/{filename}')

    def generate_averaged_increasing_infection_by_pp(self, filename: str = 'averaged_increasing_infection_by_pp.png'):
        networks = self._file['Network'].unique()
        filepath_to_save = f'{self.save_filepath}/{filename.replace("png", "").replace("jpg", "")}'

        if not os.path.exists(filepath_to_save):
            os.mkdir(filepath_to_save)


    def __read_file(self, filepath: str = None) -> pd.DataFrame:
        return pd.read_csv(self._default_filepath_data if filepath is None else filepath)
