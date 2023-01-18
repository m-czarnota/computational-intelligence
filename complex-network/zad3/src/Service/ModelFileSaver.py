import pandas as pd


class ModelFileSaver:
    def __init__(self):
        self._default_filepath = './data/model_results'

    def save(self, filename: str, data: dict, filepath: str = None):
        if filepath is None:
            filepath = self._default_filepath

        df = self._create_data_frame_from_data(data)
        df.to_csv(f'{filepath}/{filename}')

    def _create_data_frame_from_data(self, data: dict) -> pd.DataFrame:
        df = pd.DataFrame(columns=data.keys())
        data_count = len(data[list(data.keys())[0]])

        for data_iter in range(data_count):
            values = {key: None for key in data.keys()}

            for key in data.keys():
                value = data[key][data_iter]
                value = self._prepare_value_to_save(value)

                values[key] = value

            series = pd.Series(values)
            df = pd.concat([df, series.to_frame().T], ignore_index=True)

        return df

    @staticmethod
    def _prepare_value_to_save(value):
        if type(value) == list:
            return ','.join(value)

        return value
