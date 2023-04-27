import numpy as np


def transform_data_to_db_scale(data: np.array) -> np.array:
    data_max = np.max(data)

    return 10 * np.log10(data / data_max)
