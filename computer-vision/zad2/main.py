import numpy as np
from matplotlib import pyplot as plt

IMAGES_DIR = './images'
CFA_FILTER = np.array([['G', 'R'], ['B', 'G']])


def map_cfa_filter_val_to_rgb_index(value: str) -> int:
    if value == 'G':
        return 1

    if value == 'R':
        return 0

    return 2


def code_image(image: np.array) -> np.array:
    coded_image = np.empty((image.shape[0], image.shape[1]))

    for row_iter, row in enumerate(image):
        cfa_row = row_iter % CFA_FILTER.shape[0]

        for color_iter, color in enumerate(row):
            cfa_column = color_iter % CFA_FILTER.shape[1]

            color_to_select = map_cfa_filter_val_to_rgb_index(CFA_FILTER[cfa_row, cfa_column])
            selected_color = color[color_to_select]
            coded_image[row_iter, color_iter] = selected_color

    return coded_image


def interpolate_colors_in_image() -> np.array:
    color = np.empty(3)


def decode_by_interpolation(coded_image: np.array) -> np.array:
    decoded = np.array((*coded_image.shape, 3))
    pixels_to_skip_on_borders = 1

    for cfa_row in range(pixels_to_skip_on_borders, coded_image.shape[0] - pixels_to_skip_on_borders):
        row = coded_image[cfa_row]
        cfa_row = cfa_row % CFA_FILTER.shape[0]

        for value_iter, value in enumerate(row[pixels_to_skip_on_borders:-pixels_to_skip_on_borders]):
            cfa_column = value_iter % CFA_FILTER.shape[1]
            color_to_select = map_cfa_filter_val_to_rgb_index(CFA_FILTER[cfa_row, cfa_column])

            for window_row in coded_image[cfa_row - pixels_to_skip_on_borders:cfa_row + pixels_to_skip_on_borders]:
                for window_column in coded_image[cfa_column - pixels_to_skip_on_borders:cfa_column + pixels_to_skip_on_borders]:
                    pass


if __name__ == '__main__':
    image = plt.imread(f'{IMAGES_DIR}/real1.jpg')
    coded_image = code_image(image)

    # plt.figure()
    # plt.imshow(coded_image)
    # plt.show()

"""
przyjąć zwykłe zdjęcie - rzeczywisty świat
3-wymiarowa macierz
chcemy zrobić 2-wymiarową macierz z takymi samymi wymiarami

metoda 1:
    kodowanie:
        mamy macierz
        G R
        B G
        wybieramy po koleii kolory widąć takim oknem przesuwnym macierzą
        G R G R G R
        B G B G B G
        G R G R G R
        B G B G B G
    
    dekodowanie:
        bierzemy dla każdego piksela otoczenia
        mamy okienko 3x3
        dla piksela R będzie to średnia dla wszystkich składowych R w tym okienku
        dla B tak samo, dla G tak samo
        jeżeli piksel jest w środku to przepisujemy wartość bezpośrednio
        
metoda 2:
    wyszukać na necie
    nie mamy najbliższego sąsiedztwa 3x3, zazwyczaj jest brane 4x4
    
        R           R
        G           G
    R G R G R   B G B G B
        G           G
        R           B
    
    jak chcemy policzyć w tym drugim wartość zielonego G
    to bierzemy te naokoło wszystkie G z wagą 2
    i w środku ten B z wagą 4, a na zewnątrz B mają wagi -1

porównanie:
    to co widzimy
    można zastosować szumy, msr
"""