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


def decode_by_interpolation(coded_image: np.array) -> np.array:
    decoded = np.zeros((*coded_image.shape, 3))
    pixels_to_skip_on_borders = 1  # skip 1 pixel on borders to make algorithm easier

    for r in range(pixels_to_skip_on_borders, coded_image.shape[0] - pixels_to_skip_on_borders):
        row = coded_image[r]

        for c in range(pixels_to_skip_on_borders, coded_image.shape[1] - pixels_to_skip_on_borders):
            encoded_value = row[c]
            actual_rgb_component = CFA_FILTER[r % CFA_FILTER.shape[0], c % CFA_FILTER.shape[1]]
            decoded_color = np.zeros(3)  # (R, G, B)

            if actual_rgb_component == 'R':
                green = np.mean([coded_image[r - 1, c], coded_image[r, c - 1], coded_image[r, c + 1], coded_image[r + 1, c]])
                blue = np.mean([coded_image[r - 1, c - 1], coded_image[r - 1, c + 1], coded_image[r + 1, c - 1], coded_image[r + 1, c + 1]])

                decoded_color = np.array([encoded_value, green, blue])
            elif actual_rgb_component == 'B':
                red = np.mean([coded_image[r - 1, c - 1], coded_image[r - 1, c + 1], coded_image[r + 1, c - 1], coded_image[r + 1, c + 1]])
                green = np.mean([coded_image[r - 1, c], coded_image[r, c - 1], coded_image[r, c + 1], coded_image[r + 1, c]])

                decoded_color = np.array([red, green, encoded_value])
            else:
                if CFA_FILTER[r % CFA_FILTER.shape[0], (c + 1) % CFA_FILTER.shape[1]] == 'R' and CFA_FILTER[(r + 1) % CFA_FILTER.shape[0], c % CFA_FILTER.shape[1]] == 'B':
                    red = np.mean([coded_image[r, c - 1], coded_image[r, c + 1]])
                    blue = np.mean([coded_image[r - 1, c], coded_image[r + 1, c]])

                    decoded_color = np.array([red, encoded_value, blue])
                elif CFA_FILTER[(r + 1) % CFA_FILTER.shape[0], c % CFA_FILTER.shape[1]] == 'R' and CFA_FILTER[r % CFA_FILTER.shape[0], (c + 1) % CFA_FILTER.shape[1]] == 'B':
                    red = np.mean([coded_image[r - 1, c], coded_image[r + 1, c]])
                    blue = np.mean([coded_image[r, c - 1], coded_image[r, c + 1]])

                    decoded_color = np.array([red, encoded_value, blue])

            decoded[r, c] = decoded_color

    decoded = decoded / np.amax(decoded)
    decoded = np.clip(decoded, 0, 1)

    return decoded


def decode_by_malvar(coded_image: np.array) -> np.array:
    decoded = np.zeros((*coded_image.shape, 3))
    pixels_to_skip_on_borders = 2  # skip 2 pixel on borders to make algorithm easier

    for r in range(pixels_to_skip_on_borders, coded_image.shape[0] - pixels_to_skip_on_borders):
        row = coded_image[r]

        for c in range(pixels_to_skip_on_borders, coded_image.shape[1] - pixels_to_skip_on_borders):
            encoded_value = row[c]
            decoded_color = np.zeros(3)  # (R, G, B)
            actual_rgb_component = CFA_FILTER[r % CFA_FILTER.shape[0], c % CFA_FILTER.shape[1]]

            if actual_rgb_component == 'R':
                green = (2 * coded_image[r - 1, c] + 2 * coded_image[r, c - 1] + 2 * coded_image[r, c + 1] + 2 * coded_image[r + 1, c] + 4 * encoded_value + -1 * coded_image[r - 2, c] + -1 * coded_image[r, c - 2] + -1 * coded_image[r, c + 2] + -1 * coded_image[r + 2, c]) / 8
                blue = (2 * coded_image[r - 1, c - 1] + 2 * coded_image[r - 1, c + 1] + 2 * coded_image[r + 1, c - 1] + 2 * coded_image[r + 1, c + 1] + 6 * encoded_value + -3/2 * coded_image[r - 2, c] + -3/2 * coded_image[r, c - 2] + -3/2 * coded_image[r, c + 2] + -3/2 * coded_image[r + 2, c]) / 8

                decoded_color = np.array([encoded_value, green, blue])
            elif actual_rgb_component == 'B':
                green = (2 * coded_image[r - 1, c] + 2 * coded_image[r, c - 1] + 2 * coded_image[r, c + 1] + 2 * coded_image[r + 1, c] + 4 * encoded_value + -1 * coded_image[r - 2, c] + -1 * coded_image[r, c - 2] + -1 * coded_image[r, c + 2] + -1 * coded_image[r + 2, c]) / 8
                red = (2 * coded_image[r - 1, c - 1] + 2 * coded_image[r - 1, c + 1] + 2 * coded_image[r + 1, c - 1] + 2 * coded_image[r + 1, c + 1] + 6 * encoded_value + -3/2 * coded_image[r - 2, c] + -3/2 * coded_image[r, c - 2] + -3/2 * coded_image[r, c + 2] + -3/2 * coded_image[r + 2, c]) / 8

                decoded_color = np.array([red, green, encoded_value])
            else:
                if CFA_FILTER[r % CFA_FILTER.shape[0], (c + 1) % CFA_FILTER.shape[1]] == 'R' and CFA_FILTER[(r + 1) % CFA_FILTER.shape[0], c % CFA_FILTER.shape[1]] == 'B':
                    red = (4 * coded_image[r, c - 1] + 4 * coded_image[r, c + 1] + 5 * encoded_value + 1/2 * coded_image[r - 2, c] + -1 * coded_image[r - 1, c - 1] + -1 * coded_image[r - 1, c + 1] + -1 * coded_image[r, c - 2] + -1 * coded_image[r, c + 2] + -1 * coded_image[r + 1, c - 1] + -1 * coded_image[r + 1, c + 1] + 1/2 * coded_image[r + 2, c]) / 8
                    blue = (4 * coded_image[r - 1, c] + 4 * coded_image[r + 1, c] + 5 * encoded_value + -1 * coded_image[r - 2, c] + -1 * coded_image[r - 1, c - 1] + -1 * coded_image[r - 1, c + 1] + 1/2 * coded_image[r, c - 2] + 1/2 * coded_image[r, c + 2] + -1 * coded_image[r + 1, c - 1] + -1 * coded_image[r + 1, c + 1] + -1 * coded_image[r + 2, c]) / 8

                    decoded_color = np.array([red, encoded_value, blue])
                elif CFA_FILTER[(r + 1) % CFA_FILTER.shape[0], c % CFA_FILTER.shape[1]] == 'R' and CFA_FILTER[r % CFA_FILTER.shape[0], (c + 1) % CFA_FILTER.shape[1]] == 'B':
                    blue = (4 * coded_image[r, c - 1] + 4 * coded_image[r, c + 1] + 5 * encoded_value + 1/2 * coded_image[r - 2, c] + -1 * coded_image[r - 1, c - 1] + -1 * coded_image[r - 1, c + 1] + -1 * coded_image[r, c - 2] + -1 * coded_image[r, c + 2] + -1 * coded_image[r + 1, c - 1] + -1 * coded_image[r + 1, c + 1] + 1/2 * coded_image[r + 2, c]) / 8
                    red = (4 * coded_image[r - 1, c] + 4 * coded_image[r + 1, c] + 5 * encoded_value + -1 * coded_image[r - 2, c] + -1 * coded_image[r - 1, c - 1] + -1 * coded_image[r - 1, c + 1] + 1/2 * coded_image[r, c - 2] + 1/2 * coded_image[r, c + 2] + -1 * coded_image[r + 1, c - 1] + -1 * coded_image[r + 1, c + 1] + -1 * coded_image[r + 2, c]) / 8

                    decoded_color = np.array([red, encoded_value, blue])

            decoded[r, c] = decoded_color

    decoded = decoded / np.amax(decoded)
    decoded = np.clip(decoded, 0, 1)

    return decoded


def mse(image1: np.array, image2: np.array) -> float:
    error = 0

    for r in range(0, image1.shape[0]):
        for c in range(0, image2.shape[0]):
            error += np.sum(np.power(image1[r, c] - image2[r, c], 2))

    return error / (image1.shape[0] * image1.shape[1])


def experiment(filepath: str, save: bool = False) -> None:
    filename = filepath.split('/')[-1].split('.')[0]

    image = plt.imread(filepath)
    coded_image = code_image(image)

    if not save:
        print('Original image')
        plt.figure(figsize=(20, 10))
        plt.imshow(image)
        plt.show()

        print('Coded image')
        plt.figure(figsize=(20, 10))
        plt.imshow(coded_image)
        plt.show()
    else:
        plt.figure()
        plt.imsave(f'{IMAGES_DIR}/coded_image_{filename}.png', coded_image)

    decoded_image_bilinear = decode_by_interpolation(coded_image)
    mse_error = mse(image, decoded_image_bilinear)
    print(f'Mean squared error for bilinear interpolation: {mse_error}')

    if not save:
        print('Original image')
        plt.figure(figsize=(20, 10))
        plt.imshow(image)
        plt.show()

        print('Decoded image by bilinear interpolation')
        plt.figure(figsize=(20, 10))
        plt.imshow(decoded_image_bilinear)
        plt.show()
    else:
        plt.figure()
        plt.imsave(f'{IMAGES_DIR}/demosaiced_bilinear_{filename}.png', decoded_image_bilinear)

    plt.figure(figsize=(20, 10))
    if not save:
        print('Original image in zoom')
        plt.figure(figsize=(20, 10))
        plt.imshow(image[200:600, 200:600])
        plt.show()

        print('Decoded image by bilinear interpolation in zoom')
        plt.figure(figsize=(20, 10))
        plt.imshow(decoded_image_bilinear[200:600, 200:600])
        plt.show()
    else:
        plt.figure()
        plt.imsave(f'{IMAGES_DIR}/demosaiced_bilinear_zoom_{filename}.png', decoded_image_bilinear[200:600, 200:600])

    decoded_image_malvar = decode_by_malvar(coded_image)
    mse_error = mse(image, decoded_image_malvar)
    print(f'Mean squared error for malvar interpolation: {mse_error}')

    if not save:
        print('Original image')
        plt.figure(figsize=(20, 10))
        plt.imshow(image)
        plt.show()

        print('Decoded image by malvar interpolation')
        plt.figure(figsize=(20, 10))
        plt.imshow(decoded_image_malvar)
        plt.show()
    else:
        plt.figure()
        plt.imsave(f'{IMAGES_DIR}/demosaiced_malvar_{filename}.png', decoded_image_malvar)

    if not save:
        print('Original image in zoom')
        plt.figure(figsize=(20, 10))
        plt.imshow(image[200:600, 200:600])
        plt.show()

        print('Decoded image by malvar interpolation in zoom')
        plt.figure(figsize=(20, 10))
        plt.imshow(decoded_image_malvar[200:600, 200:600])
        plt.show()
    else:
        plt.figure()
        plt.imsave(f'{IMAGES_DIR}/demosaiced_malvar_zoom_{filename}.png', decoded_image_malvar[200:600, 200:600])


if __name__ == '__main__':
    file_paths = [
        f'{IMAGES_DIR}/real1.jpg',
        f'{IMAGES_DIR}/real2.jpg',
        f'{IMAGES_DIR}/real3.jpg',
    ]

    for file_path in file_paths:
        experiment(file_path, True)

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