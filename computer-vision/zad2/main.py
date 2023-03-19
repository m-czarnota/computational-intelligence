import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy import signal
import cv2

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

    for encoded_row_iter in range(pixels_to_skip_on_borders, coded_image.shape[0] - pixels_to_skip_on_borders):
        row = coded_image[encoded_row_iter]

        for encoded_column_iter in range(pixels_to_skip_on_borders, coded_image.shape[1] - pixels_to_skip_on_borders):
            encoded_value = row[encoded_column_iter]
            decoded_color = np.zeros(3)  # (R, G, B)
            decoded_color_counts = np.zeros(3)  # to divide sum to get mean

            # +2 because we need excluded as third value
            for window_row_iter in range(encoded_row_iter - 1, encoded_row_iter + 2):
                for window_column_iter in range(encoded_column_iter - 1, encoded_column_iter + 2):
                    # we consider only neighbours
                    if window_row_iter == encoded_row_iter and window_column_iter == encoded_column_iter:
                        continue

                    # we need to know if considered value is R or G or B component
                    cfa_row = window_row_iter % CFA_FILTER.shape[0]
                    cfa_column = window_column_iter % CFA_FILTER.shape[1]
                    cfa_color_to_select_index = map_cfa_filter_val_to_rgb_index(CFA_FILTER[cfa_row, cfa_column])

                    # add values to sum and increment counter
                    decoded_color[cfa_color_to_select_index] += encoded_value
                    decoded_color_counts[cfa_color_to_select_index] += 1

            for decoded_color_iter, decoded_color_component in enumerate(decoded_color):
                color_count = decoded_color_counts[decoded_color_iter]
                if color_count <= 0:
                    continue

                decoded_color[decoded_color_iter] /= color_count
            decoded[encoded_row_iter, encoded_column_iter] = decoded_color

    decoded = decoded / np.amax(decoded)
    decoded = np.clip(decoded, 0, 1)

    return decoded


def decodee_by_malvar(coded_image: np.array) -> np.array:
    decoded = np.zeros((*coded_image.shape, 3))
    pixels_to_skip_on_borders = 2  # skip 2 pixel on borders to make algorithm easier

    for r in range(pixels_to_skip_on_borders, coded_image.shape[0] - pixels_to_skip_on_borders):
        row = coded_image[r]

        for c in range(pixels_to_skip_on_borders, coded_image.shape[1] - pixels_to_skip_on_borders):
            encoded_value = row[c]
            decoded_color = np.zeros(3)  # (R, G, B)
            actual_rgb_component = CFA_FILTER[r % CFA_FILTER.shape[0], c % CFA_FILTER.shape[1]]

            if actual_rgb_component == 'R':
                red_for_green = None
                green = (2 * coded_image[r - 1, c] + 2 * coded_image[r, c - 1] + 2 * coded_image[r, c + 1] + 2 * coded_image[r + 1, c]) / 8
                blue = (2 * coded_image[r - 1, c - 1] + 2 * coded_image[r - 1, c + 1] + 2 * coded_image[r + 1, c - 1], + 2 * coded_image[r + 1, c + 1])



def demosaic_bilinear(img_raw):
    """
    Perform demosaicing of an image with bilinear interpolation.

    Args:
    - img_raw: a 2D numpy array containing the raw Bayer pattern.

    Returns:
    - a 2D numpy array containing the demosaiced image.
    """
    # Create an empty 3D numpy array to hold the demosaiced image.
    h, w = img_raw.shape
    img_demosaic = np.zeros((h, w, 3))

    # Perform bilinear interpolation for each channel.
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Get the color channels of the neighboring pixels.
            if i == 0:
                up = img_raw[i, j]
            else:
                up = img_raw[i - 1, j]
            if i == h - 1:
                down = img_raw[i, j]
            else:
                down = img_raw[i + 1, j]
            if j == 0:
                left = img_raw[i, j]
            else:
                left = img_raw[i, j - 1]
            if j == w - 1:
                right = img_raw[i, j]
            else:
                right = img_raw[i, j + 1]

            # Compute the color values of the current pixel using bilinear interpolation.
            if (i + j) % 2 == 0:
                # Red pixel.
                img_demosaic[i, j, 0] = img_raw[i, j]
                img_demosaic[i, j, 1] = 0.25 * (up + down + left + right)
                if i % 2 == 0:
                    img_demosaic[i, j, 2] = 0.25 * (img_raw[i - 1, j - 1] + img_raw[i + 1, j + 1] + img_raw[i - 1, j + 1] + img_raw[i + 1, j - 1])
                else:
                    img_demosaic[i, j, 2] = 0.25 * (img_raw[i - 1, j + 1] + img_raw[i + 1, j - 1] + img_raw[i - 1, j - 1] + img_raw[i + 1, j + 1])
            elif i % 2 == 0:
                # Green pixel in even row.
                img_demosaic[i, j, 0] = 0.5 * (left + right)
                img_demosaic[i, j, 1] = img_raw[i, j]
                img_demosaic[i, j, 2] = 0.5 * (up + down)
            else:
                # Green pixel in odd row.
                img_demosaic[i, j, 0] = 0.5 * (up + down)
                img_demosaic[i, j, 1] = img_raw[i, j]
                img_demosaic[i, j, 2] = 0.5 * (left + right)

    return img_demosaic.astype(np.uint8)


def demosaic_mhc(img_raw):
    """
    Perform demosaicing of an image with the Malvar-He-Cutler algorithm.

    Args:
    - img_raw: a 2D numpy array containing the raw Bayer pattern.

    Returns:
    - a 2D numpy array containing the demosaiced image.
    """
    # Create an empty 3D numpy array to hold the demosaiced image.
    h, w = img_raw.shape
    img_demosaic = np.zeros((h, w, 3))

    # Compute the green channel using the horizontal and vertical gradients.
    kernel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_v = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    green_h = signal.convolve2d(img_raw, kernel_h, mode='same', boundary='symm')
    green_v = signal.convolve2d(img_raw, kernel_v, mode='same', boundary='symm')
    img_demosaic[:, :, 1] = 0.5 * (green_h[1:-1:2, 2:-1:2] + green_h[1:-1:2, :-2:2] +
                                   green_v[2:-1:2, 1:-1:2] + green_v[:-2:2, 1:-1:2])

    # Compute the red and blue channels using the interpolated green channel.
    kernel_g = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4
    kernel_rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4
    img_demosaic[1:-1:2, 2:-1:2, 0] = img_raw[1:-1:2, 2:-1:2] - 0.5 * (img_demosaic[1:-1:2, 1:-2:2, 1] + img_demosaic[1:-1:2, 3::2, 1])
    img_demosaic[2:-1:2, 1:-1:2, 0] = img_raw[2:-1:2, 1:-1:2] - 0.5 * (img_demosaic[:-2:2, 1:-1:2, 1] + img_demosaic[3::2, 1:-1:2, 1])
    img_demosaic[1:-1:2, 1:-1:2, 0] = (img_raw[1:-1:2, 1:-1:2] -
                                       0.25 * (img_demosaic[2::2, 1:-1:2, 1] + img_demosaic[:-2:2, 1:-1:2, 1] +
                                               img_demosaic[1:-1:2, 2::2, 1] + img_demosaic[1:-1:2, :-2:2, 1]))
    img_demosaic[:,:, 2] = img_raw[:, :] - img_demosaic[:, :, 1] - img_demosaic[:, :, 0]

    # Convert the pixel values to the range [0, 255] and clamp them to avoid overflow.
    img_demosaic = np.clip(img_demosaic * 255, 0, 255).astype(np.uint8)

    return img_demosaic


if __name__ == '__main__':
    image = plt.imread(f'{IMAGES_DIR}/real1.jpg')
    coded_image = code_image(image)
    # decoded_image = decode_by_interpolation(coded_image)
    decoded_image = cv2.demosaicing(coded_image.astype(np.uint8), cv2.COLOR_BayerGR2BGR)

    plt.figure()
    # plt.imshow(coded_image.astype(np.uint8))
    # plt.imshow(decoded_image.astype(np.uint8))
    plt.imsave(f'{IMAGES_DIR}/decoded_image_real1.png', decoded_image)
    # plt.show()
    plt.close()

    decoded_image = cv2.demosaicing(coded_image.astype(np.uint8), cv2.COLOR_BayerRG2BGR_MHT)
    plt.figure()
    # plt.imshow(coded_image.astype(np.uint8))
    # plt.imshow(decoded_image.astype(np.uint8))
    plt.imsave(f'{IMAGES_DIR}/decoded_image_malvar_real1.png', decoded_image)
    # plt.show()
    plt.close()

    # img_demosaic = demosaic_bilinear(coded_image)  # Demosaic the image with bilinear interpolation.
    # Image.fromarray(img_demosaic).save(f'{IMAGES_DIR}/demosaiced_bilinear_image.png')  # Save the demosaiced image.

    # img_demosaic = demosaic_mhc(coded_image)  # Demosaic the image with the MHC algorithm.
    # Image.fromarray(img_demosaic).save('demosaiced_malvar_image.png')  # Save the demosaiced image.

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