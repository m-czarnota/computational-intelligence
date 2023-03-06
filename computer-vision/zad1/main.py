import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def frame_george_bush() -> np.array:
    filenames = [x[2] for x in os.walk('George_W_Bush')][0]
    images = [cv2.imread(f'George_W_Bush/{filename}') for filename in filenames]

    for image_iter, image in enumerate(images):
        framed_image = image[90:170, 90:170]

        images[image_iter] = framed_image
        cv2.imwrite(f'George_W_bush_framed/{filenames[image_iter]}', framed_image)

    return np.array(images)


def zad1():
    images = frame_george_bush()

    cb_by_image = np.empty(images.shape[0])
    cr_by_image = np.empty(images.shape[0])

    for image_iter, image in enumerate(images):
        y_cb_cr_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        cb_by_image[image_iter] = np.mean(y_cb_cr_image[:, :, 1])
        cr_by_image[image_iter] = np.mean(y_cb_cr_image[:, :, 2])

    hist_cb = cv2.calcHist(images, [1], None, [256], [0, 256])
    hist_cr = cv2.calcHist(images, [2], None, [256], [0, 256])

    plt.figure()
    plt.title('Histograms')
    plt.plot(hist_cb, label='Cb')
    plt.plot(hist_cr, label='Cr')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    zad1()

"""
zad1:
    wykadrować głowe typka
    pracujemy na Cb i Cr
    
    z piku wybieramy pewien przedział, który będzie miał stałą długość
    przedział od maximum (max - 1/2x; max + 1/2x>
    rozkład kolorów na wszytkich tych obrazach
    
    kiedy mamy już to policzone
    wpuszczamy zdjęcie innej osoby, które nie będzie wykadrowane
    wyszukujemy piksele, w które są w przedziale powyżej. i to prawdopodobnie te piksele są skórą
    
    badać szerokość zakresu
    jak będzie za mały to będą dziury w twarzy, nie cała twarz będzie wykryta
    jak za duży to wykryje też ubrania
    
    można zrobić w rgb, ale model będzie jeszcze bardziej wrażliwy na przekłamania

zad2:
    obrazy są wykadrowane i są w odcieniach szarości
    nałożyć na siebie wszystkie te zdjęcia, aby uzyskać średnią informacje
    jak nałożymy te zdjęcia na siebie to dostaniemy mniej więcej kształt jak powinna wyglądać osoba
    zagregowane cechy wielu osób
    
    po nałożeniu tych orazków w 1 powstaje taki template
    przechodzimy po obrazie testowym jakimś okienkiem i sprawdzamy czy dane okienko pasuje do template
    jak tak, to jest tam człowiek
    jak nie to nie ma tam człowieka
    w opencv są funkcje gotowe do template matchingu
"""
