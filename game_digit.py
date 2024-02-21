import pygame
import sys
import numpy as np
from keras.models import load_model
import cv2
from pygame.locals import *

# dane wyglądu okna
BOUNDRYINC = 5  # obszar wokół prostokąta
WINDOWSIZEX = 640
WINDOWSIZEY = 480

# kolory
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

iswriting = False
predict = True
image_cnt = 1
imagesave = False

number_xcord = []
number_ycord = []

model = load_model('bestmodel.h5')
labels = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}

pygame.init()
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption('Digit Board')  # ustawienie tytułu okna
font = pygame.font.SysFont('arial', 18)

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()  # zamyka silnik gry
            sys.exit()  # kończy cały program

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, white, (xcord, ycord), 5, 0)  # 5 to grubość lini,
            # chodzi o kółko czyli jakby każde kliknięcie to kółko jak się cały czas ruszamy to tysiące kółek \
            # rysują liczby

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:

            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0), min(WINDOWSIZEX,
                                                                               number_xcord[-1] + BOUNDRYINC)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDRYINC, 0), min(number_ycord[-1] + BOUNDRYINC,
                                                                               WINDOWSIZEY)

            number_xcord = []
            number_ycord = []

            pygame.draw.rect(DISPLAYSURF, red, (rect_min_x, rect_min_y, rect_max_x - rect_min_x,
                                                rect_max_y - rect_min_y), 2)  # rysowanie prostokąta/obramówki

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if imagesave:
                cv2.imwrite(f'image{image_cnt}.png', img_arr)
                image_cnt += 1

            if predict:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)  # dodanie marginesu do tego pobranego
                # obrazku, margines o wartości 10
                image = cv2.resize(image, (28, 28)) / 255

                label = str(labels[np.argmax(model.predict(image.reshape(1, 28, 28, 1)))])

                textSurface = font.render(label, True, red, white)
                textRecObj = textSurface.get_rect()
                textRecObj.left, textRecObj.bottom = rect_max_x, rect_max_y

                DISPLAYSURF.blit(textSurface, textRecObj)

        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:  # jak nacisne sobie przycisk esc to mi się wyczyści tablica
                DISPLAYSURF.fill(black)

    pygame.display.update()
