import pygame, sys  
from pygame.locals import * 
import numpy as np
from keras.models import load_model
import cv2
from pathlib import Path


# customising pygame window size
WINDOWSIZEX = 640 
WINDOWSIZEY = 480
windowsize   = (WINDOWSIZEX, WINDOWSIZEY)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)

IMAGESAVE = False  #set to True if you want to save the drawn images to disk for later use
MODEL = load_model("AI&ML//handwritten digit recognision//cnn attempt 1//best_model.keras") #load the trained model

PREDICT = True  #set to True if you want to make predictions using the trained model

#dictionary to map the numeric class labels to their corresponding string representations
LABELS = {0:'Zero', 1:'One', 2:'Two', 3:'Three', 4:'Four', 5:'Five', 6:'Six', 7:'Seven', 8:'Eight', 9:'Nine'} # used to conver numeric model output to text labels



# Initialize Pygame
pygame.init()

# Set up the drawing window
FONT = pygame.font.Font("freesansbold.ttf", 15)
DISPLAYSURFACE = pygame.display.set_mode(windowsize)
pygame.display.set_caption('Handwritten Digit Recognition Board')

iswriting = False #indicates whether the user is currently drawing on the canvas
number_xcord = [] #list to store the x-coordinates of the points where the user has drawn
number_ycord = [] 

BOUNDRYINC = 5 #boundary increase value to add some padding around the drawn digit when cropping the image for prediction
img_cnt = 0 #counter to keep track of the number of images saved (if IMAGESAVE is True)
ing_array = np.zeros((480, 640)) #initialize a blank image array to store the pixel data of the drawn digit


while True:
    # handles quitting the pygame window
    for event in pygame.event.get():
        if event.type == QUIT:  #quit pygame window if user clicks the 'X' button
            pygame.quit()
            sys.exit()

        # handle mouse events for drawing and making predictions
        if event.type == MOUSEMOTION and iswriting: 
            xcord, ycord = event.pos 
            pygame.draw.circle(DISPLAYSURFACE, WHITE, (xcord, ycord), 3, 0) 

            number_xcord.append(xcord) 
            number_ycord.append(ycord) 

        if event.type == MOUSEBUTTONDOWN: 
            iswriting = True 

        if event.type == MOUSEBUTTONUP: 
            iswriting = False 
            number_xcord = sorted(number_xcord) 
            number_ycord = sorted(number_ycord) 

            #determine the min and max x/y coordinates of the bounding box, with some boundary increase
            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDRYINC) 
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDRYINC, 0), min(WINDOWSIZEY, number_ycord[-1] + BOUNDRYINC)

            #reset the coordinates list for the next drawing
            number_xcord = [] 
            number_ycord = []

            pygame.draw.rect(DISPLAYSURFACE, RED, pygame.Rect(rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2) #draw a rectangle around the drawn digit using the calculated bounding box coordinates

            ing_array = np.array(pygame.PixelArray(DISPLAYSURFACE))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32) #cropping the image to the bounding box of the drawn digit and transposing it to match the expected format

            if IMAGESAVE:
                cv2.imwrite("input_image.png") #save the cropped image to disk for later use
                img_cnt += 1

            if PREDICT:
                image = cv2.resize(ing_array, (28, 28)) 
                image = np.pad(image, (10,10), 'constant', constant_values = 0) 
                image = cv2.resize(image, (28, 28)) / 255  

                #reshape the image to match the model's input shape and make a prediction using the trained model.
                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))] )                                                                               
                #render the predicted label as a text surface using the specified font and colors
                textSurface = FONT.render(label, True, RED, WHITE) 
                textRectObj = textSurface.get_rect()
                #position the text surface at the top-left corner of the bounding box of the drawn digit
                textRectObj.left = rect_min_x 
                textRectObj.bottom = rect_min_y

                DISPLAYSURFACE.blit(textSurface, textRectObj) 

        # handle keyboard event for clearing the canvas
        if event.type == KEYDOWN:
            if event.unicode == 'e':
                DISPLAYSURFACE.fill(BLACK) 

        pygame.display.update() #update the Pygame display to reflect any changes made to the display surface