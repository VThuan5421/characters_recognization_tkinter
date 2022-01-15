import tkinter as tk
from tkinter import *
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
import numpy as np
from PIL import EpsImagePlugin
EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs9.55.0\bin\gswin64c'

# load the saved model
model_digit = load_model("model_digit.h5")
model_alphabet = load_model("model_alphabet.h5")
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
    10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',
    20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
image_number = 0
lastx, lasty = None, None # the position of the mouse pointer
# Create Graphics User Interface
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        # Title
        self.label = tk.Label(self, text = "Characters Recognization", justify=CENTER, 
            font = ("Arial", 20, "italic"), fg = "purple")
        self.label.grid(row = 0, column = 0, columnspan = 3, pady = 2, padx = 2)
        # Canvas
        self.canvas = tk.Canvas(self, width = 500, height = 500, bg = "white")
        self.canvas.bind("<Button-1>", self.activate_event)
        self.canvas.grid(row = 1, column = 0, columnspan = 3, pady = 2, padx = 2, sticky = W)
        # Label result
        #self.result = tk.Label(self, text = "", fg = "#fff", font = ("Arial", 12, 'bold'), bg = "#606060", width = 48, height = 3)
        #self.result.grid(row = 2, column = 0, columnspan = 3, padx = 2, pady = 2)
        # Create button
        self.btn_digit = tk.Button(self, text = "Recognize digit", bg = "cyan", font = ("Arial", 10, "bold"),
            width = 18, command = self.recognize_digit)
        self.btn_digit.grid(row = 3, column = 0, pady = 1, padx = 1)
        self.btn_alphabet = tk.Button(self, text = "Recognize alphabet", bg = "cyan", font = ("Arial", 10, "bold"),
            width = 18, command = self.recognize_alphabet)
        self.btn_alphabet.grid(row = 3, column = 1, pady = 1, padx = 1)
        self.btn_clear = tk.Button(self, text = "Clear widget", bg = "cyan", font = ("Arial", 10, "bold"),
            width = 18, command = self.clear_canvas)
        self.btn_clear.grid(row = 3, column = 2, pady = 1, padx = 1)

    def activate_event(self, event):
        global lastx, lasty
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        lastx, lasty = event.x, event.y

    def draw_lines(self, event):
        global lastx, lasty
        x, y = event.x, event.y
        # Do the canvas drawings
        self.canvas.create_line((lastx, lasty, x, y), width = 10, fill = 'black',
            capstyle = ROUND, smooth = True, splinesteps = 12)
        lastx, lasty = x, y

    def clear_canvas(self):
        self.canvas.delete("all")
    
    def show_image(image):
        plt.imshow(image)
        plt.title("Image")
        plt.show()

    def save_image_from_eps(self):
        global image_number
        #image_number += 1
        filename = f"./image/{image_number}.png"
        self.canvas.update()
        self.canvas.postscript(file = "./image/number.ps")
        img = Image.open("./image/number.ps")
        img.save(filename)
        return filename
    
    def make_image_to_predict(self, is_digit):
        filename = self.save_image_from_eps()
        results = []
        accuracy = []
        # Read the image in color format
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        # Convert the image to grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply thresholding before find contours
        ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # findContours: function helps in extracting the contours from the image
        contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for cnt in contours:
            # Get bounding box extract ROI
            x, y, w, h = cv2.boundingRect(cnt)
            # Draw rectangle of each contour in the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
            # expand the border of the contour image
            top = int(0.05 * th.shape[0])
            bottom = top
            left = int(0.05 * th.shape[1])
            right = left
            th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)
            # Extract roi and resize image to 28x28 pixels
            roi = th[y - top: y + h + bottom, x - left: x + w + right]
            img = cv2.resize(roi, (28, 28), cv2.INTER_AREA)
            # Reshaping the img to support our model input
            img = img.reshape(1, 28, 28, 1)
            # For putText
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (0, 0, 255)
            thickness = 1
            # digit prediction
            if is_digit:
                # normalizing the img to support our model input
                img = img / 255.0
                # now, it's time to predict
                pred = model_digit.predict([img])[0]
                # np.argmax(array) return the indices of the max value in array
                final_pred = np.argmax(pred)
                data = str(final_pred) + " " + str(int(max(pred) * 100)) + "%"
                results.append(data)
                # cv2.putText() to draw text in the image
                cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)
            # aphabet prediction
            else:
                pred = model_alphabet.predict([img])[0]
                final_pred = np.argmax(pred)
                data = word_dict[final_pred] + " " + str(int(max(pred) * 100)) + "%"
                results.append(data)
                cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)
        return image, results

    def recognize_digit(self):
        image, results = self.make_image_to_predict(is_digit = True)
        #self.result.configure(text = " ".join(results))
        cv2.imshow('Digits Image', image)
        cv2.waitKey(0)
        self.clear_canvas()

    def recognize_alphabet(self):
        image, results = self.make_image_to_predict(is_digit = False)
        #self.result.configure(text = " ".join(results))
        cv2.imshow('Alphabets Image', image)
        cv2.waitKey(0)
        self.clear_canvas()

root = App()
root.resizable(0, 0)
#root.geometry("500x600+0+0")
root.title("Character Recognization using CNN")
root.mainloop()