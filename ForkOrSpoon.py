
import tkinter as tk
from tkinter.ttk import *
from tkinter import TOP, BOTTOM, NW, NE, LEFT, messagebox, filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import pandas as pd
import cv2
import os


CATEGORIES = ["fork", "spoon"]

# loading trained model
model = tf.keras.models.load_model("CNN.model")

# init camera
width, height = 300, 300
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


# init window GUI
window = tk.Tk()
# adding camera label to window
lmain = tk.Label(window)
lmain.pack()


def takePicture():

    retval, frame = cap.read()
    if retval != True:
        raise ValueError("Can't read frame")

    cv2.imwrite('img.jpg', frame)

    #cv2.imshow("img1", frame)
    #showMassege("Image cupture", "Image saved successfully")
    cv2.destroyAllWindows()

    load_image_and_predict()


def load_image_and_predict():

    image_path = os.path.dirname(os.path.abspath(__file__))
    image_path += "/img.jpg"

    array_image = prepare(image_path)
    result = prediction(array_image)
    msg = "The prediction is: " + result
    showMassege("Prediction", msg)


def showMassege(title, msg):
    tk.messagebox.showinfo(title=title, message=msg, parent=window)


def dialog_open_file():
    image_path = filedialog.askopenfilename()

    if image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
        array_image = prepare(image_path)
        result = prediction(array_image)
        msg = "The prediction is: " + result
        showMassege("Prediction", msg)


def loadGui():
    window.title("Fork or Spoon")
    window.geometry('800x600')
    window.bind('<Escape>', lambda e: window.quit())

    quit_label = tk.Label(window, text="esc to exit")
    quit_label.pack(side=BOTTOM, anchor=NW, padx=5)

    show_camera_frame()

    button_loadfile = Button(window, text='Select Image',
                             command=lambda: dialog_open_file())
    button_loadfile.pack(side=BOTTOM, pady=5)

    button = Button(window, text='Take picture',
                    command=lambda: takePicture())
    button.pack(side=BOTTOM, pady=5)

    window.mainloop()

# live camera frame


def show_camera_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_camera_frame)
    if cv2.waitKey(1) == 27:
        return  # esc to quit

# prepare the iamge for prediction


def prepare(file):
    IMG_SIZE = 100
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def prediction(image):
    prediction = model.predict([image])
    prediction = list(prediction[0])

    return CATEGORIES[prediction.index(max(prediction))]


def main():
    loadGui()


if __name__ == '__main__':
    main()
