import os
import tkinter as tk
from tkinter import filedialog, Label, Button, BOTTOM
from PIL import Image, ImageTk, ImageEnhance
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import time


# load pre-trained model to classify a traffic sign
model = load_model('ModelC_traffic_signnet_enhanced.h5')

# load the label names
label_names = open("signnames.csv").read().strip().split("\n")[1:]
label_names = [l.split(",")[1] for l in label_names]

def classify_image(image_path):
    start_time = time.time()
    image = Image.open(image_path)
    image = image.resize((32, 32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict_classes([image])[0]
    traffic_sign = label_names[pred]
    print("--- %s seconds ---" % (time.time() - start_time))
    print(traffic_sign)
    output.configure(text=traffic_sign)

def display_classify_btn(image_path):
    classify_btn = Button(window, text='Classify Image', command=lambda:classify_image(image_path), padx=10, pady=5)
    classify_btn.configure(bg='#0076CE', fg='white', font=('calibri', 12, 'bold'))
    classify_btn.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        image_path = tk.filedialog.askopenfilename()
        upload = Image.open(image_path)
        # upload.thumbnail(((window.winfo_width()), (window.winfo_width())))
        # ((window.winfo_width()/2), (window.winfo_width()/2))
        # resize uploaded image and improve contrast
        upload = upload.resize((100, 100))
        enhancer = ImageEnhance.Brightness(upload)
        upload = enhancer.enhance(2)
        image = ImageTk.PhotoImage(upload)
        sign_image.configure(image=image)
        sign_image.image = image
        output.configure(text='')
        display_classify_btn(image_path)
    except:
        print('Error uploading image!')

if __name__ == '__main__':
    # construct GUI
    window = tk.Tk()
    window.geometry('800x600')
    window.title('Traffic Sign Recognition')
    window.configure(bg='#f9f6f7')

    heading = Label(window, text='Traffic Sign Recognition CNN Model GUI', pady=20, font=('calibri', 24, 'bold'))
    heading.configure(bg='#f9f6f7', fg='#0076CE')
    heading.pack()

    output = Label(window, font=('calibri', 16, 'bold'))
    output.configure(bg='#f9f6f7', fg='#0076CE')

    upload_btn = Button(window, text='Upload Image', command=upload_image, padx=10, pady=5)
    upload_btn.configure(bg='#0076CE', fg='white', font=('calibri', 12, 'bold'))
    upload_btn.pack(side=BOTTOM, pady=50)

    sign_image = Label(window)
    sign_image.pack(side=BOTTOM, expand=True)
    output.pack(side=BOTTOM, expand=True)
    window.mainloop()