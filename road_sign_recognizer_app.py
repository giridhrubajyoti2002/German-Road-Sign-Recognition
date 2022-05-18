from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import pandas as pd
import numpy as np
import os
import cv2
import csv
from keras.models import load_model




def upload_image():
    global filepath
    filepath = filedialog.askopenfilename(filetypes=[('Image File', ['.jpg', '.png', '.jpeg', 'jfif'])])
    global img
    img = Image.open(filepath).resize((550,400))
    img = ImageTk.PhotoImage(img)
    img_result.configure(image=img)
    img_result.image = img
    text_result.configure(text='')
    # text_result.pack_forget()

def get_modified_string(txt):
    st = ''
    length = 0
    for s in txt.split(' '):
        if length>25:
            length = 0
            st = st + '\n'
        st = st + s + ' '
        length = length + len(s) + 1
    return st
def get_result(classId):
    sign = pd.read_csv('datasets/Meta.csv')
    sign.index = sign['ClassId']
    sign = sign['Path']
    result = ImageTk.PhotoImage(Image.open(os.path.join('datasets', str(sign.loc[classId]))).resize((550,400)))    
    img_result.configure(image=result)
    img_result.image = result

    name = pd.read_csv('datasets/sign_names.csv')
    name.index = name['ClassId']
    name = name['SignName']
    txt = str(name.loc[classId])
    if len(txt)>35:
        txt = get_modified_string(txt)
    # text_result.pack()
    text_result.configure(text=txt)

def predict():
    model = load_model('model_experiments/road_sign_detection_model')
    data = pd.read_csv(os.path.join(os.getcwd(),'model_experiments','variable_values.csv'))
    data.index = data['Variable Name']
    img = cv2.imread(filepath, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(data.loc['ydim']['Value'], data.loc['xdim']['Value']))
    img = np.array(img)
    img = img.astype('float32')
    img /= 255
    classId = np.argmax(model.predict(np.expand_dims(img, axis=0)))    
    get_result(classId)





root = Tk()
root.title('Traffic Sign Recognizer')
root.geometry("800x720")
root.configure(padx=30, pady=40, bg='cyan' )

text_result = Label(root, text="", font="helvetica 28 bold", bg='cyan', foreground='dark blue')
text_result.pack()

filepath = os.path.join(os.getcwd(), 'blank_img.jpg')
img = ImageTk.PhotoImage(Image.open(filepath).resize((550,400)))
img_result = Label(root, image=img, bg='cyan')
img_result.pack(pady=25)

Button(root, text="Recognize Image", command=predict, font='helvetica 15 bold', bg='blue', foreground='white', activebackground='navy blue', activeforeground='white').place(x=120, y=560)    #pack(side='left', padx=150)

Button(root, text=" Upload Image ", command=upload_image, font='helvetica 15 bold', bg='blue', foreground='white', activebackground='navy blue',  activeforeground='white').place(x=440, y=560)    #pack(side='top', pady=25, padx=150)


mainloop()

