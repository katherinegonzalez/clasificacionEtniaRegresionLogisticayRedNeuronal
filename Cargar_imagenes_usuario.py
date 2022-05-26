from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import numpy as np

def elegir_imagen():
    # Especificar los tipos de archivos, para elegir solo a las imágenes
    path_image = filedialog.askopenfilename(filetypes = [
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg")])
    
    if len(path_image) > 0:
        global image
        
        # Leer la imagen de entrada y la redimensionamos
        image = cv2.imread(path_image)
        image= imutils.resize(image, height=380)
        
        # Para visualizar la imagen de entrada en la GUI
        imageToShow= imutils.resize(image, width=180)
        imageToShow = cv2.cvtColor(imageToShow, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(imageToShow )
        img = ImageTk.PhotoImage(image=im)
        
        lblInputImage.configure(image=img)
        lblInputImage.image = img
        
        # Label IMAGEN DE ENTRADA
        lblInfo1 = Label(root, text="Imagen seleccionada:")
        lblInfo1.grid(column=0, row=1, padx=5, pady=5)
        
        # Al momento que leemos la imagen de entrada, vaciamos
        # la iamgen de salida y se limpia la selección de los
        # radiobutton
        lblOutputImage.image = ""
        selected.set(0)

def deteccion_color():
    # Para visualizar la imagen en lblOutputImage en la GUI
    imageToShow = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(imageToShow)
    img = ImageTk.PhotoImage(image=im)
    lblOutputImage.configure(image=img)
    lblOutputImage.image = img
    
    # Label IMAGEN DE SALIDA
    lblInfo3 = Label(root, text="Imagen clasificada:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)
            
image = None

# Creamos la ventana principal
root = Tk()

# Label donde se presentará la imagen de entrada
lblInputImage = Label(root)
lblInputImage.grid(column=0, row=2)
      
# Label donde se presentará la imagen de salida
lblOutputImage = Label(root)
lblOutputImage.grid(column=1, row=1, rowspan=6)

# Label ¿Qué color te gustaría resaltar?
lblInfo2 = Label(root, text="¿Qué modelo te gustaría usar?", width=25)
lblInfo2.grid(column=0, row=3, padx=5, pady=5)

# Creamos los radio buttons y la ubicación que estos ocuparán
selected = IntVar()
rad1 = Radiobutton(root, text='Genero', width=25,value=1, variable=selected, command = deteccion_color)
rad2 = Radiobutton(root, text='Etnia',width=25, value=2, variable=selected, command = deteccion_color)
rad3 = Radiobutton(root, text='Edad',width=25, value=3, variable=selected, command = deteccion_color)
rad1.grid(column=0, row=4)
rad2.grid(column=0, row=5)
rad3.grid(column=0, row=6)

btn = Button(root, text = "Eliga una imagen del rostro", width = 25, command = elegir_imagen)  
btn.grid(column=0, row=0, padx=5, pady=5)

root.mainloop()