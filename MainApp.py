import customtkinter as ctk 
from tkinter import Canvas, filedialog
import cv2 
from PIL import Image, ImageTk
import numpy as np
import os

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Quick Preview")
        self.geometry('830x600')
        
        #init params
        self.init_params()
        self.image_output = None
        
        #layout
        self.rowconfigure((0,1,2,3), weight=1, uniform='a')
        self.columnconfigure(0, weight=1, uniform='a')
        self.columnconfigure(1, weight=2, uniform='a')
        
        self.Import_Button_Image = ctk.CTkButton(self, text="Import Image", command=self.import_image)
        self.Import_Button_Image.grid(row=0,column=1, padx=5, pady=5)
        
        self.Import_Button_Folder = ctk.CTkButton(self, text="Import Directory", command=self.import_dir)
        self.Import_Button_Folder.grid(row=1,column=1, padx=5, pady=5)
        
        self.menuPanel = MenuPanel(self, self.start_vars)
        self.menuPanel.grid(row=2,column=1, padx=5, pady=5)
        
        self.applyButton = ctk.CTkButton(self, text="Process", command=self.analyzePorosity)
        self.applyButton.grid(row=3, column=1, columnspan=2)
        
        self.mainloop()
        
    
    def import_image(self):
        self.path = filedialog.askopenfile().name
        print("Importing Image, ",self.path)
                        
    
    
    
    
    def import_dir(self):
        self.path = filedialog.askdirectory()
        self.images = []
        acceptedFileTypes = acceptedFileTypes = ["png", "jpeg", "tif"]
        
        for file in os.listdir(self.path):
            if file.split(".")[-1] in acceptedFileTypes:
                self.images.append(cv2.imread(os.path.join(self.path, file)))
                
        print(len(self.images))        

            
    def init_params(self):
        self.start_vars = {
            "Rows": ctk.IntVar(value=0),
            "Columns": ctk.IntVar(value=0),
            "Thresh": ctk.StringVar(value="Otsu"),
            }
        
            
    def analyzePorosity(self):
        print("successfully called")
        type = self.menuPanel.get()
        
        pass
        



class MenuPanel(ctk.CTkTabview):
    def __init__(self, parent, param):
        super().__init__(parent)
        
        #Tabs
        self.add("Manual")
        # self.add("Grid Split")
        
        #Frames for the tabs
        # GridSplitFrame(self.tab("Grid Split"), []).pack()
        ManualFrame(self.tab("Manual"), param).pack()
        
        
        
class GridSplitFrame(ctk.CTkFrame):
    def __init__(self, parent, params):
        super().__init__(parent)
        
        self.e1 = ctk.CTkLabel(self, text="Rows")
        self.e1.pack()
        self.rows = ctk.CTkEntry(self)
        self.rows.pack(pady=5, padx=5, expand=True, fill='x')
        
        self.e2 = ctk.CTkLabel(self, text="Columns")
        self.e2.pack()
        self.cols = ctk.CTkEntry(self)
        self.cols.pack(pady=5, padx=5, expand=True, fill='x')
        
        
class ManualFrame(ctk.CTkFrame):
    def __init__(self, parent, param):
        super().__init__(parent)
        
        self.options = ctk.CTkOptionMenu(self, values=["Otsu", "Binary"], variable=param["Thresh"])
        self.options.pack()

        
        
App()