import customtkinter as ctk 
from tkinter import Canvas, filedialog
import cv2 
from PIL import Image, ImageTk
import numpy as np
import os
import ScratchRemoval_U2NET as S2U


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Quick Preview")
        self.geometry('830x600')
        
        #init params
        self.init_params()
        
        #layout
        self.rowconfigure((0,1,2), weight=1, uniform='a')
        self.columnconfigure(0, weight=1, uniform='a')
        self.columnconfigure(1, weight=2, uniform='a')
    
        self.Import_Button_Folder = ctk.CTkButton(self, text="Import Directory", command=self.import_dir)
        self.Import_Button_Folder.grid(row=0,column=1, padx=5, pady=5)
        
        self.menuPanel = MenuPanel(self, self.start_vars)
        self.menuPanel.grid(row=1,column=1, padx=5, pady=5)
        
        self.applyButton = ctk.CTkButton(self, text="Process", command=self.gridSplit)
        self.applyButton.grid(row=2, column=1, columnspan=2)
        
        self.mainloop()
        
    
    
    
    def import_dir(self):
        self.path = filedialog.askdirectory()
        self.maskDir = S2U.createDir(self.path, "Segment Mask")
        self.pore_maskDir = S2U.createDir(self.path, "Pore_Mask")
        self. overlay_imgDir = S2U.createDir(self.path, "Overlay")
        self.image_names = []
        acceptedFileTypes = acceptedFileTypes = ["png", "jpeg", "tif"]
        
        for file in os.listdir(self.path):
            if file.split(".")[-1] in acceptedFileTypes:
                self.image_names.append(file)
                
           

            
    def init_params(self):
        self.start_vars = {
            "Rows": ctk.IntVar(value=1),
            "Columns": ctk.IntVar(value=1),
            "Thresh": ctk.StringVar(value="Otsu"),
            }
        
        self.all_names = []
        self.Porosity = []
            
    def analyzePorosity(self):
        print("successfully called")
        analysisType = self.menuPanel.get()
        
        for img in self.image_names:
            print("Processing: ", img)
            n, r = S2U.processImage(img,self.path, self.maskDir, self.pore_maskDir,
                             self.overlay_imgDir, self.start_vars["Thresh"].get())
            
            self.all_names.extend(n)
            self.Porosity.extend(r)
            
            
    def gridSplit(self):
        print("Processing using Grid Split")
        
        #SAFEGUARD
        r = self.start_vars['Rows'].get()
        c = self.start_vars['Columns'].get()
        
        print(r,c)
        
        for img in self.image_names:            
            n,r = S2U.processImageGridSplit(img,  self.path,  self.maskDir, 
                                     self.pore_maskDir,  self.overlay_imgDir,
                                     "Otsu", r,c)


class MenuPanel(ctk.CTkTabview):
    def __init__(self, parent, param):
        super().__init__(parent)
        
        #Tabs
        self.add("Manual")
        self.add("Grid Split")
        
        #Frames for the tabs
        ManualFrame(self.tab("Manual"), param).pack()
        GridSplitFrame(self.tab("Grid Split"), param).pack()
        
        
        
class GridSplitFrame(ctk.CTkFrame):
    def __init__(self, parent, params):
        super().__init__(parent)
        
        self.e1 = ctk.CTkLabel(self, text="Rows")
        self.e1.pack()
        self.rows = ctk.CTkEntry(self, textvariable=params["Rows"])
        self.rows.pack(pady=5, padx=5, expand=True, fill='x')
        
        self.e2 = ctk.CTkLabel(self, text="Columns")
        self.e2.pack()
        self.cols = ctk.CTkEntry(self, textvariable=params["Columns"])
        self.cols.pack(pady=5, padx=5, expand=True, fill='x')
        
        
class ManualFrame(ctk.CTkFrame):
    def __init__(self, parent, param):
        super().__init__(parent)
        
        self.options = ctk.CTkOptionMenu(self, values=["Otsu", "Binary", "Manual"], variable=param["Thresh"])
        self.options.pack()

        
        
App()