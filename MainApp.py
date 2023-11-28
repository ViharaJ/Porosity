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
        self.geometry('600x830')
        
        #init params
        self.init_params()
        self.image_output = None
        
        #layout
        self.rowconfigure((0,1,2), weight=1, uniform='a')
        self.columnconfigure(0, weight=1, uniform='a')
        self.columnconfigure(1, weight=2, uniform='a')
    
        self.importPanel()
        
        self.menuPanel = MenuPanel(self, self.start_vars)
        self.menuPanel.grid(row=1,column=1, padx=5, pady=5)
        
        self.applyButton = ctk.CTkButton(self, text="Process", command=self.processImages)
        self.applyButton.grid(row=2, column=1, columnspan=2)
        
        self.mainloop()
        
    
    def processImages(self):
        self.maskDir = S2U.createDir(self.start_vars["dir"].get(), "Segment Mask")
        self.pore_maskDir = S2U.createDir(self.start_vars["dir"].get(), "Pore_Mask")
        self. overlay_imgDir = S2U.createDir(self.start_vars["dir"].get(), "Overlay")
        
        analysisType = self.menuPanel.get()
        
        if analysisType == "Manual":
            self.analyzePorosity()
        else: self.gridSplit()
        
    
    
    def import_dir(self):
        self.path = filedialog.askdirectory()
        
        self.start_vars["dir"].set(self.path)
        self.image_names = []
        acceptedFileTypes = ["png", "jpeg", "tif"]
        
        for file in os.listdir(self.path):
            if file.split(".")[-1] in acceptedFileTypes:
                self.image_names.append(file)
        
        self.import_image()
                
           
    def import_image(self):
        self.original = cv2.imread(os.path.join(self.start_vars["dir"].get(), self.image_names[0]))   
        self.preview = self.original.copy()
        self.image_ratio = self.original.shape[1]/self.original.shape[0]
        
        self.changeImageOutput()
    
    
    def changeImageOutput(self):
        if self.image_output is None:
            #Canvas
            self.image_output = ImageOutput(self, self.resize_image)
            self.image_output.grid(row=0, column=0, rowspan=3, sticky = 'nsew')
            self.canvas_width = self.image_output.winfo_reqwidth()
            self.canvas_height = self.image_output.winfo_reqheight()
        elif self.image_output:
            self.image_output.delete("all")
            print("Removed old image")
            self.updatePreview_Image()
            print("inserting new image")
        
        self.updatePreview_Image()
        
        
    def updatePreview_Image(self):
       #get new dimensions
       canvas_ratio =  self.canvas_width / self.canvas_height
       if canvas_ratio > self.image_ratio: #canvas wider than image
           im_height = self.canvas_height
           im_width = im_height * self.image_ratio
       else:                               #canvas narrower than image
           im_width = self.canvas_width
           im_height = im_width / self.image_ratio
       
       #update image
       self.preview_img = Image.fromarray(self.preview).resize((int(im_width), int(im_height)))
       self.drawImage()
       
        
    def drawImage(self):
        #update preview_tk and place
        self.preview_tk = ImageTk.PhotoImage(self.preview_img)
        self.image_output.create_image(int(self.canvas_width/2), 
                                       int(self.canvas_height/2),
                                       anchor="center",
                                       image = self.preview_tk)    
            
    def init_params(self):
        self.start_vars = {
            "dir": ctk.StringVar(value=""),
            "Rows": ctk.IntVar(value=1),
            "Columns": ctk.IntVar(value=1),
            "Thresh": ctk.StringVar(value="Otsu"),
            }
        
        self.all_names = []
        self.Porosity = []
    
    def importPanel(self):
        self.f1 = ctk.CTkFrame(self)
        
        self.Import_Button_Folder = ctk.CTkButton(self.f1, text="Import Directory", command=self.import_dir)
        self.import_e = ctk.CTkEntry(self.f1, textvariable=self.start_vars["dir"])
        
        self.import_e.pack()
        self.Import_Button_Folder.pack()
        
        self.f1.grid(row=0,column=1, padx=5, pady=5)
        
    def analyzePorosity(self):
        print("successfully called")
        
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
            
            
    def resize_image(self, event):
        self.image_output.delete("all")
        self.canvas_width = event.width 
        self.canvas_height = event.height 
        self.updatePreview_Image()


#===========================PANELS===================

class MenuPanel(ctk.CTkTabview):
    def __init__(self, parent, param):
        super().__init__(parent)
        
        #Tabs
        self.add("Manual")
        self.add("Grid Split")
        
        #Frames for the tabs
        ManualFrame(self.tab("Manual"), param).pack()
        GridSplitFrame(self.tab("Grid Split"), param, self.checkIfInt).pack()
        
    
    def checkIfInt(self, p):
        if str.isdigit(p):
            return True
        else: return False
        
        
        
class GridSplitFrame(ctk.CTkFrame):
    def __init__(self, parent, params, cback):
        super().__init__(parent)
        
        self.options = ctk.CTkOptionMenu(self, values=["Otsu", "Binary", "Manual"], variable=params["Thresh"])
        self.options.pack()
        
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


class ImageOutput(Canvas):
    def __init__(self, parent, resize_func):
        super().__init__(parent, background='white')
        self.bind("<Configure>", lambda e: resize_func(e))
        
        
App()