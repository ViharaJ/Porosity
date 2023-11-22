import customtkinter as ctk 
from tkinter import Canvas, filedialog
import cv2 
from PIL import Image, ImageTk
import numpy as np


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Quick Preview")
        self.geometry('830x600')
        
        self.init_params()
        
        self.Import_Button = ctk.CTkButton(self.buttonPanel, text="Import Image", command=self.import_image)
        self.Import_Button.grid(row=0,column=0, padx=5, pady=5)
        
        
        self.mainloop()
    
    def import_image(self):
        
        self.path = filedialog.askopenfile().name
        print("Importing Image, " ,self.path)
    
                
    def init_params(self):
        self.start_vars = {
            "Rows": ctk.IntVar(value=0),
            "Columns": ctk.IntVar(value=0),
            "Thresh": ctk.StringVar(value=0),
            }
        
            
    def analyzePorosity(self):
        print("successfully called")
        pass
        

class MenuPanel(ctk.CTkTabview):
    def __init__(self, parent):
        super().__init__(parent)
        
        #Tabs
        self.add("Manual")
        self.add("Grid Split")
        
        #Frames for the tabs
        GridSplitFrame(self.tab("Grid Split")).pack()
        ManualFrame(self.tab("Manual")).pack()
        
        
        
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
    def __init__(self, parent):
        super().__init__(parent)
        
        self.options = ctk.CTkOptionMenu(self, values=["Otsu", "Binary"])
        self.options.pack()
        
    
        
        
App()