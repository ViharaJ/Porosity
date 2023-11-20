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
        
        self.menuPanel = MenuPanel(self)
        self.menuPanel.pack()
        
        self.process_button = ctk.CTkButton(self, text="Apply", command=self.analyzePorosity)
        self.process_button.pack()
        
        self.mainloop()
    
        
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
    def __init__(self, parent):
        super().__init__(parent)
        
        self.e1 = ctk.CTkLabel(self, text="Rows")
        self.e2.pack()
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