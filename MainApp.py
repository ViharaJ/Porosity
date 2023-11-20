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
        
        self.tabs = ctk.CTkTabview(master = self)
        self.tabs.add("Manual")
        self.tabs.add("Grid")
        
        self.tabs.pack()
        
        self.frame1 = GridSplitFrame(self)
        self.frame1.pack() 
        
        self.frame2 = ManualFrame(self)
        self.frame2.pack()
        
        self.process_button = ctk.CTkButton(self, text="Apply", command=analyzePorosity)
        self.process_button.pack()
        
        self.mainloop()
    
        
    def analyzePorosity(self, args):
        
        
        
        
class GridSplitFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.rows = ctk.CTkEntry(self)
        self.rows.pack(pady=5, padx=5, expand=True, fill='x')
        self.cols = ctk.CTkEntry(self)
        self.cols.pack(pady=5, padx=5, expand=True, fill='x')
        
        
class ManualFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.options = ctk.CTkOptionMenu(self, values=["Otsu", "Binary"])
        self.options.pack()
        
    
        
        
App()