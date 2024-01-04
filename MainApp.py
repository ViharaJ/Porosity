import customtkinter as ctk 
from tkinter import Canvas, filedialog
import tkinter as tk
import cv2 
from PIL import Image, ImageTk
import numpy as np
import os
# import ScratchRemoval_U2NET as S2U


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Calculate Porosity")
        self.geometry('600x830')
        
        #init params
        self.init_params()
        self.image_output = None
        
        #layout
        self.rowconfigure((0,1,2), weight=1, uniform='a')
        self.columnconfigure(0, weight=1, uniform='a')
        self.columnconfigure(1, weight=2, uniform='a')
        
        #frame for canvas
        self.frame1 = ctk.CTkFrame(self)
        self.frame1.grid(row=0, column=0, rowspan=3, sticky = 'nsew')
        self.frame1.rowconfigure(0, weight=4)
        self.frame1.rowconfigure(1, weight=1)
        self.frame1.columnconfigure((0,1), weight=1, uniform='a')
        
        # frame for Panel
        self.frame2 = ctk.CTkFrame(self)
        self.frame2.grid(row=0, column=1, rowspan=3, sticky = 'nsew')
        
        self.frame2.rowconfigure((0,1,2,3), weight=1, uniform='a')
        
        # click left and right buttons
        self.leftButton = ctk.CTkButton(self.frame1, text="<", command=lambda: self.updateCounter(-1))
        self.rightButton = ctk.CTkButton(self.frame1, text=">", command=lambda: self.updateCounter(1))
        
        self.leftButton.grid(row=1,column=0, padx=5, pady=5)
        self.rightButton.grid(row=1,column=1, padx=5, pady=5)
        
        #import buttons
        self.importBttn = ctk.CTkButton(self.frame2, text="Import Dir", command=self.import_dir)
        self.importBttn.grid(row=0,column=0, padx=5, pady=5)
    
        # radio buttons 
        self.rembg_manual_frame = RadioFrame(self.frame2, ["Auto", "Manual"], "Mask Creation")
        self.rembg_manual_frame.grid(row=1,column=0, padx=5, pady=5, sticky = 'nsew')
        
        
        self.poreThresh_frame = RadioFrame(self.frame2, ["Otsu", "Binary", "Manual"], "Pore Mask Creation")
        self.poreThresh_frame.grid(row=2,column=0, padx=5, pady=5, sticky = 'nsew')
        
        
        #APPLY BUTTON
        self.applyButton = ctk.CTkButton(self.frame2, text="Apply", command=self.apply)
        self.applyButton.grid(row=3, column=0, padx=5, pady=5)
        self.mainloop()
        
    
    def apply(self):
        print(self.rembg_manual_frame.getActiveButton())
        print(self.poreThresh_frame.getActiveButton())
        
        
    def updateCounter(self, args):
        self.image_num = self.image_num + args
        
        self.image_num = np.clip(self.image_num, 0, len(self.image_names))
        
        if len(self.image_names) > 0:
            self.load_Image()
        
        
    def init_params(self):
        self.image_names = []
        #variable to track which image is being previewd 
        self.image_num = -1
        
        
    def import_dir(self):
        self.path = filedialog.askdirectory()
        
        self.image_names = []
        acceptedFileTypes = ["png", "jpeg", "tif"]
        
        for file in os.listdir(self.path):
            if file.split(".")[-1] in acceptedFileTypes:
                self.image_names.append(os.path.join(self.path,file))
                
        self.load_Image()
    
    def load_Image(self):
        image_path_for_loading = self.image_names[self.image_num]
        
        self.original = cv2.imread(image_path_for_loading)   
        self.preview = self.original.copy()
        self.image_ratio = self.original.shape[1]/self.original.shape[0]
        
        self.changeImageOutput()
        
                
    def changeImageOutput(self):
        if self.image_output is None:
            #Canvas
            #place canvas onto Frame 1
            self.image_output = ImageOutput(self.frame1, self.resize_image)
            self.image_output.grid(row=0, column=0, columnspan=2, sticky = 'nsew')
            # self.image_output.grid(row=0, column=0, rowspan=3, sticky = 'nsew')
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
        
    def resize_image(self, event):
       self.image_output.delete("all")
       self.canvas_width = event.width 
       self.canvas_height = event.height 
       self.updatePreview_Image()

#===========================PANELS===================

class RadioFrame(ctk.CTkFrame):
    def __init__(self, parent, names, label):
        super().__init__(parent)
        
        #label 
        l = ctk.CTkLabel(self, text=label)
        l.pack()
 
        self.activeButton = tk.StringVar(value=names[0])
        self.names = names
        
        #create radio button 
        for i in range(0, len(names) - 1):
            v = self.names[i]
            button = ctk.CTkRadioButton(self, text=v, variable=self.activeButton, 
                                        value=v, command=lambda: self.radioClicked(v))
            button.pack()
            
        # last button 
        button = ctk.CTkRadioButton(self, text=self.names[-1], variable=self.activeButton, 
                                    value = self.names[-1], command=lambda: self.radioClicked(self.names[-1]))
        button.pack()
        
        # entry 
        self.entry = ctk.CTkEntry(self,state="disabled")
        self.entry.pack()
        
        
        
    def radioClicked(self, args):
       self.activeButton.set(args)
       
       if args == "Manual":
           self.entry.configure(state="normal")
       else:
           self.entry.configure(state="disabled")
       
    
    def getActiveButton(self):
        return self.activeButton.get()
       
       
class ImageOutput(Canvas):
    def __init__(self, parent, resize_func):
        super().__init__(parent, background='white')
        self.bind("<Configure>", lambda e: resize_func(e))
        
        
        
App()