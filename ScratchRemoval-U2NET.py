'''
Calculate porosity of opaque pores 



Options: 
    For mask to segement regolith mask from background
        1. REMBG mask
        2. use a pre-existing mask path. (Indicate directory below)
    If using option 2, set createMaskDir to False
    
    
    For pore mask:
        1. create a binarized mask
        2. create a mask from uisng otsu threshholding
        3. use a pre-existing directory (Indicate directory below)
    

Procedure:
    1. REMBG remove background
    2. Get pore mask 
    3. Count black pixels 
    4. Save pore mask as white background and black pores
'''
import cv2 
import os 
import numpy as np 
from rembg import remove, new_session
import pandas as pd
import matplotlib.pyplot as plt


#==============================FUNCTIONS===============================           

def getRemBGMask(image, post_process=True):
    '''
    post_process default is True
    return  mask
    '''
    full_mask = remove(image, post_process_mask=post_process, only_mask=True)  
    
    p1 = np.full((image.shape[0]+2, image.shape[1]+2), fill_value=0, dtype="uint8")
    
    cv2.floodFill(full_mask, p1, (0,0), 0)
    
    p1 = cv2.bitwise_not(p1*255)
    
    plt.imshow(p1, cmap="gray")
    plt.show()
    return p1[1:-1, 1:-1]    
    

def postProcess(image):
    '''
    remove background with processing on mask
    return image with background removed (alpha channel included)
    '''
    return remove(image, post_process_mask=True, session=new_session("u2net"))    


def createPoreMaskBin(image, mask, low, hi):
    '''
    image: gray image array
    mask: grayscale image array
    return: black and white pore mask
    '''
    segmented = cv2.bitwise_and(image, mask)    
    ret, binar = cv2.threshold(segmented, low, hi, cv2.THRESH_BINARY)
    
    return binar + cv2.bitwise_not(mask)

def createPoreMaskOtsu(image, mask):
    '''
    image: gray image array
    mask: grayscale image array
    return: black and white pore mask
    '''
    segmented = cv2.bitwise_and(image, mask)    
    ret, otsu = cv2.threshold(segmented, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return otsu + cv2.bitwise_not(mask)


def makeBGWhite(image, mask):
    '''
    image: coloured image array
    mask: grayscale image, background is black 
    '''
    segment = cv2.bitwise(image, mask)
    mask_inv = cv2.bitwise_not(mask)
    
    return segment + mask_inv


def createDir(root, folderName):
    '''
    creates new folder if it doesn't exist
    returns: new folder's path
    '''
    newPath = os.path.join(root, folderName)
    if not os.path.exists(newPath):
        os.makedirs(newPath)
        
    return newPath


def getPoreMask(option, path = None):
    
    '''
    option is a string, acceptable inputs are: Otsu, Bin, Manual
    returns: white background with black pores mask
    '''
    pore_mask = None
    if option.strip() =='Otsu':
        pore_mask = createPoreMaskOtsu(gray, mask)
        pore_mask_inv = cv2.bitwise_not(pore_mask)
        pore_mask_inv = cv2.medianBlur(pore_mask_inv, 5) 
        pore_mask = cv2.bitwise_not(pore_mask_inv)
    elif option.strip() == 'Bin':
        pore_mask = createPoreMaskBin(gray, mask, 70, 255)   
        pore_mask_inv = cv2.bitwise_not(pore_mask)
        pore_mask_inv = cv2.medianBlur(pore_mask_inv, 5) 
        pore_mask = cv2.bitwise_not(pore_mask_inv)
    elif option.strip() =='Manual':
        pore_mask = cv2.imread(path)
    else: return -1
    
    return pore_mask
    

def createOverlayImage(img, pore_m, mask):
    """
    img: color image array
    pore_m: pores are black, background is white
    mask: grayscale image array
    return: image with background pixels with green tint, pores with red tint
    """
    blackx, blacky = np.where(pore_m == 0) # pore pixels
    bgx, bgy = np.where(mask == 0) # background pixels
    
    #make mask BGR and pores appear as red on it
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask[blackx, blacky,:] = [0,0,255]
    mask[bgx, bgy,:] = [0,255,0]
 
    # overal mask onto original
    added_image = cv2.addWeighted(original,0.4,mask,0.3,0)
    
    return added_image


#=============================MAIN========================================
rootDir = "C:\\Users\\v.jayaweera\\Pictures\\FindingEdgesCutContour\\OneFile"
acceptedFileTypes = ["png", "jpeg", "tif"]

# CHANGE VARIABLES BELOW ACOORDING TO USE
createMaskDir = True 
maskDir =  createDir(rootDir, "BestMask")
pore_maskDir = createDir(rootDir, "Pore_Mask")


overlay_imgDir = createDir(rootDir, "Overlay")
image_names = []
porosity = []
for image_name in os.listdir(rootDir):
    
    if image_name.split(".")[-1] in acceptedFileTypes:
        image_names.append(image_name)
        print("Processing ", image_name)
        original = cv2.imread(rootDir + "\\" + image_name)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        mask = None
        
        if createMaskDir:
            mask = getRemBGMask(original)
            cv2.imwrite(maskDir + "\\" + image_name, mask)
        else:
            mask =  cv2.imread(maskDir + "\\" + image_name, cv2.IMREAD_GRAYSCALE)
        
        
        # SELECT HOW PORE MASK WILL BE PRODUCED
        print("Creating pore mask")
        pore_mask = getPoreMask('Otsu')
        print("Finished creating pore mask")
    
        
        #overlay image
        overlay_mask = createOverlayImage(original, pore_mask, mask)
        cv2.imwrite(os.path.join(overlay_imgDir, image_name), overlay_mask)
        
        
        cv2.imshow("g", pore_mask)
        cv2.waitKey()
        cv2.destroyAllWindows()
         
        cv2.imwrite(pore_maskDir  + "\\" + image_name, pore_mask)
        
        print("Calculating porosity")
        mx,my = np.where(mask == 255)
        x, y = np.where(pore_mask < 30)
        
        porosity.append(len(x)/len(mx))
        print(porosity[-1])


df = pd.DataFrame(data=porosity, columns=['Porosity'], index=image_names)
df.to_excel(rootDir + "\\" + " Porosity.xlsx")