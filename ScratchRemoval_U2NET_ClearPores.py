"""
Calculates porosity of regolith samples with clear pores. Results are placed 
into an excel file in the directory containing the images. 
Muliple folders are created. 

Full Process: 
    Part 1.
        1. Load your image 
        2. Use a mask to segment regoligth from the background 
        3. Extract a mask for the black pores 
        4. Regolith - Black pores = Regolith + Clear pores (this makes sense, just think for like a min)
        5. Get the average colour of (Regolith + Clear pore)
        6. Paint the black pores in this color
        7. Multiply the mask for segmenting with the image from Step 6. 
        8. Switch to ImageJ/Fiji 
        
    Part 2 (In ImageJ/Fiji): 
        1. Open Image(s)
        2. Threshold and set bounds to extract clear Pores
        3. Save results (clear pore mask)
        
    Part 3:
        1. Open segmenting mask 
        2. Open pore mask and clear pore mask
        3. Combine masks in Step 2 to create the final pore mask 
        4. Calculate Porosity 
        
!!!How to use!!!: 
    1. Install rembg if you don't have it already: https://github.com/danielgatis/rembg
    2. Change rootDir to your images folder
    3. Change thresh_type to the thresh holding type to be used to generate the pore mask
    4. Run
    
    
    
Changeable Options: 
    For the mask to segement the regolith from background. 
    
    You can either use a: 
        1. REMBG mask (created in script)
        2. use a pre-existing mask path. (Indicate directory below)
    Change the boolean variable createMask accordingly
    
    
    For the pore mask you can:
        1. create a binarized mask
        2. create a mask from uisng otsu threshholding
        3. use a pre-existing directory (Indicate directory below)
    Change the variable thresh_type to one of the following options: Otsu, Binary, Manual
"""
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

def getPoreMask(option, gray, mask, path = None):
    
    '''
    option is a string, acceptable inputs are: Otsu, Bin, Manual
    gray: gray scale image,
    mask: black and white mask,
    returns: white background with black pores mask
    '''
    pore_mask = None
    if option.strip() =='Otsu':
        pore_mask = createPoreMaskOtsu(gray, mask)
        pore_mask_inv = cv2.bitwise_not(pore_mask)
        pore_mask_inv = cv2.medianBlur(pore_mask_inv, 5) 
        pore_mask = cv2.bitwise_not(pore_mask_inv)
    elif option.strip() == 'Binary':
        pore_mask = createPoreMaskBin(gray, mask, 70, 255)   
        pore_mask_inv = cv2.bitwise_not(pore_mask)
        pore_mask_inv = cv2.medianBlur(pore_mask_inv, 5) 
        pore_mask = cv2.bitwise_not(pore_mask_inv)
    elif option.strip() =='Manual':
        pore_mask = cv2.imread(path)
    else: return -1
    
    return pore_mask

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

def getAverageColour(image, mask, p_mask):
    '''
    image: RGB image matrix
    mask: mask to remove background 
    p_mask: mask of black pores
    '''
    assert len(image.shape) == 3, \
        print ("Please use an RGB image to find the average colour of the material")
    
    copy = mask.copy()
    x, y = np.where(p_mask == 0)
    copy[x,y] = 0
    
    # everthing that is material or clear pore
    mx, my = np.where(copy != 0)#
    
    # get average colour of everthing
    average_colour = np.average(image[mx, my, :], axis = 0).astype(int)
    
    return average_colour

def paintInMask(image, mask, colour):
    '''
    Colours in image with colour everywhere mask is black
    
    Arguments: 
    image: image array
    mask: grayscale image
    colour: either 1 or 3 length array

    returns coloured in image
    '''
    assert len(image.shape) == len(colour)
    assert len(mask.shape) == 2

    copy = image.copy()
    
    x,y = np.where(mask == 0)
    
    if len(image.shape) == 3: 
        copy[x,y,:] = colour 
    else: 
        copy[x,y] = colour[0]
        
    return copy

def multiplyImages(img1, img2):
    assert img1.shape == img2.shape, \
        print("Images must grayscale and the same shape for image multiplication")

    img1 = np.array(img1)
    img2 = np.array(img2)
    n = np.multiply(img1, img2)
    
    # clip values to range [0, 255]   
    for row in range(n.shape[0]):
        f = n[row, :]
        n[row, :]=np.clip(f, 0,255)
           
   
    return n

def colourANDMask(image, mask):
    copy = image.copy()
    x,y = np.where(mask == 0)
    
    copy[x,y,:] = 0
    
    return copy

def makeKernel(size):
    '''
    Return: size x size matrix with all values 1
    '''
    k = size
    if k % 2 == 0:
        k = k + 1
        print("The kernel size for median filtering is even. It has been changed to: ", k + 1)
        
    return np.ones((k, k), np.uint8)
    
def getHarrisCorners(image, a = 12, b = 5, c = 0.06, max_thresh = 0.07, dilate = False, kernel_size = 13):
    '''
    image: must be GRAYSCALE image
    a,b,c: values for cornerHarris
    dilate, kernel_size: optional arguments 
    Return: potential corners plotted on black and white image
    '''
    m = image
    
    #variables to change to get different results in corner map
    dest = cv2.cornerHarris(m, a, b, c)
    
    #dilate image 
    if dilate:
        kernel = makeKernel(kernel_size)
        dest = cv2.dilate(dest, kernel)
    
    
    mask = np.zeros(image.shape, dtype="uint8")
    indices = dest>max_thresh*dest.max()
    mask[indices]=255
    
    return mask

def saveToExcel(porosity_data, names, rootDir):
    df = pd.DataFrame(data=list(porosity_data), columns=['Porosity'], index=names)
    df.to_excel(rootDir + "\\" + " Porosity.xlsx")


def processPart1():
    global inputDir, pore_maskDir, multiplyDir, harrisDir, coloredInDir,maskDir
    for image_name in os.listdir(inputDir):
        print("Processing ", image_name)
        
        # load original image
        original = cv2.imread(inputDir + "/" + image_name)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
       
        # masks
        if createMask:
            general_mask = getRemBGMask(original)  
        else: 
            general_mask = cv2.imread(os.path.join(maskDir, image_name), cv2.IMREAD_GRAYSCALE)
        
        cv2.imwrite(maskDir + "/" + image_name, general_mask)    
        segment = colourANDMask(original, general_mask)
        
        
        pore_mask = getPoreMask(thresh_type, gray, general_mask)
        
        #remove particles 
        pore_mask_inv = cv2.bitwise_not(pore_mask)
        pore_mask_inv = cv2.medianBlur(pore_mask_inv, 5) 
        pore_mask = cv2.bitwise_not(pore_mask_inv)
        cv2.imwrite(os.path.join(pore_maskDir, image_name), pore_mask)
        
        # average colour of material + clear pores
        reg_avgColor = getAverageColour(original, general_mask, pore_mask)
        
        painted_in = paintInMask(segment, pore_mask, reg_avgColor)
        cv2.imwrite(coloredInDir + "/" + image_name, painted_in)
       
        multiply = multiplyImages(painted_in, cv2.cvtColor(general_mask, cv2.COLOR_GRAY2BGR))           
        multiply = cv2.cvtColor(multiply, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(multiplyDir + "/" + image_name, multiply)
        
        
def processPart2():
    global inputDir, pore_maskDir, multiplyDir, harrisDir, coloredInDir, maskDir
                
#=============================MAIN========================================
rootDir = "C:/Users/v.jayaweera/Pictures/20231012_Remelting"
inputDir = "C:/Users/v.jayaweera/Pictures/20231012_Remelting/20231012_Remelting"
createMask = True
thresh_type = "Otsu"


maskDir = createDir(rootDir, "General_Mask")
pore_maskDir = createDir(rootDir, "pore_mask")
allpores_maskDir = createDir(rootDir, "allpores_mask")
coloredInDir = createDir(rootDir, "coloured_in")
multiplyDir = createDir(rootDir, "multiply")
harrisDir = createDir(rootDir, "harris_corners")


for image_name in os.listdir(inputDir):
    print("Processing ", image_name)
    
    # load original image
    original = cv2.imread(inputDir + "/" + image_name)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
   
    # masks
    if createMask:
        general_mask = getRemBGMask(original)  
    else: 
        general_mask = cv2.imread(os.path.join(maskDir, image_name), cv2.IMREAD_GRAYSCALE)
    
    cv2.imwrite(maskDir + "/" + image_name, general_mask)    
    segment = colourANDMask(original, general_mask)
    
    
    pore_mask = getPoreMask(thresh_type, gray, general_mask)
    
    #remove particles 
    pore_mask_inv = cv2.bitwise_not(pore_mask)
    pore_mask_inv = cv2.medianBlur(pore_mask_inv, 5) 
    pore_mask = cv2.bitwise_not(pore_mask_inv)
    cv2.imwrite(os.path.join(pore_maskDir, image_name), pore_mask)
    
    # average colour of material + clear pores
    reg_avgColor = getAverageColour(original, general_mask, pore_mask)
    
    painted_in = paintInMask(segment, pore_mask, reg_avgColor)
    cv2.imwrite(coloredInDir + "/" + image_name, painted_in)
   
    multiply = multiplyImages(painted_in, cv2.cvtColor(general_mask, cv2.COLOR_GRAY2BGR))           
    multiply = cv2.cvtColor(multiply, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(multiplyDir + "/" + image_name, multiply)
    

image_names = []
porosity = []

# CHANGE HERE
fijiDir = rootDir + "/" + "fiji"
contourfill = createDir(rootDir, "contourFill")

for image_name in os.listdir(pore_maskDir):
    #open general maks
    image_names.append(image_name)
    mask = cv2.imread(maskDir + "/" + image_name, cv2.IMREAD_GRAYSCALE)
    
    #open black pore mask
    black_pores_mask = cv2.imread(allpores_maskDir + "/" + image_name, cv2.IMREAD_GRAYSCALE)
    
    #open fiji result (clear pores mask)
    fiji = cv2.imread(fijiDir + "/" + image_name)
    fiji = cv2.cvtColor(fiji, cv2.COLOR_BGR2GRAY)
    clearx, cleary = np.where(fiji == 255)
    
    #colour in clear pores to black pore mask
    black_pores_mask[clearx, cleary] = 0
    all_pores_mask = cv2.medianBlur(black_pores_mask, 3)
    
    #save full pore mask
    cv2.imwrite(os.path.join(allpores_maskDir, image_name), all_pores_mask)
    
    plt.imshow(all_pores_mask, cmap="gray")
    plt.show()
    
    #calc poreosity
    pores = len(np.where(all_pores_mask == 0)[0])
    material = len(np.where(mask == 255)[0])
    
    print(image_name, " ", pores/material)
    porosity.append(pores/material)


# save data
saveToExcel(porosity, image_names, rootDir)    
