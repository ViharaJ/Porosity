'''
Calculate porosity of opaque pores. Can select multiple ROI per image

For ROI selection: use space or enter to finish current selection 
    and start a new one, use esc to terminate multiple ROI selection process.

How to use: 
    1. Install rembg if you don't have it already: https://github.com/danielgatis/rembg
    2. Change rootDir to your image path and then run 
    
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
from PIL import Image

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
    added_image = cv2.addWeighted(img,0.4,mask,0.3,0)
    
    return added_image

def createROI(image, shrink_rate = 0.25):
    """
    shrink_rate: how much to shrink the image by
    returns [Top_Left_X, Top_Left_Y, Width, Height]
    """
    # Resize image
    im = Image.fromarray(image)
    smaller_img = im.resize((int(im.size[0]*shrink_rate), int(im.size[1]*shrink_rate)))
   
    #get ROIS
    r = cv2.selectROIs("select the area", np.array(smaller_img)) 
    cv2.destroyAllWindows()
    r = np.array(r)
    
    # # remove any empty ROIS
    # for i in range(len(r)):
    #     if not np.any(r[i]):
    #        del r[i]
         
    # resize coordinates to make original dimensions 
    r = (r*(1/shrink_rate)).astype(int)
    
    return r 
    

def labelImage(image, coordinates):
    
    for i in range(len(coordinates)):
        coord = coordinates[i]
        position = (int(coord[0] + coord[2]/2), int(coord[1] + coord[3]/2))
        cv2.putText(image, 
                    "ROI " + str(i + 1), 
                    position,
                    cv2.FONT_HERSHEY_SIMPLEX, #font family
                     3, #font size
                     (255, 80, 203), #font color
                     5) #font stroke)
    
    return image

def getSegmentMask(img, createMaskDir=True, crop_coord=None, fullPath=None):
    """
    img: original image array
    crop_coord: optional
    returns: mask to segement regolith from background
    """
    mask = None
    if createMaskDir:             
        mask = getRemBGMask(img)
    else:
        mask =  cv2.imread(fullPath, cv2.IMREAD_GRAYSCALE)
        
    if crop_coord is not None:
        # Create cropped mask if necessary
        blackCanvas = np.full(shape=(img.shape[0], img.shape[1]), fill_value=0, dtype=np.uint8)
            
        for j in range(len(crop_coord)):   
            each_crop = crop_coord[j]
            blackCanvas[each_crop[1]: each_crop[1] + each_crop[3], 
                        each_crop[0]: each_crop[0] + each_crop[2]] = 255
        
        mask = np.bitwise_and(blackCanvas, mask)
        
    
    return mask

def calculatePorosity(m, p_m, crop_coords=None):
    """
    m: mask
    p_m: pore mask
    crop_coord: optional, ROI coordinates
    returns: array of porosity
    """
    porosity = []
    
    if crop_coords is None:
        bg = cv2.countNonZero(m)
        p = cv2.countNonZero(cv2.bitwise_not(p_m))
        
        porosity.append(p/bg)
        print(porosity[-1])
    else:
        for j  in range(len(crop_coords)):   
            each_crop = crop_coords[j]
          
            crop = m[each_crop[1]: each_crop[1] + each_crop[3], 
                        each_crop[0]: each_crop[0] + each_crop[2]]
            pore_crop = p_m[each_crop[1]: each_crop[1] + each_crop[3], 
                        each_crop[0]: each_crop[0] + each_crop[2]]
            
            bg = cv2.countNonZero(crop)
            p = cv2.countNonZero(cv2.bitwise_not(pore_crop))
            
            if bg != 0:
                porosity.append(p/bg)
                print(porosity[-1])
    
    return porosity

def gridSplit(img, rows, cols):
    """
    img: image
    m: mask
    returns: list of crop coordinates [[Top_Left_X, Top_Left_Y, Width, Height]...]
    """
    #get contour
    contours, hierarchy = cv2.findContours(getSegmentMask(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # create box around contour 
    cnt = contours[0] 
    x,y,w,h = cv2.boundingRect(cnt) # col, row, num of cols, num of rows
    
    grid_w, grid_h = int(w/cols), int(h/rows)
    #iterate over box, create ROIs
    crop_coord = []
    curr_x, curr_y = x , y 
    
    for i in range(cols):
        curr_x = x + i*grid_w
        for j in range(rows):
            curr_y = y + j*grid_h
            each_coord = [curr_x, curr_y, grid_w, grid_h]
            crop_coord.append(each_coord)
        curr_y = y    
    
    return crop_coord


def processImage(img, rootDir, maskDir, pore_maskDir, overlay_imgDir, use_same_ROI=None):
    crop_coord = None
    image_names = [] 
    
    original = cv2.imread(rootDir + "\\" + img)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # get crop coordinates, 
    # Controls: use space or enter to finish current selection 
    # and start a new one, use esc to terminate multiple ROI selection process.
    if not use_same_ROI:
        crop_coord = createROI(original)
        user_val = input("Use same ROI (Y)?")
        
        if user_val.lower() == "y":
            use_same_ROI = True

    #Update name list:
    if (len(crop_coord) == 0):
        image_names.append(img)
    else:
        for i in range(len(crop_coord)):
            image_names.append(img.split('.')[0] + "_ROI_" + str(i + 1))
            
    
    # create general Mask
    #TODO: change to use variable
    mask = getSegmentMask(original, True, crop_coord)
    cv2.imwrite(maskDir + "\\" + img, mask)    
        
    # SELECT HOW PORE MASK WILL BE PRODUCED
    print("Creating pore mask")
    pore_mask = getPoreMask('Otsu',gray, mask)
    
    # save pore mask
    cv2.imwrite(pore_maskDir  + "\\" + img, pore_mask)
    print("Finished creating pore mask")
    
    #overlay image
    overlay_mask = createOverlayImage(original, pore_mask, mask)
    overlay_mask = labelImage(overlay_mask, crop_coord) if len(crop_coord) > 0 else overlay_mask
    cv2.imwrite(os.path.join(overlay_imgDir, img), overlay_mask)
    
    
    print("Calculating porosity")
    return image_names, calculatePorosity(mask, pore_mask, crop_coord)
    
#=============================MAIN========================================
# rootDir = "C:\\Users\\v.jayaweera\\Pictures\\FindingEdgesCutContour\\Tjorben"
# acceptedFileTypes = ["png", "jpeg", "tif"]

# # CHANGE VARIABLES BELOW ACOORDING TO USE
# createMaskDir = True 
# maskDir =  createDir(rootDir, "Segment Mask")
# pore_maskDir = createDir(rootDir, "Pore_Mask")


# overlay_imgDir = createDir(rootDir, "Overlay")
# image_names = []
# Porosity = []
# use_same_ROI = False
# crop_coord = None

# for image_name in os.listdir(rootDir):
#     crop_coord = None
#     if image_name.split(".")[-1] in acceptedFileTypes:
#         print("Processing ", image_name)
#         original = cv2.imread(rootDir + "\\" + image_name)
#         gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        
#         # get crop coordinates, 
#         # Controls: use space or enter to finish current selection 
#         # and start a new one, use esc to terminate multiple ROI selection process.
#         if not use_same_ROI:
#             crop_coord = createROI(original)
#             user_val = input("Use same ROI (Y)?")
            
#             if user_val.lower() == "y":
#                 use_same_ROI = True
        
#         #TODO: REMOVE AFTER TESTING
#         crop_coord = gridSplit(original, 3, 4)
#         print(crop_coord)
    
#         #Update name list:
#         if (len(crop_coord) == 0):
#             image_names.append(image_name)
#         else:
#             for i in range(len(crop_coord)):
#                 image_names.append(image_name.split('.')[0] + "_ROI_" + str(i + 1))
                
        
#         # create general Mask
#         mask = getSegmentMask(original, crop_coord)
#         cv2.imwrite(maskDir + "\\" + image_name, mask)    
            
#         # SELECT HOW PORE MASK WILL BE PRODUCED
#         print("Creating pore mask")
#         pore_mask = getPoreMask('Otsu')
        
#         # save pore mask
#         cv2.imwrite(pore_maskDir  + "\\" + image_name, pore_mask)
#         print("Finished creating pore mask")
        
#         #overlay image
#         overlay_mask = createOverlayImage(original, pore_mask, mask)
#         overlay_mask = labelImage(overlay_mask, crop_coord) if len(crop_coord) > 0 else overlay_mask
#         cv2.imwrite(os.path.join(overlay_imgDir, image_name), overlay_mask)
        
        
#         print("Calculating porosity")
#         Porosity.extend(calculatePorosity(mask, pore_mask, crop_coord))
    
        
        
# df = pd.DataFrame(data=list(Porosity), columns=['Porosity'], index=image_names)
# df.to_excel(rootDir + "\\" + " Porosity.xlsx")