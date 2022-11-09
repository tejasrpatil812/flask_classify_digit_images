import pickle
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import glob

MODEL = "./new_model.sav"
model = pickle.load(open(MODEL, 'rb'))

IMAGE_DIR = "./ManualTestData/*"

max_value = 255
delta = 5
max_depth = 25
threshold = 135

def is_valid (x,y,m,n,dict):
    return (x >= 0 and y >= 0 and x < m and y < n) and ((x,y) not in dict)

def update_pixels(img, xi, yi, w, h):
    queue, dict, depth = [], {}, 0
    queue.append((xi,yi,0))
   
    while len(queue):
        (x,y,depth) = queue.pop()
        if is_valid(x,y,w,h,dict) and depth<max_depth:
            img.putpixel((x,y), max(img.getpixel((x,y)),max_value - delta*depth))
            queue.append((x+1,y,depth+1))
            queue.append((x-1,y,depth+1))
            queue.append((x,y+1,depth+1))
            queue.append((x,y-1,depth+1))
        dict[(x,y)] = 1
        depth += 1
    return img

def broaden(img1):
    img2 = img1.copy()
    w = img1.width
    h = img1.height 
    for x in range(w):
        for y in range(h):
            if img1.getpixel((x,y)) == 255:
                img2 = update_pixels(img2, x, y, w, h)
    return img2

def is_white_background(img):
    w = img.width
    h = img.height 
    white_pixel, total_pixel = 0, w*h
    for x in range(w):
        for y in range(h):
            if img.getpixel((x,y)) > threshold:
                white_pixel+=1
    return white_pixel>total_pixel//2

def preprocess(img):
    img = img.convert('L')
    img = img.resize((256, 256))

    if is_white_background(img):
        img = ImageOps.invert(img)
        
    img = img.point( lambda p : 255 if p > threshold else 0)
    img = broaden(img)
    img = img.filter(ImageFilter.BLUR)
    img = img.resize((28, 28))
    return img


def classify(file_name):
    img = Image.open(file_name)
    img = preprocess(img)

    npimg = np.asarray(img)
    npimg = npimg / 255.0
    npimg = npimg[np.newaxis,:,:,np.newaxis]
    result = model.predict(npimg, verbose=1)
    
    return str(result.argmax()), result
   
def validate_manual_images(file_name):
    label_number, _ = classify(file_name)
    actual_number = file_name.split('/')[-1].split('.')[0].split('_')[0]
    print(f"L:{label_number}---A:{actual_number} : {int(label_number) == int(actual_number)}")
    print("---------------------------------------------------")
    return int(label_number) == int(actual_number)

if __name__ == '__main__':
    correct, count = 0,0
    for file_name in glob.glob(IMAGE_DIR):
        match = validate_manual_images(file_name)
        count += 1
        correct += 1 if match else 0
    print(correct/count)