import sys,os, cv2



for idx, filename in enumerate(os.listdir(".")):
    if filename.endswith(".tif") : 
            image = cv2.imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(os.path.join(filename))
            cv2.imwrite(str(idx+1) + ".png", gray)
    
    else:
        continue