from model import *
from data import *
import sys,os, cv2

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def error(gt, result):
	# Jaccard similarity: which is the ratio between the intersection 
	# and union of the segmented results and the ground truth
	# J= (A intersect B) / (A union B) 

	#print(gt.shape, result.shape)
	gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
	_, gt = cv2.threshold(gt, 128, 255,cv2.THRESH_BINARY)
	result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
	_, result = cv2.threshold(result, 92, 255,cv2.THRESH_BINARY)

	#cv2.imshow('mask', gt)
	#cv2.imshow('nn', result)
	#cv2.imshow("comb", gt/2 + result/2)
	#cv2.waitKey(1000)
	#cv2.destroyAllWindows()
	
	intersect = cv2.bitwise_and(gt, result)
	union = cv2.bitwise_or(gt, result)
	
	whitesI = np.sum(intersect == 255)
	whitesU = np.sum(union == 255)
	#print(whitesI, whitesU)
	e = whitesI/whitesU
	return e



data_gen_args = dict(rotation_range=0.2,
					width_shift_range=0.05,
					height_shift_range=0.05,
					shear_range=0.05,
					zoom_range=0.05,
					horizontal_flip=True,
					fill_mode='nearest')
myGene = trainGenerator(2,'unet/dataset/vessel/train','images','labels',data_gen_args,save_to_dir = None)

#model = unet()
#model_checkpoint = ModelCheckpoint('unet_vessel.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit_generator(myGene,steps_per_epoch=47,epochs=1,callbacks=[model_checkpoint])

model = unet("unet/unet_vessel.hdf5")
testGene = testGenerator("unet/dataset/vessel/test/images")
results = model.predict_generator(testGene,20,verbose=1)
saveResult("unet/dataset/vessel/output",results)
   

for idx, filename in enumerate(os.listdir("unet/dataset/vessel/output/measure")):
	if filename.endswith(".png") : 
		os.remove("unet/dataset/vessel/output/measure/" + filename)

for idx, filename in enumerate(os.listdir("unet/dataset/vessel/output")):
	if filename.endswith(".png") : 
		image = cv2.imread("unet/dataset/vessel/output/"+filename)
		_, res = cv2.threshold(image, 92, 255, cv2.THRESH_BINARY)
		grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#res = cv2.adaptiveThreshold(grayscale , 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,15,2)
		#res = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,2)
		
		cv2.imwrite(os.path.join("unet/dataset/vessel/output/measure", filename), res) 
	else:
		continue

et = 0
em = 0




for idx, filename in enumerate(os.listdir("unet/dataset/vessel/output/measure")):
	if filename.endswith(".png") : 
		if len(filename) == 13:
			os.rename("unet/dataset/vessel/output/measure/"+filename, "unet/dataset/vessel/output/measure/0"+filename)

for idx, filename in enumerate(os.listdir("unet/dataset/vessel/output/measure")):
	if filename.endswith(".png") : 
		print(filename)
		image = cv2.imread("unet/dataset/vessel/output/measure/" + filename)
		image = cv2.resize(image, (1000, 1000))
		cv2.imwrite("unet/dataset/vessel/output/measure/" + filename, image)
		name = "0" + str(idx+48)    
		mask = os.path.join("dataset", "labels", name+".tif")
		#print(mask)
		mask = cv2.imread(mask)
		e = error(mask, image)
		et += e
		print(e)
		#e = error(mask, mask)
		#em += e
		#print(e)
	else:
		continue

print("Average error:", et/20)