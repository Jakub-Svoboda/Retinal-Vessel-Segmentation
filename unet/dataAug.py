from data import *
import cv2

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGenerator = trainGenerator(20,'unet/dataset/vessel/train','images','labels',data_gen_args,
save_to_dir = "unet/dataset/vessel/aug")

#you will see 60 transformed images and their masks in data/membrane/train/aug
num_batch = 3
for i,batch in enumerate(myGenerator):
    print(batch[0])
    cv2.imshow('image',batch[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#image_arr,mask_arr = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/")
#np.save("dataset/image_arr.npy", image_arr)
#np.save("dataset/mask_arr.npy", mask_arr)

