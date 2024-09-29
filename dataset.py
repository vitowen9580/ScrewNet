import glob 
import glob as gl 
import cv2
import numpy as np


def load_data(args,folder):
    filenames = gl.glob(f'{folder}/*.jpg')
    # filenames.extend(f'{folder}/*.png')
    dataset=[]
    for file in filenames:
        dataset.append(np.array(cv2.imread(file,0)))
    data = np.asarray(dataset)
    data = data.astype('float32') / 255.
    data = np.reshape(data, (len(data), args.image_size, args.image_size, 1))  # adapt this if using `channels_first` image data format
    return data,filenames

def check_pair_training_samples(args,x_train,x_train_noisy):
    for i in range(x_train.shape[0]):
        x_train_img=np.reshape(x_train[i],(args.image_size,args.image_size))
        x_train_noisy_img=np.reshape(x_train_noisy[i] ,(args.image_size,args.image_size))
        merge_img=np.hstack((x_train_img ,x_train_noisy_img ))
        cv2.imwrite(f'{args.paired_sample_folder}/img_{i}.jpg',merge_img*255)
    print('Save paired samples..')
    
    
