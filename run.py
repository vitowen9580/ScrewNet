
from  tensorflow.keras.models import Model
from  tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mean_squared_error, Huber
import glob 
import glob as gl 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import cv2
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow import keras
import numpy as np
import cv2
from sklearn.metrics import label_ranking_average_precision_score
import time 
from skimage import morphology 
import os
import PIL.Image as Image
import shutil
import math
from skimage import filters
import time
import random
import math
from tensorflow.keras.layers import *
import logging
import util
import dataset
from model import AE,CycleGAN



logging.basicConfig(level=logging.INFO)

'''
Myself Lib
'''
#from Metrics import _metrics
from ImageProcessing import Imageprocess



class runner:
    def __init__(self,args):

        self.Imageprocess=Imageprocess(args)
        
        self.args=args
        self.image_size=args.image_size
        self.Noise_Type=args.Noise_Type
        self.Noise_Proportion=args.Noise_Proportion

        self.test_folder=args.test_folder
        self.recontruction_folder=args.recontruction_folder
        self.residual_folder=args.residual_folder
        self.training_folder=args.training_folder
        self.val_folder=args.val_folder
        self.result_img= fr'models/{args.Anomaly_Network}_{args.Noise_Type}_{args.Screw_Type}.png' 
        self.AE=AE(args)
        self.CycleGAN=CycleGAN(args)

        self.train_noise_folder=args.train_noise_folder
        self.val_noise_folder=args.val_noise_folder



    def train(self):
        print('Load training image of undamaged screw ..')  
        Y_train,Y_train_filenames=dataset.load_data(self.args,self.training_folder)
        print('Load validation image of undamaged screw ..')  
        Y_val,Y_val_filenames=dataset.load_data(self.args,self.val_folder)


        print('Undamaged screw Y_train shape:',Y_train.shape)    
        print('Undamaged screw Y_val shape:',Y_val.shape)    

        if(self.Noise_Type=='CycleGAN'):
            print('load train Noise Input..') 
            
            x_train_noisy,x_train_generated_names,x_val_noisy,x_val_generated_names=self.CycleGAN.generate_synthetic_damaged_screw()

        elif(self.Noise_Type=='Gaussian'):
            x_train_noisy = Y_train + self.Noise_Proportion * np.random.normal(loc=0.0, scale=1.0, size=Y_train.shape)  
            x_val_noisy = Y_val + self.Noise_Proportion * np.random.normal(loc=0.0, scale=1.0, size=Y_val.shape)
            x_train_noisy = np.clip(x_train_noisy, 0., 1.)  
            x_val_noisy = np.clip(x_val_noisy, 0., 1.)
        elif(self.Noise_Type=='Pepper'):
            x_train_noisy = util.add_pepper_noise(Y_train, self.Noise_Proportion)
            x_val_noisy = util.add_pepper_noise(Y_val, self.Noise_Proportion)
            x_train_noisy = np.clip(x_train_noisy, 0., 1.)  
            x_val_noisy = np.clip(x_val_noisy, 0., 1.)
        elif(self.Noise_Type=='Normal'):
            x_train_noisy = Y_train 
            x_val_noisy = Y_val         
               
     

        print('damaged screw x_train shape:',x_train_noisy.shape)
        print('damaged screw x_val shape:',x_val_noisy.shape)
        dataset.check_pair_training_samples(self.args,Y_train[:5],x_train_noisy[:5])
        history=self.AE.train(x_train_noisy,Y_train,x_val_noisy,Y_val)
        util.plot_training_history(history, self.result_img)
        
        
    def test(self):
        
        filenames = gl.glob(self.test_folder+'/*.jpg')
        filenames.extend(gl.glob(self.test_folder+'/*.png'))
        dataset=[]
        filename_list=list()
        
        for file in filenames:
            dataset.append(np.array(cv2.imread(file,0)))
            name=file.split('/')[-1]
            filename_list.append(name)
            
        x_test = np.asarray(dataset)
        x_test = x_test.astype('float32') / 255.
        x_test = np.reshape(x_test, (len(x_test), self.image_size, self.image_size, 1)) 


        autoencoder = load_model(self.args.model_path,custom_objects={'custom_loss':self.AE.custom_loss})

        start = time.time()
        Restored_images = autoencoder.predict(x_test)
        end = time.time()
        seconds = (end - start)/x_test.shape[0]
        FPS = 1 / seconds


        for index in range(Restored_images.shape[0]):
            output = Restored_images[index]*255
            cv2.imwrite(f'{self.recontruction_folder}/{filename_list[index]}', output)

    def Generate_results(self):
        filelist = os.listdir(self.test_folder)
        for pic_name in filelist:

            imageA = cv2.imread(f'{self.test_folder}/{pic_name}')
            imageB = cv2.imread(f'{self.recontruction_folder}/{pic_name}')
            

            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
           
 
            (score, diff) = ssim(grayA, grayB, full=True)
            if(score!=1):
                diff = (diff * 255).astype("uint8")

                ret1, th1 = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)  #方法选择为THRESH_OTSU
                
                kernel = morphology.disk(3)
                mask = morphology.opening(th1, kernel)
                mask=255-mask

                if(self.args.noise_filter):
                    mask=self.Imageprocess.Noise_filter(mask)
                    head_mask = np.ones((256,256), dtype=np.int)*255
                    if(self.args.Screw_Type=='Z_M6x20-200'):
                        head_mask[0:30,:]=0
                    else:
                    #for others
                        head_mask[0:100,:]=0
                    # head_mask[240:256,:]=0
                    mask=np.bitwise_and(mask,head_mask)


                cv2.imwrite(f'{self.residual_folder}/{pic_name}_mask.png',mask)

                mask=255-mask
                vis=util.set_img_color(imageA.copy(), mask, weight_foreground=0.3, grayscale='grayscale')
                # mask *= 255
                cv2.imwrite(f'{self.residual_folder}/{pic_name}_org.png',imageA)
                cv2.imwrite(f'{self.residual_folder}/{pic_name}_res.png',imageB)
                cv2.imwrite(f'{self.residual_folder}/{pic_name}_diff.png',vis)



