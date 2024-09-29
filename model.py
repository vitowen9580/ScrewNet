from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mean_squared_error, Huber
import glob 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import cv2
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
from tensorflow.keras.layers import *
import util
import dataset
import logging
import os
from keras.callbacks import EarlyStopping


def lr_decay(epoch, lr, total_epochs):
    decay_rate_90 = 0.1  # 90%時的衰減率
    decay_rate_95 = 0.01  # 95%時的衰減率

    if epoch >= int(total_epochs * 0.95):
        return lr * decay_rate_95
    elif epoch >= int(total_epochs * 0.90):
        return lr * decay_rate_90
    return lr


# 定义显示学习率的回调
class PrintLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)) 
        print(f'\nEpoch {epoch+1}: Current learning rate is {lr:.6f}')

# 使用学习率调度器
# lr_scheduler = LearningRateScheduler(lr_decay)
print_lr = PrintLearningRate()


def mse_loss(y_pred,y_true):
    return tf.keras.losses.MeanSquaredError()



# 定义自动编码器类
class AE:
    def __init__(self, args):
        self.args = args
        self.autoencoder = None
        self.build_model()


    # 自定义损失函数
    def custom_loss(self,y_true, y_pred):
        mse_loss = mean_squared_error(y_true, y_pred)
        huber_loss = Huber()(y_true, y_pred)
        # Compute magnitude similarity loss
        magnitude_sim_loss = tf.reduce_mean(tf.abs(tf.norm(y_true) - tf.norm(y_pred)))
        # Weighting factors
        alpha = 0.5

        # Combined loss
        combined_loss = alpha * mse_loss + (1-alpha) * huber_loss 
        return combined_loss

    def conv2d_block(self, input_tensor, n_filters, kernel_size=3, batchnorm=True):
        # First layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("elu")(x)
        
        # Second layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("elu")(x)
        return x
    
    # 建立自动编码器模型
    def build_model(self):
        n_filters = 16
        dropout = 0.5
        batchnorm = True
        input_img = Input(shape=(self.args.image_size, self.args.image_size, 1))

        c1 = self.conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout*0.5)(p1)

        c2 = self.conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = self.conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)
        
        c4 = self.conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
        encoder = Model(inputs=[input_img], outputs=[c4])

        u5 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')(c4)
        u5 = Dropout(dropout)(u5)
        c5 = self.conv2d_block(u5, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

        u6 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

        u7 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
        decoder = Model(inputs=[input_img], outputs=[c7])

        outputs = Conv2D(1, (1, 1), activation='sigmoid', name='visualized_layer')(c7)

        self.autoencoder = Model(inputs=[input_img], outputs=[outputs])

        # 编译模型
        self.autoencoder.compile(optimizer=Adam(lr=self.args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss=self.custom_loss, metrics=['mean_squared_error'])
        print(self.autoencoder.summary())




    def train(self, x_train_noisy, Y_train, x_val_noisy, Y_val):
        # 训练模型
        total_epochs = self.args.epochs  # 從self.args獲取總的epochs數
        lr_scheduler = LearningRateScheduler(lambda epoch, lr: lr_decay(epoch, lr, total_epochs))
        # early_stop = EarlyStopping(monitor='mean_squared_error', patience=10, mode='min', verbose=1)

        history = self.autoencoder.fit(
            x_train_noisy, Y_train,
            epochs=self.args.epochs, 
            batch_size=32,
            shuffle=True,
            validation_data=(x_val_noisy, Y_val),
            callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False), lr_scheduler, print_lr]
        )
        self.autoencoder.save(self.args.model_path)
        return history


import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from arch import define_Gen, define_Dis
from PIL import Image

from tqdm import tqdm
from torch.utils.data import Dataset

class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith(('.jpg', '.png', '.jpeg'))]
        
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # 打开图像并转换为RGB
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)  # 返回图像张量和文件名
class CycleGAN:
    def __init__(self, args):
        self.args = args
        self.ngf = 64
        self.no_dropout = False
        self.norm = 'instance'
        self.batch_size=1
        self.num_workers=1

    def generate_synthetic_damaged_screw(self):
        # 图像预处理
        transform = transforms.Compose(
            [transforms.Resize((self.args.image_size, self.args.image_size)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )


        trainA_data = CustomImageFolder(f'{self.args.Screw_Type}/trainA', transform=transform)  # 使用自定义的 Dataset
        trainA_loader = torch.utils.data.DataLoader(trainA_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        valA_data = CustomImageFolder(f'{self.args.Screw_Type}/valA', transform=transform)  # 使用自定义的 Dataset
        valA_loader = torch.utils.data.DataLoader(valA_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        Gba = define_Gen(input_nc=3, output_nc=3, ngf=self.ngf, netG='resnet_9blocks', norm=self.norm,
                         use_dropout=not self.no_dropout)
        path = f'checkpoints/{self.args.Screw_Type}/latest_lamda10.ckpt'
        print(f'load pre-trained generator : {path}..')
        try:
            
            ckpt = util.load_checkpoint(path)
            Gba.load_state_dict(ckpt['Gba'])
        except:
            print(' [*] No checkpoint!')


        x_train_noisy = []
        x_train_generated_names = []
        x_val_noisy = []
        x_val_generated_name = []


        trainB_folder=f'{self.args.Screw_Type}/trainB'
        if not os.path.exists(trainB_folder):
            os.makedirs(trainB_folder)

        valB_folder=f'{self.args.Screw_Type}/valB'
        if not os.path.exists(valB_folder):
            os.makedirs(valB_folder)


        for a_real_test, img_name in tqdm(trainA_loader, desc="Generating images"):
            a_real_test = Variable(a_real_test, requires_grad=True)
            a_real_test = a_real_test.to('cuda:0')
            Gba.eval()
            with torch.no_grad():
                b_fake_test = Gba(a_real_test)
            b_fake_test_pic = ((b_fake_test).data + 1) / 2.0
            
            b_fake_test_np = b_fake_test_pic.squeeze().cpu().numpy()  
            
            
            b_fake_test_np = np.transpose(b_fake_test_np, (1, 2, 0))
            if b_fake_test_np.dtype != np.float32:
                b_fake_test_np = b_fake_test_np.astype(np.float32)
            gray_img = cv2.cvtColor(b_fake_test_np, cv2.COLOR_RGB2GRAY)
            gray_img_tensor = torch.tensor(gray_img, dtype=torch.float32).unsqueeze(0)  # 添加一个通道维度


            gray_img_tensor_cpu = gray_img_tensor.cpu() 
            gray_img_np = gray_img_tensor_cpu.numpy()  
             
            x_train_noisy.append((gray_img_np))  
            x_train_generated_names.append(img_name[0])

            torchvision.utils.save_image(gray_img_tensor, f'{trainB_folder}/{img_name[0]}')


        for a_real_test, img_name in tqdm(valA_loader, desc="Generating images"):
            a_real_test = Variable(a_real_test, requires_grad=True)
            a_real_test = a_real_test.to('cuda:0')
            Gba.eval()
            with torch.no_grad():
                b_fake_test = Gba(a_real_test)
            b_fake_test_pic = ((b_fake_test).data + 1) / 2.0
            b_fake_test_np = b_fake_test_pic.squeeze().cpu().numpy()  
            b_fake_test_np = np.transpose(b_fake_test_np, (1, 2, 0))
            if b_fake_test_np.dtype != np.float32:
                b_fake_test_np = b_fake_test_np.astype(np.float32)
            gray_img = cv2.cvtColor(b_fake_test_np, cv2.COLOR_RGB2GRAY)
            gray_img_tensor = torch.tensor(gray_img, dtype=torch.float32).unsqueeze(0)  # 添加一个通道维度
            gray_img_tensor_cpu = gray_img_tensor.cpu() 
            gray_img_np = gray_img_tensor_cpu.numpy()  
             
            x_val_noisy.append((gray_img_np))  
            x_val_generated_name.append(img_name[0])

            torchvision.utils.save_image(gray_img_tensor, f'{valB_folder}/{img_name[0]}')




        x_train_noisy=np.array(x_train_noisy)
        x_train_noisy = np.transpose(x_train_noisy, (0,2,3,1))
        x_val_noisy=np.array(x_val_noisy)
        x_val_noisy = np.transpose(x_val_noisy, (0,2,3,1))

        return x_train_noisy,x_train_generated_names, x_val_noisy,x_val_generated_name