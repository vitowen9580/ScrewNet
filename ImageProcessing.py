import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
import os
import random

class Imageprocess:
    def __init__(self,args):

        # self.methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
        #     'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        
        '''
        just fot Z_M6x20-200
        '''
        self.methods = ['cv2.TM_CCOEFF']
        self.args=args


    def Template_Matching(self,img_name,template_name,save_folder):

        image = cv2.imread(img_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h,w=gray.shape
        for index in range(4):
            slice_img=gray[:,index*int(w/4):int(w/4)*(index+1)]

            template = cv2.imread(template_name)
            # 轉換為灰度圖片
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            # 讀取測試圖片並將其轉化為灰度圖片

            # 執行邊緣檢測
            template = cv2.Canny(template, 50, 200)
            (tH, tW) = template.shape[:2]
            # 顯示模板
            #cv2.imshow("Template", template)
            found = None

            # 循環遍歷不同的尺度
            for scale in np.linspace(0.2, 1.0, 20)[::-1]:
                # 根據尺度大小對輸入圖片進行裁剪
                resized = imutils.resize(slice_img, width = int(slice_img.shape[1] * scale))
                r = slice_img.shape[1] / float(resized.shape[1])

                # 如果裁剪之後的圖片小於模板的大小直接退出
                if resized.shape[0] < tH or resized.shape[1] < tW:
                    break

            # 首先進行邊緣檢測，然後執行模板檢測，接著獲取最小外接矩形
                edged = cv2.Canny(resized, 50, 200)
                #匹配
                #TM_SQDIFF  匹配方法，歸一化的方法更好用
                result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                # 結果可視化
                # 繪製矩形框並顯示結果
                clone = np.dstack([edged, edged, edged])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                # cv2.imshow("Visualize", clone)
                # cv2.waitKey(0)

                # 如果發現一個新的關聯值則進行更新
                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r)

            # 計算測試圖片中模板所在的具體位置，即左上角和右下角的坐標值，並乘上對應的裁剪因子
            (_, maxLoc, r) = found
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

            # 繪製並顯示結果
            # cv2.rectangle(slice_img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            ROI_img=slice_img[startY:endY,startX:endX]
            ROI_img=cv2.resize(ROI_img,(self.args.image_size,self.args.image_size))
            # cv2.imshow(img_name,ROI_img)
            # cv2.waitKey(0)

            name=str(index)+'_'+img_name.split('/')[-1].split('\\')[-1]
            cv2.imwrite(save_folder+'/'+name,ROI_img)
            print(save_folder+'/'+name)


    def rename(self,readpath):
        

        f = os.listdir(readpath)
        # print(f)
        print(len(f))
        name_list=random.sample(range(len(f)), len(f))

        n = 0
        numb=0
        i = 0
        for i in f:
        # 設定舊檔名（就是路徑+檔名）
            oldname = f[n]

        # 設定新檔名
            # newname =str(numb) + '_.png'
            newname=str(name_list[n])+'.png'
        # 用os模組中的rename方法對檔案改名
            os.rename(readpath+oldname, readpath+newname)
            print(oldname, '======>', newname)

            n += 1
            numb+=1

        
    def Noise_filter(self,mask):
        h, w= mask.shape
        threshold = h/25 * w/25   #设定阈值

        # threshold = h/30 * w/30   #设定阈值
        #cv2.fingContours寻找图片轮廓信息

        contours,hierarch=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i]) #计算轮廓所占面积
            if area < threshold:                         #将area小于阈值区域填充背景色，由于OpenCV读出的是BGR值
                cv2.drawContours(mask,[contours[i]],-1, 0, thickness=-1)     #原始图片背景BGR值(84,1,68)
                # cv2.imshow('img',mask)
                # cv2.waitKey(0)
                continue
            
                    
        return mask