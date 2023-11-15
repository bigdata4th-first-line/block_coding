import cv2
import numpy as np

from keras.models import load_model

import warnings
warnings.filterwarnings(action='ignore')


class Predict():
    def __init__(self, image_path):
        self.input_img = cv2.imread(image_path)
        self.img = cv2.resize(self.input_img, (640, 640))
    
    
    def _preprocessing(self, label):
        
        crop_img = self.img[int(label['y1']):int(label['y2']), 
                        int(label['x1']):int(label['x2'])]
        
        out = crop_img.copy()
        out = 255 - out
        
        # 48 x 48
        output_img_48 = cv2.resize(out, (48, 48), interpolation = cv2.INTER_AREA)

        # 그레이 스케일 및 이진화
        gray_img = cv2.cvtColor(output_img_48, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)
        
        X_test = np.array(th1)
        input_shape = 2304
        X_test = X_test / 255
        X_test = X_test.reshape(-1, input_shape)
        
        return X_test
    
    
    def predict(self, sorted_label):
        arrow_result_list = []
        number_result_list = []
        action_result_list = []


        for i in sorted_label:
            X_test = self._preprocessing(i)
            
            if (i['label'] == 'up') | (i['label'] == 'down') | (i['label'] == 'left') | (i['label'] == 'right'):
                loaded_model = load_model('./model/arrow_model_01.h5')

                test_pred = loaded_model.predict(X_test)
                for i, a in enumerate(test_pred[0]):
                    if a == test_pred.max():
                        if i == 0:
                            arrow_result_list.append('d')
                        elif i == 1:
                            arrow_result_list.append('l')
                        elif i == 2:
                            arrow_result_list.append('r')
                        elif i == 3:
                            arrow_result_list.append('u')
                
            else:
                loaded_model = load_model('./model/number_model_01.h5')

                test_pred = loaded_model.predict(X_test)
                for i, a in enumerate(test_pred[0]):
                    if a == test_pred.max():
                        if i == 0:
                            number_result_list.append(2)
                        elif i == 1:
                            number_result_list.append(3)
                        elif i == 2:
                            number_result_list.append(4)
                        elif i == 3:
                            number_result_list.append(5)
                            
        
        # action predict
        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)

        for i in range(len(sorted_label)):
            if i % 2 == 0:
                if type(sorted_label[i]['label']) == str:
                    x = int(sorted_label[i]['cx'] + (sorted_label[i]['cx'] - sorted_label[i+1]['cx']) * 2)
                    y = int(sorted_label[i]['cy'] + (sorted_label[i]['cy'] - sorted_label[i+1]['cy']) * 2)
                    action_pixel = (y, x)
                elif type(sorted_label[i]['label']) == int:
                    x = int(sorted_label[i+1]['cx'] + (sorted_label[i+1]['cx'] - sorted_label[i]['cx']) * 2)
                    y = int(sorted_label[i+1]['cy'] + (sorted_label[i+1]['cy'] - sorted_label[i]['cy']) * 2)
                    action_pixel = (y, x)
                    
                if img_hsv[action_pixel][0] < 50:
                    action_result_list.append('Run')
                elif img_hsv[action_pixel][0] < 115:
                    action_result_list.append('Hand')
                else:
                    action_result_list.append('Jump')
        
        return arrow_result_list, number_result_list, action_result_list