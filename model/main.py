import cv2
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from yolo import YoloLabel
from model import Predict
import getimg

api = FastAPI()

@api.get('/predict')
def startmain(img_name):
    
    getimg.get_image(img_name)
    
    img_path = './images/' + img_name + '.jpg'
    
    
    
    img = cv2.imread(img_path)
    
    sorted_label = YoloLabel().inference_image(img)
    
    arrow_result_list, number_result_list, action_result_list = Predict(img_path).predict(sorted_label)
    
    print(arrow_result_list)
    print(number_result_list)
    
    response_data = {
        "direction": arrow_result_list,
        "number": number_result_list,
        "action": action_result_list
    }
    
    return JSONResponse(content=response_data)