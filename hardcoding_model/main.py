import cv2
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from getindex import GetIndex
from model import Predict
import getimg


api = FastAPI()

@api.get('/predict')
def startmain(img_name):
    getimg.get_image()
    
    IMG_PATH = './images/' + img_name + '.jpg'
    
    print(IMG_PATH)
    
    img = cv2.imread(IMG_PATH)
    
    arrow_idx_list, number_idx_list, action_pixel_list = GetIndex(IMG_PATH).get_idx_list()
    
    img_list, _ = GetIndex(IMG_PATH)._get_max_index()
    
    arrow_result = Predict().predict_arrow(arrow_idx_list, img_list)
    number_result = Predict().predict_number(number_idx_list, img_list)
    action_result = Predict().predict_action(action_pixel_list, img)
   
    response_data = {
        "direction": arrow_result,
        "number": number_result,
        "action": action_result
    }
    
    return JSONResponse(content=response_data)
