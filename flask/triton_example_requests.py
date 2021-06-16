import numpy as np
import time, os, json
import requests
import cv2

def inference(image, host, port, score):
    try:
        image = np.expand_dims(image, 0)
        image = image.astype(np.uint8)
        batch, row, col, ch = image.shape
        image = image.tobytes()
        
        data = image

        body = {
            "inputs":[
                {
                    "name":"image_arrays:0",
                    "shape":[batch,row,col,ch],
                    "datatype":"UINT8",
                    "parameters":{
                        "binary_data_size":len(data)
                    }
                }
            ],
            
            "parameters":{
                "binary_data_output": False
            }
        }

        # 결과값 파싱을 편하게 하기위한 공백 제거        
        body = json.dumps(body).replace(' ', '')
        body = body.encode()
        data = body + data

        infer_content_length = len(body)

        header = {
            "Content-Type": "application/octet-stream",
            "Inference-Header-Content-Length": str(infer_content_length),
            "Accept": "*/*"
        }
    
        start_time = time.time()
        response = requests.post('http://' + host + ':' + str(port) + '/v2/models/thermal_detection/versions/1/infer', 
                                data=data, headers=header)
        print("* Thermal inference time: {:.2f}".format(time.time() - start_time))

        # TODO: parse header

        data = json.loads(response.content.decode())['outputs'][0]
        results = data['data']
        results = np.array(results).reshape((-1, 7))

        # 모델 결과를 보기 쉽게 변환하는 부분
        label = {
            1:'car_normal', 
            2:'car_abnormal', 
            3:'factory_inside_normal', 
            4:'factory_inside_abnormal', 
            5:'factory_outside_normal', 
            6:'factory_outside_abnormal', 
            7:'fan_normal', 
            8:'fan_abnormal', 
            9:'panel_normal', 
            10:'panel_abnormal',
            11:'person_normal', 
            12:'person_abnormal', 
            13:'pipe_normal', 
            14:'pipe_abnormal', 
            15:'ship_normal', 
            16:'ship_abnormal', 
            17:'tank_normal', 
            18:'tank_abnormal', 
            19:'valve_normal', 
            20:'valve_abnormal'
        }

        prediction = results

        boxes = prediction[:, 1:5]
        classes_num = prediction[:, 6]
        classes = [label[i] for i in prediction[:, 6].astype(int)]
        scores = prediction[:, 5]

        # 디텍트된 박스의 좌표, 클래스, 스코어를 보여줌
        for i in range(len(scores)):
            if scores[i] > score:
                print('*'*75)
                print('box pos: {}\nclass: {}({})\nscore: {}'.format(boxes[i], classes[i], int(classes_num[i]), scores[i]))
                print('*'*75)
            else:
                break

        return boxes, classes, scores
        

    except Exception as e:
        print(e, 'asdfqwe')
        return [0], [0], [0]
