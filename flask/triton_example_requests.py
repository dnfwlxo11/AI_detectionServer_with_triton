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
                "binary_data_output": True
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
        response = requests.post('http://' + host + ':' + str(port) + '/v2/models/mymodel/versions/1/infer', 
                                data=data, headers=header)
        print("* Thermal inference time: {:.2f}".format(time.time() - start_time))

        # TODO: parse header

        results = response.content[str(response.content).find("]}"):]
        results = np.frombuffer(results, np.float32).reshape((-1, 7))

        # 모델 결과를 보기 쉽게 변환하는 부분
        label = ['face']
        prediction = results

        boxes = prediction[:, 1:5]
        classes = [label[i - 1] for i in prediction[:, 6].astype(int)]
        scores = prediction[:, 5]

        # 디텍트된 박스의 좌표, 클래스, 스코어를 보여줌
        for i in range(len(scores)):
            if scores[i] > score:
                print('*'*75)
                print('box pos: {}\nclass: {}\nscore: {}'.format(boxes[i], classes[i], scores[i]))
                print('*'*75)
            else:
                break

        return boxes, classes, scores
        

    except Exception as e:
        print(e)
        return [0], [0], [0]
