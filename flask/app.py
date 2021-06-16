from flask import Flask, render_template, request, redirect
import triton_example_requests as infer
from werkzeug.utils import secure_filename
import cv2, json
import numpy as np
import hashlib
import os

hash_ = hashlib.sha256()

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('infer.html', input='../static/img/example_obj_raw.png', output='../static/img/example_obj_raw.png', opacity=0.4)

@app.route("/infer", methods=['POST'])
def infer_img():
	img = request.files['file'].read()
	real_FileName = request.files['file'].filename
	base_path = '../static/detect/'
	hash_.update(real_FileName.encode())
	hash_str = hash_.hexdigest()
	filename_before = hash_str + '_before.jpg'

	img = np.frombuffer(img, np.uint8)
	img = cv2.imdecode(img, cv2.IMREAD_COLOR)
	cv2.imwrite('./static/detect/' + filename_before, img)

	boxes, classes, scores = infer.inference(img, '192.168.0.101', 13100, 0.1)

	cnt = 0
	for detect in zip(boxes, classes, scores):
		if detect[0][2] > 0.1:
			x1, y1, x2, y2 = round(detect[0][1]), round(detect[0][0]), round(detect[0][3]), round(detect[0][2])
			text = "{} ({}%)".format(detect[1], round(detect[2]*100, 2))
			# 영문자는 한 글자당 5씩, 여백과 퍼센트 부분은 일정크기 이상으로 넘어가지 않으니 정수를 더함
			text_len = len(detect[1]) * 5 + int((x2-x1)/20) + 50
			print(x1, y1, x2, y2)
			cv2.rectangle(img, (x1, y1), (x1 + text_len, y1+int((y2-y1)/20)), (0,255,255), -1)
			cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,255), 1)
			cv2.putText(img, text, (x1 + int((x2-x1)/25), y1 + int((y2-y1)/25)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
		else:
			break
	
	
	filename_after = hash_str + '_after.jpg'
	cv2.imwrite('./static/detect/' + filename_after, img)

	return render_template('infer.html', input=base_path + filename_before, output=base_path + filename_after, opacity=1)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=15000, debug=True)
