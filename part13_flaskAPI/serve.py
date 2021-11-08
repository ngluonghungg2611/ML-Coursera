'''
    Có chức năng khởi taok và run app. Khởi tạo các API chp project
'''
import re
from PIL import Image
from flask import Flask, request
import numpy as np
import cv2
from numpy.lib.type_check import imag
from tensorflow.keras.models import load_model
import flask
import json
import io
import utils
import imagenet
import hyper as hp

#   Khởi tạo model
# Khởi tạo model.
global model 
model = None
# Khởi tạo flask app
app = Flask(__name__)

#Khai báo các route 1 cho API
@app.route("/", methods=["GET"])
# Khai báo hàm xử lý dữ liệu.
def _hello_world():
	return "Hello world"

# Khai báo các route 2 cho API
@app.route("/predict", methods=["POST"])
# Khai báo hàm xử lý dữ liệu.
def _predict():
	data = {"success": False}
	if request.files.get("image"):
		# Lấy file ảnh người dùng upload lên
		image = request.files["image"].read()
		# Convert sang dạng array image
		image = Image.open(io.BytesIO(image))
		# resize ảnh
		image_rz = utils._preprocess_image(image,
			(hp.IMAGE_WIDTH, hp.IMAGE_HEIGHT))
		# Dự báo phân phối xác suất
		dist_probs = model.predict(image_rz)
		# argmax 5
		argmax_k = np.argsort(dist_probs[0])[::-1][:5]
		# classes 5
		classes = [imagenet.classes[idx] for idx in list(argmax_k)]
		# probability of classes
		classes_prob = [dist_probs[0, idx] for idx in list(argmax_k)]	
		data["probability"] = dict(zip(classes, classes_prob))
		data["success"] = True
	return json.dumps(data, ensure_ascii=False, cls=utils.NumpyEncoder)

if __name__ == "__main__":
	print("App run!")
	# Load model
	model = utils._load_model()
	app.run(debug=False, host=hp.IP, threaded=False)