import os
import cv2
import glob
import json
import time
from PJ import constant
from datetime import timedelta
from keras.models import load_model
from werkzeug.utils import secure_filename
from PJ.recognize.code.Recognizer import Recognizer
from PJ.preprocess.code.PositionAdjuster import PositionAdjuster
from PJ.detect.code.ContextDetector import Detector as ContextDetector
from PJ.detect.code.ReferPointDetector import Detector as ReferPointDetector
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify

# 设置允许upload的文件格式
ALLOWED_EXTENSIONS = {'png', 'PNG', 'jpg', 'JPG', 'bmp'}


# 判断图片格式
def allow_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)

# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


# 加载检测识别所需的模型,并设置中间文件保存路径
def run_detect(project_path, image_path):
    rp_detector_pb_path = os.path.join(project_path, "PJ/detect/model", "MobileNet_rp.pb")
    rp_detector = ReferPointDetector(pb_path=rp_detector_pb_path)
    print("load rp_dector")
    tecognizer_pb_path = os.path.join(project_path, "PJ/recognize/model", "weights_densenet.h5")
    recognizer = Recognizer(tecognizer_pb_path)
    print("load recognizer")
    rp_mask_txt = os.path.join(project_path, "static/image_mask", "template_mask.txt")
    position_adjuster = PositionAdjuster(rp_mask_txt, rp_detector, recognizer)
    print("init RP")

    _, image_name = os.path.split(image_path)
    image = cv2.imread(image_path)
    position_adjuster.run(image, image_name)

    image_path = os.path.join(project_path, "static/image_preprocess", image_name)
    image = cv2.imread(image_path)
    detect_out_path = os.path.join(project_path, "static/image_segment")
    txt_path = os.path.join(project_path, "static/image_mask", "03.txt")
    context_detector_pb_path = os.path.join(project_path, "PJ/detect/model", "MobileNet_1219.pb")
    context_detector = ContextDetector(pb_path=context_detector_pb_path, txt=txt_path)
    context_detector.run(image, True, detect_out_path)
    image_file_list = glob.glob(detect_out_path + "/*.*")
    dict = {}
    for image_file in image_file_list:
        result = recognizer.run(image_file)
        _, name_ext = os.path.split(image_file)
        name, extension = os.path.splitext(name_ext)
        dict[name] = result
    return jsonify(dict)


@app.route('/')
def index():
    # return 'success'
    return redirect(url_for('upload'))


# 上传文件
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        # 检查文件upload的类型
        if not (f and allow_file(f.filename)):
            return jsonify({"error": 404, "msg": "请检查上传的图片类型,仅限于png,jpg,bmp"})
        file_name = secure_filename(f.filename)
        # 设置保存upload文件所在路径
        base_path = os.path.dirname(__file__)
        upload_path = os.path.join(base_path, 'static/image_upload', file_name)
        f.save(upload_path)
        return jsonify({'file_name': file_name})
    return 'Failed'


# 检测/识别
@app.route('/detect', methods=['POST', 'GET'])
def detect():
    if request.method == 'POST':
        # 项目路径
        data = json.loads(request.get_data())
        image_name = data['image_name']
        print(image_name)
        base_path = os.path.dirname(__file__)
        image_path = os.path.join(base_path, 'static/image_upload', image_name)
        result = run_detect(constant.PROJECT_PATH, image_path)
        print(result)
        return result
    return 'Failed'


if __name__ == '__main__':
    app.run()
