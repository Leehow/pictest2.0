from flask import Flask, render_template, abort, request
from datetime import timedelta

import os
import configparser
import json


#-------------------------------配置文件加载
conf = configparser.ConfigParser()
conf.read("conf.config", encoding="utf-8")
TESTPIC_DIR = conf.get('config', 'cf_pic_path')
TMP_PIC = conf.get('config', 'cf_tmp_pic')
TESTJSON_DIR = conf.get('config', 'cf_xml_path')
TESTPIC_LIST = conf.get('config', 'cf_list_file')

app = Flask(__name__)
app.secret_key = 'some_secret'

# 设置静态文件缓存过期时间
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
@app.route('/error')
def error():
    abort(404)

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@app.route('/')
def hello():
    return render_template('index.html')

#训练页面
@app.route('/pretrpage')
def pretrpage():
    return render_template('tr.html')

#训练过程
@app.route('/trpage')
def trpage():
    os.system("python xml_to_csv.py")
    os.system("python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record")
    os.system("python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record")
    os.system("python model_main.py")
    return 'Ok! <a href="/" onClick=”javascript :history.go(-1);”>back</a>'

#特定步数训练
@app.route('/trwsteps', methods=['GET', 'POST'])
def trwsteps():
    steps = request.form['steps']
    # print("python model_main.py --num_train_steps="+steps)
    os.system("python model_main.py --num_train_steps="+steps)
    os.system("python export_inference_graph.py --if_retrain=True")
    return 'Ok! <a href="/" onClick=”javascript :history.go(-1);”>back</a>'

#输出
@app.route('/output')
def output():
    os.system("python export_inference_graph.py")
    return 'Ok! <a href="/" onClick=”javascript :history.go(-1);”>back</a>'



#列出图片id列表
@app.route('/list_img', methods=['GET', 'POST'])
def list_img():
    with open(TESTPIC_LIST, 'r') as f:
        data = json.load(f)
        result = "<ul>"
        for d in data:
            idd = d[0]['id']
            name = d[0]['name']
            result = result+'<li>'+name+'&nbsp;&nbsp;<a href="/show_img?id='+idd+'&option=bild">picture</a>&nbsp;&nbsp;<a href="/show_img?id='+idd+'&option=json">json</a></li>'
        f.close()
    result = result+"</ul>"
    return result+'<p></p> <a href="/" onClick=”javascript :history.go(-1);”>back</a>'

#用id来显示图片或者json
@app.route('/show_img', methods=['GET', 'POST'])
def show_img():
    idd = request.args.get('id')
    option = request.args.get('option')
    img_path = TESTPIC_DIR+'/'
    if option == 'bild':
        img_name = 'pic_'+idd+'.png'
        imge = img_path+img_name
        ife = os.path.exists(imge)
        if ife == False:
            return render_template('404.html'), 404
        print("200 ok")
        return render_template('upload_ok.html', pic=imge)
    else:
        jso=img_path+'json_'+idd+'.json'
        ife=os.path.exists(jso)
        if ife == False:
            return render_template('404.html'), 404
        with open(jso, 'r') as f:
            data = json.load(f)
            f.close()
        print("200 ok")
        return json.dumps(data)

#上传界面
@app.route('/uppage')
def uppage():
    return render_template('upload.html')

#上传以后的处理
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['picture']
        name = request.form['name']
        if not f:
            return 'You should update picture <a href="/" onClick=”javascript :history.go(-1);”>back</a>'
        f.save(TMP_PIC)
        os.system("python eval.py --image=%s  --name=%s" %(TMP_PIC, name))
        return 'Ok! <a href="/" onClick=”javascript :history.go(-1);”>back</a>'
    else:
        return 'You should update picture <a href="/" onClick=”javascript :history.go(-1);”>back</a>'
