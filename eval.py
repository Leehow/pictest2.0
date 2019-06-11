"""
Usage:
  # Create analy a image :
  python eval.py --image=abc.png/jpg  --name=good_picture
  save the images and jsons in static, update the list in static/list/list.json
"""
import numpy as np
import tensorflow as tf
import time
import json

from matplotlib import pyplot as plt
from PIL import Image

from tesserocr import PyTessBaseAPI, RIL

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
import configparser

config = configparser.ConfigParser()
config.read("conf.config", encoding="utf-8")

flags = tf.app.flags
flags.DEFINE_string('image', '', 'A image file to analys')
flags.DEFINE_string('name', '', 'Name of the image')
FLAGS = flags.FLAGS


class TOD(object):
    def __init__(self, image, threshold=0.5):
        output_model_u = config.get('config', 'cf_output_model')
        graf_u = config.get('config', 'cf_graf')
        # text在配置文件中的id
        texts_id_u = config.get('config', 'cf_texts_id')
        # num_class_u = config.get('tenssor', 'ts_num_class')
        mode_u = config.get('tenssor', 'ts_mode')
        line_thickness_u = config.get('tenssor', 'ts_line_thickness')

        self.PATH_TO_CKPT = output_model_u
        self.PATH_TO_LABELS = graf_u
        # self.NUM_CLASSES = int(num_class_u)
        self.TEXTSID = int(texts_id_u)
        # 至少百分之多少准确度
        self.THRESHOLD = float(threshold)
        # 画线粗度
        self.LINE_THI = int(line_thickness_u)
        self.detection_graph = self._load_model()
        self.category_index = self._load_label_map()

        self.mode = mode_u
        self.eval_image = image
        self.image_width = ''
        self.image_height = ''

    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    # 获取label类别
    def _load_label_map(self):
        category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS, use_display_name=True)
        return category_index

    # 用rgb颜色打开图片，并载入图片的长宽信息，图片由构造函数加载
    def _open_image(self):
        image = Image.open(self.eval_image).convert('RGB')
        (self.image_width, self.image_height) = image.size

        return image

    # 将图片用numpy reshape
    def load_image_into_numpy_array(self):
        image = self._open_image()
        return np.array(image.getdata()).reshape(
            (self.image_height, self.image_width, 3)).astype(np.uint8)

    # 从官网上摘下来的，用来检测一张图片
    def run_inference_for_single_image(self):
        graph = self.detection_graph
        # image = self.eval_image
        image_np = self.load_image_into_numpy_array()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image = np.expand_dims(image_np, axis=0)

        # 如果是gpu去掉显存限制
        if self.mode == 'GPU':
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[1], image.shape[2])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: image})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.int64)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def check_ocr(self):
        image = self._open_image()
        image_width = self.image_width
        image_height = self.image_height
        texts_id = self.TEXTSID
        with PyTessBaseAPI() as api:
            api.SetImage(image)
            boxes = api.GetComponentImages(RIL.TEXTLINE, True)

            boxes_out = np.empty([0, 4])
            class_out = []
            confi_out = []
            for i, (im, box, _, _) in enumerate(boxes):
                # im is a PIL image object
                # box is a dict with x, y, w and h keys
                api.SetRectangle(box['x'], box['y'], box['w'], box['h'])

                # 这个是输出具体文字的方法，需要时去掉注释
                ocrResult = api.GetUTF8Text()
                conf = api.MeanTextConf()

                ymin = float(box['y'] / image_height)
                xmin = float(box['x'] / image_width)
                ymax = float((box['y'] + box['h']) / image_height)
                xmax = float((box['x'] + box['w']) / image_width)
                bb = np.array([ymin, xmin, ymax, xmax])
                bb = bb.reshape((1, 4))

                # print("x:%s y:%s w:%s h:%s" % (box['x'], box['y'], box['w'], box['h']))
                # print("w:%s h:%s" % (image_width, image_height))
                # print("ymin:%s xmin:%s ymax:%s xmax:%s" % (ymin, xmin, ymax, xmax))
                boxes_out = np.append(boxes_out, bb, axis=0)
                class_out.append(texts_id)
                if conf == 0:
                    conf = 99
                conf = int(conf) / 100
                confi_out.append(conf)
                # print(ocrResult)
                # print(conf)
                # print(u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
                #       "confidence: {1}, text: {2}".format(i, conf, ocrResult, **box))
            out = []
            out.append(boxes_out)
            out.append(class_out)
            out.append(confi_out)

            return out

    # 用来给图片标框，只要输入图片，框的位置，类别和分数
    # --------------box的4个参数是ymin, xmin, ymax, xmax
    def get_labels(self, image, boxes, classes, scores):
        threshold = self.THRESHOLD
        category_index = self.category_index
        line = self.LINE_THI

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            boxes,
            classes,
            scores,
            category_index,
            max_boxes_to_draw=1000,
            min_score_thresh=threshold,
            # instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=line)


    # 将数据放入json格式
    def setJson(self, boxes, classes, scores, arr = []):
        ii = len(arr)
        im_width = self.image_width
        im_height = self.image_height
        threshold = self.THRESHOLD
        category_index = self.category_index
        for c in range(boxes.shape[0]):
            if scores[c] > threshold:
                y_min = boxes[c][0] * im_height
                x_min = boxes[c][1] * im_width
                y_max = boxes[c][2] * im_height
                x_max = boxes[c][3] * im_width
                class_name = ''
                if classes[c] in category_index.keys():
                    class_name = category_index[classes[c]]['name']

                sc = int(scores[c] * 100) / 100

                # print("序号：%s x: %s y:%s h: %s w:%s 名字:%s 分数:%s" % (c + 1, x_c, y_c, h_c, w_c, class_name, sc))
                # print("序号：%s ymin: %s xmin:%s ymax: %s xmax:%s 名字:%s 分数:%s" % (c + 1, boxes[c][0], boxes[c][1], boxes[c][2], boxes[c][3], class_name, sc))
                item = [{'id': ii, 'class name': class_name, 'scores': float(sc),
                         'y_min': int(y_min), 'x_min': int(x_min),
                         'y_max': int(y_max), 'x_max': int(x_max)}]
                ii += 1
                arr.append(item)
        return arr

    def save_image(self, image_np, out_png_path):
        im_height = float(self.image_height) / 100
        im_width = float(self.image_width) / 100

        fig = plt.gcf()
        fig.set_size_inches(im_width / 3, im_height / 3)

        plt.imshow(image_np)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        fig.savefig(out_png_path, format='png', transparent=True, dpi=300, pad_inches=0)


def check(image_path, out_pic_file, out_json_file):
    # 最小正确率
    min_score_u = config.get('tenssor', 'ts_min_score')
    threshold = float(min_score_u)

    # 识别图片， threshold参数是识别率为threshold标出来否则不标
    detecotr = TOD(image_path, threshold)
    image_np = detecotr.load_image_into_numpy_array()
    output_dict = detecotr.run_inference_for_single_image()

    boxes = output_dict['detection_boxes']
    classes = output_dict['detection_classes']
    scores = output_dict['detection_scores']

    # 获取ocr识别文字
    out = detecotr.check_ocr()

    boxes_2 = out[0]
    classes_2 = out[1]
    scores_2 = out[2]
    # 用来给图片标框，只要输入图片，框的位置，类别和分数（ 注意！ boxes为numpy二维数组！ 其他为一维数组）
    # --------------box的4个参数是ymin, xmin, ymax, xmax
    detecotr.get_labels(image_np, boxes_2, classes_2, scores_2)
    detecotr.get_labels(image_np, boxes, classes, scores)

    # 最终以json格式输出
    Json_output = []
    Json_output = detecotr.setJson(boxes, classes, scores, Json_output)
    Json_output = detecotr.setJson(boxes_2, classes_2, scores_2, Json_output)
    # print(output)
    # print(len(output))
    # -----------------------------------------------将json写入文件中
    with open(out_json_file, 'w') as f:
        json.dump(Json_output, f)
    print('Seccess to save json!')

    # -----------------------------------------------保存图像（已去掉白边和坐标轴）
    detecotr.save_image(image_np, out_pic_file)
    print('Seccess to save image!')

    return Json_output

def check_save(image_path, name=''):
    if not image_path:
        print('no picture')
        return 0
    pic_path_u = config.get('config', 'cf_pic_path')
    xml_path_u = config.get('config', 'cf_xml_path')
    list_file_u = config.get('config', 'cf_list_file')
    t = time.time()
    t = int(t)
    t = str(t)
    if not name:
        name = t
    out_pic_file = pic_path_u + '/pic_' + t + '.png'
    out_json_file = xml_path_u + '/json_' + t + '.json'
    check(image_path, out_pic_file, out_json_file)

    # -----------------------------------------------将id保存在json列表里
    with open(list_file_u, 'r') as f:
        data = json.load(f)
        item = [{'id': t, 'name': str(name)}]
        data.append(item)

    with open(list_file_u, 'w') as f:
        json.dump(data, f)

    print('Seccess to update json list!')


def main(_):
    check_save(FLAGS.image, FLAGS.name)

if __name__ == '__main__':
    tf.app.run()

