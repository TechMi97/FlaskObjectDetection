import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory,Response
from werkzeug import secure_filename
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import json
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util
from flask import jsonify,make_response
import requests
import re


# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_CKPT= "C:\\Users\\Developer Tayub\\Downloads\\reproduceflipfrozen_inference_graph.pb"
PATH_TO_LABELS = "C:\\Users\\Developer Tayub\\Downloads\\10018-12131-12221_label_mapha1.pbtxt"


NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


#desiease path and score 

#download image
@app.route('/upload', methods=['POST'])
def upload():
    print("anything for thou")
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # filename="huhu"+filename
        link=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(link)
        
        print(link)
        path=link
        # ----- NEW CODE ---- ##
        PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
        TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR,filename.format(i)) for i in range(1, 2) ]
        IMAGE_SIZE = (12, 8)
        images_namas=[]
        images_namas.append(filename)
        responbalik= []
        
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                for image_path in TEST_IMAGE_PATHS:
                    image = Image.open(image_path)
                    image_np = load_image_into_numpy_array(image)
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.50)

                    im = Image.fromarray(image_np)
                    im.save('uploads/'+filename)  

                    
                    objects = []
                    objectsArray = [] 
                    threshold = 0.5
                    for index, value in enumerate(classes[0]):
                        object_dict = {}
                        if scores[0, index] > threshold: 
                            nama =(category_index.get(value)).get('name')
                            scora=scores[0,index]
                            scora2=str(int(100*scora))+"%"
                            display = '{}: {}%'.format(nama,int(100*scora))
                            objects.append(display)
                            objectsArray.append({
                                "name":nama,
                                "score":scora2,
                                })
                            print(objects)
                                                
                    
                    # for index, value in enumerate(classes[0]):
                    #   object_dict = {}
                    #   if scores[0, index] > threshold:
                    #     object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                    #                         scores[0, index]
                    #     objects.append(object_dict)
                    # print(objects)

        result = {
            "data": objectsArray,
            "path": path 
        }
 
        # return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        return jsonify(result), 200


# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
#     TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR,filename.format(i)) for i in range(1, 2) ]
#     IMAGE_SIZE = (12, 8)
#     images_namas=[]
#     images_namas.append(filename)
#     responbalik= []
    
#     with detection_graph.as_default():
#         with tf.Session(graph=detection_graph) as sess:
#             for image_path in TEST_IMAGE_PATHS:
#                 image = Image.open(image_path)
#                 image_np = load_image_into_numpy_array(image)
#                 image_np_expanded = np.expand_dims(image_np, axis=0)
#                 image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#                 boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#                 scores = detection_graph.get_tensor_by_name('detection_scores:0')
#                 classes = detection_graph.get_tensor_by_name('detection_classes:0')
#                 num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#                 (boxes, scores, classes, num_detections) = sess.run(
#                     [boxes, scores, classes, num_detections],
#                     feed_dict={image_tensor: image_np_expanded})
#                 # print("huhu")
#                 vis_util.visualize_boxes_and_labels_on_image_array(
#                     image_np,
#                     np.squeeze(boxes),
#                     np.squeeze(classes).astype(np.int32),
#                     np.squeeze(scores),
#                     category_index,
#                     use_normalized_coordinates=True,
#                     line_thickness=8)

#                 # print(vis_util.visualize_boxes_and_labels_on_image_array.i)

#               #   print([category_index.get(i) for i in classes[0]])
#               # print(scores)
#                 # # print(class_name)
#                 # print(display_str)
#                 im = Image.fromarray(image_np)
#                 im.save('uploads/'+filename)  
#                 # objects = []
#                 # threshold = 0.5
#                 # for index, value in enumerate(classes[0]):
#                 #   object_dict = {}
#                 #   if scores[0, index] > threshold:
#                 #       object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
#                 #                            str(scores[0, index])
#                 #       objects.append(object_dict)
#                 # print(objects)
                
#                 objects = []
#                 threshold = 0.5
#                 for index, value in enumerate(classes[0]):
#                     object_dict = {}
#                     if scores[0, index] > threshold: 
#                         nama =(category_index.get(value)).get('name')
#                         scora=scores[0,index]
#                         display = '{}: {}%'.format(nama,int(100*scora))
#                         objects.append(display)
#                         # object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
#                         #                      scores[0, index]
#                         # print((category_index.get(value)).get('name')+" "+ str(scores[0,index]))
#                         # print(float(str(round(scores[0,index], 3))))
#                         # display = '{}: {}%'.format(nama,int(100*scora))
#                         # print(display)    
#                         # objects.append(object_dict)


#                 print(objects)
#                 # result=json.dumps(objects)
#                 # for x in objects:
#                 #     for a in x:
#                 #         print(a)


#        #              responbalik.append({
#        #              "class": class_names[int(classes[0][i])],
#        #              "confidence": float("{0:.2f}".format(np.array(scores[0][i])*100))
#        #              })
#     #             responbalik.append({
#                 #             "image": image_namas[0],
#                 #             "detections": responses
#                 # })
#     # result = {"objects": objects}
#     return jsonify(objects), 200           
#     # return send_from_directory(app.config['UPLOAD_FOLDER'],
#     #                            filename)
#image and data
@app.route('/json/<filename>')
def tst(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)

