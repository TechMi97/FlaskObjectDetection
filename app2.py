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
from utils import ops as utils_ops
from utils import visualization_utils as vis_util
from flask import jsonify,make_response
import requests
import re


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

#till here all is the same


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def run_inference_for_single_image(image, graph):
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
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.2), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      # output_dict_array = []

      # for od in output_dict:
      #     od['num_detections'] = int(od['num_detections'][0])
      #     od['detection_classes'] = od[
      #         'detection_classes'][0].astype(np.uint8)
      #     od['detection_boxes'] = od['detection_boxes'][0]
      #     od['detection_scores'] = od['detection_scores'][0]
      #     if 'detection_masks' in od:
      #       od['detection_masks'] = od['detection_masks'][0]
            # output_dict_array.append(od)
  return output_dict

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
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            output_dict = run_inference_for_single_image(image_np, detection_graph)
            # output_dict_array = run_inference_for_single_image(image_np, detection_graph)
# Visualization of the results of a detection.
            #return arr as output_dict
            #
            for od in output_dict:
                print(od)
            #   
                vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  od['detection_boxes'],
                  od['detection_classes'],
                  od['detection_scores'],
                  category_index,
                  instance_masks=od.get('detection_masks'),
                  use_normalized_coordinates=True,
                  line_thickness=8)
            # print(output_dict['detection_boxes'])
            print(od['detection_classes'])
            # print(output_dict['detection_scores'])

            im = Image.fromarray(image_np)
            im.save('uploads/'+filename)      

 
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
      




@app.route('/json/<filename>')
def tst(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)

