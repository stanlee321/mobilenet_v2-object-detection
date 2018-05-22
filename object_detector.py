import numpy as np
import multiprocessing
import tensorflow as tf

# Google's object detection API libs
# Here are the imports from the object detection module.
from libs.tools import DetectionParams

class ObjectDetection(multiprocessing.Process):


    def __init__(self, out_pipe, send_pipe):
        """
            Object Detection class, it uses ssdlite_mobilenet_v2_coco_2018_05_09
            this class is used to detect objects in a image_array


        :param out_pipe: In of pipe to receive input image_arr
        :param send_pipe: Out of pipe to send detections from Class

        :output (box predictions)
        """
        super(ObjectDetection, self).__init__()


        # Input  pipe connection to this clas
        self.out_pipe = out_pipe
        self.send_pipe = send_pipe

        # Load a (frozen) Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(DetectionParams.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name = '')
                print('OBJECT DETECTION MODEL LOADED')

    def run(self):

        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                while True:
                    image_array = self.out_pipe.recv()
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_numpy_expanded = np.expand_dims(image_array, axis=0)
                    image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

                    # Each box represents a part of the image where a particular object was detected.
                    boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_numpy_expanded})

                    self.send_pipe.send((boxes, scores, classes, num_detections))
