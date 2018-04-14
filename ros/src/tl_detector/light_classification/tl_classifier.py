from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2
import os

class TLClassifier(object):
    
    def __init__(self, graph_source):
        graph_file = graph_source + '/frozen_inference_graph.pb'
        self.TL_COLOR_GREEN = 1
        self.TL_COLOR_RED = 2
        self.TL_COLOR_YELLOW = 3
        self.TL_COLOR_UNKNOWN = 4
        self.confidence_cutoff = 0.8

        wd = os.path.dirname(os.path.realpath(__file__))
        # load frozen graph
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(wd + graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.image_tensor = graph.get_tensor_by_name('image_tensor:0')
        self.detection_scores = graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = graph.get_tensor_by_name('detection_classes:0')
        self.detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        self.graph = graph

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        image = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        with tf.Session(graph=self.graph) as sess:
            (_, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                        feed_dict={self.image_tensor: image})
        
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        max_scores = [j for (i,j) in zip(scores, classes) if i >= self.confidence_cutoff]
        if max_scores is None:
            return TrafficLight.UNKNOWN
        else:
            return self.get_tl_color(int(max_scores[0]))
    
    def get_tl_color(self, color_class):
        return {
            self.TL_COLOR_GREEN: TrafficLight.GREEN,
            self.TL_COLOR_RED: TrafficLight.RED,
            self.TL_COLOR_YELLOW: TrafficLight.YELLOW,
            self.TL_COLOR_UNKNOWN: TrafficLight.UNKNOWN
        }[color_class]
