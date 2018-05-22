import cv2
import numpy as np
import time
import multiprocessing

from object_detection.utils import visualization_utils as vis_util
from object_detector import ObjectDetection

from libs.tools import DetectionParams


def main():
    out_pipe, in_pipe = multiprocessing.Pipe(duplex=False)
    receive_pipe, send_pipe = multiprocessing.Pipe(duplex=False)

    object_detection = ObjectDetection(out_pipe, send_pipe)
    object_detection.start()

    cap = cv2.VideoCapture(0)
    time.sleep(10)

    print('Staring...')

    while True:
        ret, image_np = cap.read()

        in_pipe.send(image_np)

        (boxes, scores, classes, num_detections) = receive_pipe.recv()

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            DetectionParams.category_index,
            use_normalized_coordinates = True,
            line_thickness = 6)

        cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            object_detection.terminate()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()