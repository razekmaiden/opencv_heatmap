import numpy as np
import cv2
import random


def create_canvas(w_img, h_img, bboxs,  channels=3):
    canvas = np.zeros((h_img, w_img, channels), dtype=np.uint8)
    for bbox in bboxs:
        print("[DEBUG] BBOX {} ".format(bbox))
        x, y, w, h = bbox
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (255, 255, 255), -1)
    return canvas.astype(np.uint16)


def bbox_generator(w_img, h_img, cant, max_w_box=60, max_h_box=150, min_w_box=40, min_h_box=60):
    bboxs = []
    for i in range(cant):
        x = random.randrange(0, w_img-max_w_box)
        y = random.randrange(0, h_img-max_h_box)
        w = random.randrange(min_w_box, max_w_box)
        h = random.randrange(min_h_box, max_h_box)
        bboxs.append([x, y, w, h])
    return bboxs


def min_max_normalization(input_array):
    if input_array.max() > 0:
        temp = input_array / input_array.max()
        temp = np.floor(temp * 255)
        return temp.astype(np.uint8)
    else:
        return input_array.astype(np.uint8)


def heatmap_creator(background, bboxs_stack, alpha=0.55):
    n_frames = len(bboxs_stack)
    #print("[DEBUG]: N_FRAMES: {}".format(n_frames))
    factor = 1 / n_frames
    stack = None
    h, w = background.shape[:2]
    for i in range(n_frames):
        temp_canvas = create_canvas(w, h, bboxs_stack[i])
        if stack is None:
            stack = factor*temp_canvas
        else:
            stack = stack + factor * temp_canvas
    norm_stack = min_max_normalization(stack)
    map_norm_stack = cv2.applyColorMap(norm_stack, cv2.COLORMAP_JET)
    b, g, r = cv2.split(map_norm_stack)
    t, b = cv2.threshold(b, 128, 255, cv2.THRESH_BINARY)
    map_norm_stack = cv2.merge((b, g, r))
    return cv2.addWeighted(background, alpha, map_norm_stack, 1-alpha, 0)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to image")
    ap.add_argument("-n", "--number", required=True, type=int, help="Number of frames with 'generated detections'")
    args = vars(ap.parse_args())

    background = cv2.imread(args["image"])
    cant_images = args["number"]
    h, w = background.shape[:2]
    bboxs_stack = []
    for i in range(cant_images):
        bboxs = bbox_generator(w, h, 3)
        bboxs_stack.append(bboxs)
    heatmap = heatmap_creator(background, bboxs_stack)
    cv2.imshow("[INFO] HeatMap", heatmap)
    cv2.waitKey(0)



