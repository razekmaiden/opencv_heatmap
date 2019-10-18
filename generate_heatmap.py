import numpy as np
import cv2
import random


def create_canvas(w_img, h_img, bboxs,  channels=3):
    canvas = np.zeros((h_img, w_img, channels), dtype=np.uint8)
    for bbox in bboxs:
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
    temp = input_array / input_array.max()
    temp = np.floor(temp * 255)
    return temp.astype(np.uint8)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to image")
    ap.add_argument("-n", "--number", required=True, type=int, help="Number of frames with 'generated detections'")
    ap.add_argument("-c", "--colormap", required=True, help="Colormap: jet or hot ")
    args = vars(ap.parse_args())

    background = cv2.imread(args["image"])
    colormap = args["colormap"]
    cant_images = args["number"]
    alpha = 0.55
    factor = 1 / cant_images
    stack = None
    h, w = background.shape[:2]
    for i in range(cant_images):
        bboxs = bbox_generator(w, h, 3)
        test_canvas = create_canvas(w, h, bboxs)
        #cv2.imshow("[INFO] Test", test_canvas.astype(np.uint8))
        if stack is None:
            stack = factor*test_canvas
        else:
            stack = stack + factor * test_canvas
        #cv2.waitKey(0)
    res_show = min_max_normalization(stack)
    if colormap == "jet":
        res_show = cv2.applyColorMap(res_show, cv2.COLORMAP_JET)
        b, g, r = cv2.split(res_show)
        t, b = cv2.threshold(b, 128, 255, cv2.THRESH_BINARY)
        res_show = cv2.merge((b, g, r))
    elif colormap == "hot":
        res_show = cv2.applyColorMap(res_show, cv2.COLORMAP_HOT)

    cv2.imshow("[INFO] Average", res_show)
    heatmap = cv2.addWeighted(background, alpha, res_show, 1-alpha, 0)
    cv2.imshow("[INFO] HeatMap", heatmap)
    cv2.waitKey(0)



