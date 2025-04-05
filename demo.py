import numpy as np
from PIL import Image, ImageDraw
from jolopy import YOLODetector
import matplotlib.pyplot as plt


if __name__ == "__main__":
    yoloc = YOLODetector("yolov8n.onnx", 640, 640, 80)

    img = Image.open("image.jpeg")
    img_np = np.array(img)
    detections = yoloc.detect(img_np)
    print(detections)

    draw = ImageDraw.Draw(img)

    for det in detections:
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)
        label = f"Class {det['class_id']}: {det['confidence']:.2f}"
        draw.text((x, y - 10), label, fill="red")

    plt.imshow(img)
    plt.axis("off")
    plt.show()
