import cv2
from ultralytics import YOLO

class YOLO_V8:
    def predict(self, image):
        class_names = []

        model = YOLO('yolov8n.pt')
        results = model(image)

        bndboxes = results[0].boxes.data
        # class_id = results[0].boxes.cls

        names = results[0].names
        # print()
        # print(names)

        # bound box
        for i, bndbox in enumerate(bndboxes):
            class_id = int(bndbox[5])
            class_names.append(names[class_id])

        return class_names

if __name__ == '__main__':
    model = YOLO_V8()

    path = 'lenna.png'
    img = cv2.imread(path)
    result = model.predict(img)

    print(result)
