from ultralytics import YOLO

class YOLO_V8:
    img_path = 'Lenna.png'
    class_names = []

    model = YOLO('yolov8n.pt')
    results = model(img_path)

    bndboxes = results[0].boxes.data
    class_id = results[0].boxes.cls

    names = results[0].names
    # print()
    # print(names)

    # bound box
    for i, bndbox in enumerate(bndboxes):
        class_id = int(bndbox[5])
        class_names.append(names[class_id])

    # print(class_names)
