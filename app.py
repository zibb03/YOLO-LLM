import yolov8
import gradio as gr

description = "사진을 첨부하면 사진 속 객체를 감지하여 블로그 글을 작성해 줍니다. "
title = "Blog Writter using image"

def predict(img):
    model = yolov8.YOLO_V8()

    result = model.predict(img)

    # result 배열에서 객체 수 셀 수 있도록

    return result

demo = gr.Interface(
    fn=predict,
    inputs='image',
    outputs='text',
    description=description,
    title=title,
)

demo.launch(share=False)