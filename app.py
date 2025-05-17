import yolov8
import gradio as gr
from transformers import pipeline

description = "사진을 첨부하면 사진 속 객체를 감지하여 블로그 글을 작성해 줍니다. "
title = "Blog Writter using image"

def predict(img):
    model = yolov8.YOLO_V8()

    label = model.predict(img)

    # result 배열에서 객체 수 셀 수 있도록

    # 텍스트 생성을 진행하는 pipeline을 생성하는 부분
    nlp = pipeline("text-generation")
    # 텍스트 생성을 진행하는 부분
    prompt = "Write a script for a 2-minute video about "

    result = nlp(prompt + ', '.join(label))

    # result의 첫번쨰 값을 가져오고, generated_text의 value를 가져오는 부분
    # print(result[0]['generated_text'])

    return result[0]['generated_text']


    '''
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained("klue/roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

    question_answer = pipeline("question-answering", model="klue/roberta-base") #모델명은 허깅페이스에서 검색하여 제목 그대로 사용
    question_answer(
        question="Write a script for a 2-minute video about a topic",
        context=' '.join(label)
    )

    print(question_answer)

    return 0

    #출력
    {'score': 0.00026570854242891073,
     'start': 40,
     'end': 53,
     'answer': '독일의 본에서 태어났으며'}
    '''

demo = gr.Interface(
    fn=predict,
    inputs='image',
    outputs='text',
    description=description,
    title=title,
)

demo.launch(share=False)