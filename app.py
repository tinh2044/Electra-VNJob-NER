from transformers import pipeline

import gradio as gr
from transformers import AutoTokenizer, ElectraForTokenClassification

repo_id = "tinh2312/Electra-VNJob-NER"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = ElectraForTokenClassification.from_pretrained(repo_id, num_labels=17, ignore_mismatched_sizes=True)
token_classifier = pipeline(model=model, tokenizer=tokenizer, task="ner", aggregation_strategy="simple")
examples = [
    "Nhân viên thi công lắp đặt PCCC full-time, nhân viên chuyên viên Long An, trên 1 năm xây dựng kết cấu công "
    "trình, thi công xây dựng, kinh doanh điện tử, điện lạnh, bán hàng điện, 8 - 12 triệu VNĐ",
    "Nhân viên phụ bếp, không yêu cầu kinh nghiệm, full-time, part-time, nhân viên chuyên viên Hà Nội, không yêu cầu, "
    "chế biến thực phẩm, phục vụ bàn, thực phẩm ăn uống, phụ bếp, lao động phổ thông, nhà hàng, khách sạn, "
    "6 - 10 triệu VNĐ",
    "NV lễ tân, phục vụ trà nước showroom ô tô full-time, nhân viên chuyên viên Hồ Chí Minh, dưới 1 năm hành chính "
    "văn phòng, phục vụ bàn, lễ tân, receptionist, bán lẻ, hành chính, bán sỉ, văn phòng, nhà hàng, khách sạn, "
    "cửa hàng bán lẻ, từ 8 triệu VNĐ"
]



def ner(text):
    output = token_classifier(text.lower())
    return {"text": text, "entities": output}


demo = gr.Interface(ner,
                    gr.Textbox(placeholder="Enter sentence here...", lines=10),
                    gr.HighlightedText(),
                    examples=examples)

if __name__ == "__main__":
    demo.launch()
