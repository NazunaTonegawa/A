import torch
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.title('要約')
st.markdown('BARTモデルを使っています')

_num_beams = 4 
_no_repeat_ngram_size = 3
_length_penalty = 2
_min_length = 50
_max_length = 5000
_early_stopping = True

col1, col2, col3 = st.columns(3)
_length_penalty = col1.number_input("要約する文の長さの制限", value=_length_penalty)
_min_length = col2.number_input("最小文字数", value=_min_length)
_max_length = col3.number_input("最大文字数", value=_max_length)

text = st.text_area('要約する文を入力してください')

def run_model(input_text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    input_text = str(input_text).replace('\n', '')
    input_text = ' '.join(input_text.split())
    input_tokenized = t5_tokenizer.encode(input_text, return_tensors="pt").to(device)
    summary_task = torch.tensor([[21603, 10]]).to(device)
    input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(device)
    summary_ids = t5_model.generate(input_tokenized,
    num_beams=_num_beams,
    no_repeat_ngram_size=_no_repeat_ngram_size,
    length_penalty=_length_penalty,
    min_length=_min_length,
    max_length=_max_length,
    early_stopping=_early_stopping)
    output = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    st.write('要約です')
    st.success(output[0])


if st.button('Submit'):
    run_model(text)