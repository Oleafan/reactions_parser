# -*- coding: utf-8 -*-
import numpy as np
import re
import torch
import transformers as ppb
import pickle
from copy import deepcopy

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

with open('token_classifier/bert_classifier.pkl', 'rb') as f:
    lr_model = pickle.load(f)

def get_classes(token_list):    
    token_list = deepcopy(token_list)
    texts_tokenized = []
    max_len = 512

    for text in token_list:
        text = text[:512]
        texts_tokenized.append(tokenizer.encode(text, add_special_tokens=True))
        
    padded_init_texts = np.array([text + [0]*(max_len-len(text)) for text in texts_tokenized])
    padded_init_texts = padded_init_texts.reshape(len(token_list),1, 512)
    input_ids_texts = torch.tensor(padded_init_texts)

    text_probs = []
    text_lables = []

    for tens in input_ids_texts:
        with torch.no_grad():
            last_hidden_states_init_texts = model(tens)
            text_probs.append(lr_model.predict_proba(last_hidden_states_init_texts[0][:,0,:].numpy()))
        
    text_lables = []
    for probs in text_probs:
        if probs[0][1] >= 0.3:
            text_lables.append(1)
        else:
            text_lables.append(0)

                
    return text_lables





