from flask import Flask, jsonify, request, make_response
from flask_restful import Resource
from sources.model import KoElectra_api_3,BERT_API
from transformers import ElectraModel, ElectraTokenizer
from sources.create_dataset import Create_API_Data
from pytorch_transformers import AdamW
from torch.utils.data import DataLoader
from flask.views import MethodView
import torch
import json
import requests
import logging
from sources.utils import init_logger
import config
import argparse
import six, os, copy
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

init_logger()
logger = logging.getLogger(__name__)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
class API(Resource): 
    def get(self):
        global args
        global model
        global tokenizer
        global device

        result_dict = {}
        inputs = request.args.get('sentence', None)

        if inputs is None or len(inputs) < 1:
            result_dict = dict(message="fail")
            return json.dumps(result_dict, ensure_ascii=False)

        encodings = tokenizer.encode_plus(inputs,
                                        None,
                                        add_special_tokens=True, 
                                        max_length=128,
                                        padding='max_length',
                                        return_token_type_ids=False,
                                        return_attention_mask=True, 
                                        truncation=True,  
                                        return_tensors='pt')
                        

        with torch.no_grad():
            ids = encodings['input_ids'].to(device, dtype=torch.long)
            mask = encodings['attention_mask'].to(device, dtype=torch.long)
            pred,softmax = model(ids)
        
        softmax = softmax.tolist()[0]

        emotion_3 = {}
        for index, result in enumerate(softmax):
            emotion_3[index] = result
        emotion_3 = dict([(config.e_3.get(key), value) for key, value in emotion_3.items()])

        sentiment_score = sum([v*p for p,(k,v) in zip([1,-1,0],emotion_3.items())])

        max_values_3 = max(emotion_3.values())
        max_key_3 = {v:k for k,v in emotion_3.items()}.get(max_values_3)

        result_dict['sentence'] = inputs
        result_dict['sentiment_score'] = sentiment_score
        result_dict['result(emotion_3)'] = max_key_3
        result_dict['emotion_3'] = emotion_3

        return make_response(json.dumps(result_dict,  ensure_ascii=False))

    def post(self):
        global args
        global model
        global tokenizer
        global device

        result_dict = {}
        inputs = request.get_json().get('sentences', None)
        result_dict['sentences'] = copy.deepcopy(inputs)

        if inputs is None or len(inputs) < 1:
            result_dict = dict(message="fail")
            return json.dumps(result_dict, ensure_ascii=False)

        encodings = Create_API_Data(datas=inputs,
                                    max_len=config.MODEL_CONFIG['max_len'],
                                    tokenizer=tokenizer)
        encodings = DataLoader(encodings, batch_size=config.MODEL_CONFIG['batch_size'], shuffle=False, num_workers=0)
                        

        with torch.no_grad():
            emotion_3_total = []
            max_key_3_total =[]
            sentiment_score_total=[]
            for batch_idx, data in enumerate(encodings):
                ids = data['input_ids'].to(device, dtype=torch.long)
                mask = data['attention_mask'].to(device, dtype=torch.long)
                pred,softmax = model(ids)
                softmax = softmax.tolist()

                for i in softmax:
                    emotion_3 = {}
                    for index, result in enumerate(i):
                        emotion_3[index] = result
                    emotion_3 = dict([(config.e_3.get(key), value) for key, value in emotion_3.items()])
                    emotion_3 = dict(sorted(emotion_3.items(), key=lambda x : x[1], reverse=True))
                    emotion_3_total.append(emotion_3)

                    sentiment_score = sum([v*p for p,(k,v) in zip([1,-1,0],emotion_3.items())])
                    sentiment_score_total.append(sentiment_score)

                    max_values_3 = max(emotion_3.values())
                    max_key_3 = {v:k for k,v in emotion_3.items()}.get(max_values_3)

                    max_key_3_total.append(max_key_3)

        result_dict['result(emotion_3)'] = max_key_3_total
        result_dict['sentiment_score'] = sentiment_score_total
        result_dict['emotion_3'] = emotion_3_total

        return make_response(json.dumps(result_dict, ensure_ascii=False))

def load_model():
    global args
    global model
    global tokenizer
    global device

    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.target_gpu))
        logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
        logger.info('We will use the GPU:{}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        logger.info('No GPU available, using the CPU instead.')


    model = BERT_API(args)

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=config.MODEL_CONFIG['learning_rate'],
                        eps=config.MODEL_CONFIG['adam_epsilon'])

    model.load_state_dict(torch.load(args.load_ck,map_location=device))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
