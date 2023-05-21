import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraTokenizer,  ElectraConfig
from transformers import BertModel, AutoModelForSequenceClassification
import config
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class BERT(torch.nn.Module):
    def __init__(self,args):
        super(BERT, self).__init__()
        self.num_labels = 3
        if args.model_path:
            self.model = BertModel.from_pretrained(args.model_path)
        else:
            self.model = BertModel.from_pretrained("bert-base-german-cased")
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    def forward(self, input_ids, labels):

        discriminator_hidden_states = self.model(input_ids)
        
        pooled_output = discriminator_hidden_states[1] #(64, 768)
        pooled_output = self.dropout(pooled_output) #(64, 768)
        logits = self.linear(pooled_output) #(64, 34)
        outputs = (logits,) + discriminator_hidden_states[2:]

        softmax = torch.nn.functional.softmax(outputs[0], dim=1) #(64, 34)

        pred = softmax.argmax(dim=1) #(64)
        correct = pred.eq(labels)  #(64)

        loss = self.loss_fct(logits, labels)
        
        outputs = (loss,) + outputs # loss, logits, hidden_states

        return outputs, pred, labels, correct, softmax

class BERT_API(torch.nn.Module):
    def __init__(self,args):
        super(BERT_API, self).__init__()
        self.num_labels = 3
        self.model = BertModel.from_pretrained(args.load_ck)
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    def forward(self, input_ids):

        discriminator_hidden_states = self.model(input_ids)
        
        pooled_output = discriminator_hidden_states[1] #(64, 768)
        pooled_output = self.dropout(pooled_output) #(64, 768)
        logits = self.linear(pooled_output) #(64, 34)
        outputs = (logits,) + discriminator_hidden_states[2:]

        softmax = torch.nn.functional.softmax(outputs[0], dim=1) #(64, 34)

        pred = softmax.argmax(dim=1) #(64)

        return pred, softmax
