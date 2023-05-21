import config
from sklearn.preprocessing import MultiLabelBinarizer
from konlpy.tag import Mecab

class Preprocessor(object):
    def __init__(self,datas):
        self.datas = datas
        self.m = Mecab()

    def pre_process(self):
        sentences, labels_str, labels, indexes = [], [], [], []
        for i in self.datas:
            sentence = i.get('sentence')
            sentence = " ".join(sentence)
            sentences.append(sentence)
            labels.append(i.get('label'))
            indexes.append(i.get('index'))
            labels_str.append(str(i.get('label')))

        encoed_labels = self.label_encoding(labels_str)

        return sentences, encoed_labels, labels, labels_str
