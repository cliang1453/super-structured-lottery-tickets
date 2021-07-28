from transformers import *
from module.san_model import SanModel
from module.modeling_bert_masked import GradBasedMaskedBertModel

MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "hmbert" : (BertConfig, GradBasedMaskedBertModel, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetModel, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertModel, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer),
    "san": (BertConfig, SanModel, BertTokenizer),
}
