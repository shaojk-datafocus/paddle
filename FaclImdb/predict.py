import json
import paddle
import paddle.nn.functional as F
from model import SentimentClassifier
from dataset import DataGenerator, data_preprocess, convert_corpus_to_id
from easydict import EasyDict

config = EasyDict()
config.NUM_LAYERS = 1
config.HIDDEN_SIZE = 256
config.EMBEDDING_SIZE = 256
config.MAX_SEQ_LEN = 128
config.VOCA_SIZE = 252173
config.DROPOUT_RATE = 0.2

with open("word2id_dict.json", 'r',encoding='utf-8') as f:
    word2id_dict = json.loads(f.read())
test_corpus=[["Previous reviewer Claudio Carvalho gave a much better recap of the film's plot details than I could. What I recall mostly is that it was just so beautiful, in every sense - emotionally, visually, editorially - just gorgeous.<br /><br />If you like movies that are wonderful to look at, and also have emotional content to which that beauty is relevant, I think you will be glad to have seen this extraordinary and unusual work of art.<br /><br />On a scale of 1 to 10, I'd give it about an 8.75. The only reason I shy away from 9 is that it is a mood piece. If you are in the mood for a really artistic, very romantic film, then it's a 10. I definitely think it's a must-see, but none of us can be in that mood all the time, so, overall, 8.75.",1]]
print(test_corpus[0])
test_corpus = data_preprocess(test_corpus)
test_corpus = convert_corpus_to_id(test_corpus, word2id_dict)
g = DataGenerator(word2id_dict,test_corpus,1,config.MAX_SEQ_LEN)

model = SentimentClassifier(config.HIDDEN_SIZE, config.VOCA_SIZE, config.EMBEDDING_SIZE, 
        num_steps=config.MAX_SEQ_LEN, num_layers=config.NUM_LAYERS, dropout_rate=config.DROPOUT_RATE)
    
model.load_dict(paddle.load("sentiment_classifier.pdparams"))
model.eval()

sentences,_ = next(g)
sentences = paddle.to_tensor(sentences)
outputs = model(sentences)
probs = F.softmax(outputs)
print("预测结果是：",probs[0][1] > probs[0][0])
print("实际结果是:", test_corpus[0][1])