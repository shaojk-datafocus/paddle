# encoding = utf-8
import paddle
import paddle.nn.functional as F
from model import SentimentClassifier
from easydict import EasyDict
from dataset import DataGenerator, load_imdb, data_preprocess, convert_corpus_to_id
from tqdm import tqdm
import json

def train(model, config, logger=None):
    # 检测是否可以使用GPU，如果可以优先使用GPU
    use_gpu = True if paddle.get_device().startswith("gpu") else False
    if use_gpu:
        paddle.set_device('gpu:0')

    # 指定优化策略，更新模型参数
    optimizer = paddle.optimizer.Adam(learning_rate=config.LEARNING_RATE, beta1=0.9, beta2=0.999, parameters=model.parameters()) 

    # 开启模型训练模式
    model.train()
    
    # 建立训练数据生成器，每次迭代生成一个batch，每个batch包含训练文本和文本对应的情感标签
    with open(config.WORD_DICT_PATH, 'r',encoding='utf-8') as f:
        word2id_dict = json.loads(f.read())
    train_corpus = load_imdb(True)
    train_corpus = data_preprocess(train_corpus)
    train_corpus = convert_corpus_to_id(train_corpus, word2id_dict)
    
    i = 0
    for epoch in range(config.EPOCH_NUM):
        bar = tqdm(
            DataGenerator(word2id_dict, train_corpus, config.BATCH_SIZE, config.MAX_SEQ_LEN),
            total=len(train_corpus)//config.BATCH_SIZE)
        for sentences, labels in bar:
            sentences = paddle.to_tensor(sentences)
            labels = paddle.to_tensor(labels)
            
            logits = model(sentences)

            loss = F.cross_entropy(input=logits, label=labels, soft_label=False)
            loss = paddle.mean(loss)

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            
            bar.postfix = "{}/{}, Loss: {:.6f}".format(epoch+1, config.EPOCH_NUM, loss.numpy()[0])
            if logger and i%100==0:
                logger.add_scalar(tag = 'loss', step = i, value = loss.numpy()[0])
            i += 1

if __name__ == "__main__":
    config = EasyDict()
    config.EPOCH_NUM = 5
    config.BATCH_SIZE = 128
    config.LEARNING_RATE = 0.01
    config.DROPOUT_RATE = 0.2
    config.NUM_LAYERS = 1
    config.HIDDEN_SIZE = 256
    config.EMBEDDING_SIZE = 256
    config.MAX_SEQ_LEN = 128
    config.VOCA_SIZE = 252173
    config.WORD_DICT_PATH = "word2id_dict.json"

    model = SentimentClassifier(config.HIDDEN_SIZE, config.VOCA_SIZE, config.EMBEDDING_SIZE, 
        num_steps=config.MAX_SEQ_LEN, num_layers=config.NUM_LAYERS, dropout_rate=config.DROPOUT_RATE)

    train(model, config)

    # 保存模型，包含两部分：模型参数和优化器参数
    model_name = "sentiment_classifier"
    # 保存训练好的模型参数
    paddle.save(model.state_dict(), "{}.pdparams".format(model_name))
    # 保存优化器参数，方便后续模型继续训练
    paddle.save(model.state_dict(), "{}.pdopt".format(model_name))