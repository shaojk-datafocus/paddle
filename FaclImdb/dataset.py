# encoding = utf-8
import re
import random
import tarfile
import numpy as np

def load_imdb(is_training):
    data_set = []
    # aclImdb_v1.tar.gz解压后是一个目录
    # 我们可以使用python的rarfile库进行解压
    # 训练数据和测试数据已经经过切分，其中训练数据的地址为：
    # ./aclImdb/train/pos/ 和 ./aclImdb/train/neg/，分别存储着正向情感的数据和负向情感的数据
    # 我们把数据依次读取出来，并放到data_set里
    # data_set中每个元素都是一个二元组，（句子，label），其中label=0表示负向情感，label=1表示正向情感
    
    for label in ["pos", "neg"]:
        with tarfile.open("./aclImdb_v1.tar.gz") as tarf:
            path_pattern = "aclImdb/train/" + label + "/.*\.txt$" if is_training \
                else "aclImdb/test/" + label + "/.*\.txt$"
            path_pattern = re.compile(path_pattern)
            tf = tarf.next()
            while tf != None:
                if bool(path_pattern.match(tf.name)):
                    sentence = tarf.extractfile(tf).read().decode()
                    sentence_label = 0 if label == 'neg' else 1
                    data_set.append((sentence, sentence_label)) 
                tf = tarf.next()

    return data_set

def data_preprocess(corpus):
    data_set = []
    for sentence, sentence_label in corpus:
        # 这里有一个小trick是把所有的句子转换为小写，从而减小词表的大小
        # 一般来说这样的做法有助于效果提升
        sentence = sentence.strip().lower()
        sentence = sentence.split(" ")
        
        data_set.append((sentence, sentence_label))

    return data_set

# 构造词典，统计每个词的频率，并根据频率将每个词转换为一个整数id
def build_dict(corpus):
    word_freq_dict = dict()
    for sentence, _ in corpus:
        for word in sentence:
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
            word_freq_dict[word] += 1

    # 按照词频降序排列
    word_freq_dict = sorted(word_freq_dict.items(), key = lambda x:x[1], reverse = True)
    
    word2id_dict = dict()
    word2id_freq = dict()

    # 一般来说，我们把oov和pad放在词典前面，给他们一个比较小的id，这样比较方便记忆，并且易于后续扩展词表
    word2id_dict['[oov]'] = 0
    word2id_freq[0] = 1e10

    word2id_dict['[pad]'] = 1
    word2id_freq[1] = 1e10

    for word, freq in word_freq_dict:
        word2id_dict[word] = len(word2id_dict)
        word2id_freq[word2id_dict[word]] = freq

    return word2id_freq, word2id_dict

# 把语料转换为id序列
def convert_corpus_to_id(corpus, word2id_dict):
    data_set = []
    for sentence, sentence_label in corpus:
        # 将句子中的词逐个替换成id，如果句子中的词不在词表内，则替换成oov
        # 这里需要注意，一般来说我们可能需要查看一下test-set中，句子oov的比例，
        # 如果存在过多oov的情况，那就说明我们的训练数据不足或者切分存在巨大偏差，需要调整
        sentence = [word2id_dict[word] if word in word2id_dict \
                    else word2id_dict['[oov]'] for word in sentence]    
        data_set.append((sentence, sentence_label))
    return data_set

# 编写一个迭代器，每次调用这个迭代器都会返回一个新的batch，用于训练或者预测
def DataGenerator(word2id_dict, corpus, batch_size, max_seq_len, shuffle = True, drop_last = True):

    # 模型将会接受的两个输入：
    # 1. 一个形状为[batch_size, max_seq_len]的张量，sentence_batch，代表了一个mini-batch的句子。
    # 2. 一个形状为[batch_size, 1]的张量，sentence_label_batch，每个元素都是非0即1，代表了每个句子的情感类别（正向或者负向）
    sentence_batch = []
    sentence_label_batch = []

    #每个epoch前都shuffle一下数据，有助于提高模型训练的效果
    #但是对于预测任务，不要做数据shuffle
    if shuffle:
        random.shuffle(corpus)

    for sentence, sentence_label in corpus: # 遍历每个句子
        sentence_sample = sentence[:min(max_seq_len, len(sentence))] # 过滤到允许的最大长度
        if len(sentence_sample) < max_seq_len: # 长度不够则填充【pad】
            for _ in range(max_seq_len - len(sentence_sample)):
                sentence_sample.append(word2id_dict['[pad]'])
        
        sentence_sample = [[word_id] for word_id in sentence_sample] # 数据增加一个维度

        sentence_batch.append(sentence_sample) # 添加到批次中
        sentence_label_batch.append([sentence_label])

        if len(sentence_batch) == batch_size:
            yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")
            sentence_batch = []
            sentence_label_batch = []

    if not drop_last and len(sentence_batch) > 0:
        yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")


if __name__ == '__main__':
    """
    数据预处理
    1. 对语料切片，使用空格分割。
    2. 对使用到的词，构造一个ID词典 [oov](out-of-vocabulary),用于处理那些不在词典中的词
    3. 将原始语料转化为ID序列
    4. 将句子截断、填充，生成固定长度的句子。
    """

    train_corpus = load_imdb(True)
    test_corpus = load_imdb(False)

    # for i in range(1):
    #     print("sentence %d, %s" % (i, train_corpus[i][0]))    
    #     print("sentence %d, label %d" % (i, train_corpus[i][1]))

    train_corpus = data_preprocess(train_corpus)
    test_corpus = data_preprocess(test_corpus)
    # print(train_corpus[:5])
    # print(test_corpus[:5])
    
    word2id_freq, word2id_dict = build_dict(train_corpus)
    import json
    with open("word2id_freq.json",'w',encoding='utf-8') as f:
        f.write(json.dumps(word2id_freq))
    with open("word2id_dict.json",'w',encoding='utf-8') as f:
        f.write(json.dumps(word2id_dict))
    
    vocab_size = len(word2id_freq)
    print("there are totoally %d different words in the corpus" % vocab_size)
    for _, (word, word_id) in zip(range(10), word2id_dict.items()):
        print("word %s, its id %d, its word freq %d" % (word, word_id, word2id_freq[word_id]))


    train_corpus = convert_corpus_to_id(train_corpus, word2id_dict)
    test_corpus = convert_corpus_to_id(test_corpus, word2id_dict)
    print("%d tokens in the corpus" % len(train_corpus))
    print(train_corpus[:5])
    print(test_corpus[:5])



    for batch_id, batch in enumerate(build_batch(word2id_dict, train_corpus, batch_size=3, epoch_num=3, max_seq_len=30)):
        print(batch)

