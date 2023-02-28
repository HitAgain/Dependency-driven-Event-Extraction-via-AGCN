# -*- coding: utf-8 -*-
# @Date:   2021/3/11 20:13
# @Modify: 2021/4/08 17:00

import os
os.environ['TF_KERAS'] = '1'
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import pylcs
import numpy as np
from collections import defaultdict
import json
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

from gensim.models import KeyedVectors

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Lambda
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

from bert4keras.models import build_transformer_model
from bert4keras.snippets import open, sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer, load_vocab

from crf.crf import CRF
from dep.depgcn import DepGcn
from tool.ltp_tool import ltp_parse, critic, search

# albert-tokenizer init
with open('./label/attribute.json', 'r', encoding="utf-8") as f:
    label2id = json.load(f)
    id2label = {value:key for key, value in label2id.items()}
    num_labels = len(id2label) * 2 + 1
logging.info("[Config]Label2id && id2label dic init success")
token_dict = load_vocab(dict_path="./albert/vocab.txt")
tokenizer = Tokenizer(token_dict, do_lower_case=False)
logging.info("[Config]Albert tokenizer init success")

# albert-tokenizer init
count = 0
dep_map = {}
with open('./label/deptype.txt', 'r', encoding="utf-8") as fin:
    for line in fin:
        dep_type = line.strip("\n")
        dep_map[dep_type] = count
        count += 1
logging.info("[Config]Dep type map init success")

# pretrain word2vec init
model = KeyedVectors.load_word2vec_format("/home/EventAlgorithm/hot_event_type_model/pretrained/model/train.w2v.100000.model", binary=False)
word_vocabulary = dict([(k, v.index + 3) for k, v in model.vocab.items()])
word_vocabulary["</p>"] = 0  # 补零
word_vocabulary["</u>"] = 1  # oov
word_vocabulary[" "] = 2  # 空格
weights = np.vstack((model.syn0[-1], model.syn0[-2], model.syn0[-3], model.syn0))
logging.info("[Config]Pretrain Word2Vec dict && weights prepared success")


def gen_word_to_char_matrix(word_ls, char_ls):
    matrix = np.zeros((len(char_ls), len(word_ls)))
    start_ind = 0
    for ind, word in enumerate(word_ls):
        for j in range(start_ind, start_ind + len(word)):
            matrix[j][ind] = 1
        start_ind += len(word)
    return matrix


def gen_dep_matrix(relations, heads):
    dep_link_matrix = np.eye(len(relations), dtype=float)
    for tail, head in enumerate(heads):
        if head != 0:
            dep_link_matrix[tail][head - 1] = 1.0
            dep_link_matrix[head - 1][tail] = 1.0
    dep_type_matrix = np.zeros((len(relations), len(relations)), dtype = int)
    for i in range(len(relations)):
        if heads[i] == 0 and relations[i] == 'HED':
            dep_type_matrix[i][i] = dep_map["HED"]
        else:
            dep_type_matrix[i][heads[i] - 1] = dep_map[relations[i]]
            dep_type_matrix[heads[i] - 1][i] = dep_map[relations[i]]
    return dep_link_matrix, dep_type_matrix

def matrix_padding(max_cols, max_rows, ori_mat_ls):
    res = []
    for mat in ori_mat_ls:
        # col padding
        mat = np.concatenate((mat, np.zeros((len(mat), max_cols - len(mat[0])))), axis = 1)
        # row padding
        mat = np.concatenate((mat, np.zeros((max_rows - len(mat), len(mat[0])))), axis = 0)
        res.append(mat)
    return np.array(res)

#############################数据读入模块#############################
def dataset(filename):
    sample = []
    with open(filename, 'r', encoding="utf-8") as f_in:
        for l in f_in:
            data = json.loads(l)
            text = data['text'].strip("\n").replace("\n", "").replace(" ", "。")
            label_dic = data['labels'][0]
            subj = label_dic["object"][0] if label_dic["object"] else ""
            trigger = label_dic["trigger"][0] if label_dic["trigger"] else ""
            obj = label_dic["subject"][0] if label_dic["subject"] else ""

            role_feature = defaultdict(str)
            dep_feature = defaultdict(str)

            # 语义角色特征
            role_feature["subject"] = subj
            if not obj:
                role_feature["trigger_half"] = trigger
            else:
                role_feature["trigger_full"] = trigger
                role_feature["object"] = obj

            # 句法依存特征
            words, postags, arcs = ltp_parse(text)
            dep_feature["word_ls"] = "|".join(words)
            dep_feature["dep_type"] = "|".join([arc.relation for arc in arcs])
            dep_feature["dep_link"] = "|".join([str(arc.head) for arc in arcs])
            sample.append((text, role_feature, dep_feature))
    return sample

class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_word_ids, batch_dep_type_mat, batch_dep_link_mat, batch_word2char_mat, batch_labels\
             = [], [], [], [], [], [], []
        for is_end, (text, role_feature, dep_feature) in self.sample(random):
            # char input
            char_tokens = list(text)
            token_ids, segment_ids = tokenizer.encode(char_tokens)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)

            char_ls = [tokenizer.decode([token_id]) for token_id in token_ids]
            word_ls = dep_feature["word_ls"].split("|")
            try:
                word2char_matrix = gen_word_to_char_matrix(word_ls, char_ls)
            except IndexError:
                print(char_ls)
                print(word_ls)
                raise
            word_ids = [word_vocabulary.get(word, 1) for word in word_ls]
            batch_word_ids.append(word_ids)
            batch_word2char_mat.append(word2char_matrix)

            # dep input
            relations = dep_feature["dep_type"].split("|")
            heads = [int(head) for head in dep_feature["dep_link"].split("|")]
            dep_link_mat, dep_type_mat = gen_dep_matrix(relations, heads)
            batch_dep_type_mat.append(dep_type_mat)
            batch_dep_link_mat.append(dep_link_mat)

            # label
            labels = [0] * len(token_ids)
            for schema, word in role_feature.items():
                a_token_ids = tokenizer.encode(word)[0][1:-1]
                start_index = search(a_token_ids, token_ids)
                if start_index != -1:
                    labels[start_index] = label2id[schema] * 2 + 1
                    for i in range(1, len(a_token_ids)):
                        labels[start_index + i] = label2id[schema] * 2 + 2
            batch_labels.append(labels)

            if len(batch_token_ids) == self.batch_size or is_end:
                ## max rows
                max_rows = max([len(i) for i in batch_token_ids])
                ## max cols
                max_cols = max([len(i) for i in batch_word_ids])

                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                batch_word_ids = sequence_padding(batch_word_ids)
                batch_word2char_mat = matrix_padding(max_cols, max_rows, batch_word2char_mat)
                batch_dep_link_mat = matrix_padding(max_cols, max_cols, batch_dep_link_mat)
                batch_dep_type_mat = matrix_padding(max_cols, max_cols, batch_dep_type_mat)

                # crf层要求label dims=3 最后一个维度为1
                batch_label_reshape = batch_labels.reshape(
                    (batch_labels.shape[0], batch_labels.shape[1], 1))
                yield [batch_token_ids, batch_segment_ids, batch_word_ids, batch_dep_type_mat, batch_dep_link_mat, batch_word2char_mat], batch_label_reshape
                batch_token_ids, batch_segment_ids, batch_word_ids, batch_dep_type_mat, batch_dep_link_mat, batch_word2char_mat, batch_labels\
                    = [], [], [], [], [], [], []

#############################训练回调模块#############################
class Debug(Callback):
    def __init__(self, args):
        self.dev_path = args.dev_file
        self.test_path = args.test_file
        self.dataset_dev = dataset(self.dev_path)
        self.test_data = list()
        # load test data
        with open(self.test_path, 'r', encoding="utf-8") as f:
            for l in f:
                data = json.loads(l.strip("\n"))
                text = data['text'].strip("\n").replace("\n", "").replace(" ", "。")
                self.test_data.append(text)
        self.best_F1 = 0.0

    def extract(self, text):
        token_ids, segment_ids = tokenizer.encode(list(text))
        token_np = np.array([token_ids])
        segment_np = np.array([segment_ids])
        char_ls = [tokenizer.decode([token_id]) for token_id in token_ids[1:-1]]
        word_ls, postags, arcs = ltp_parse(text)
        word_id_np = np.array([word_vocabulary.get(word, 1) for word in word_ls])
        relations = [arc.relation for arc in arcs]
        heads = [arc.head for arc in arcs]
        word2char_matrix = gen_word_to_char_matrix(word_ls, char_ls)
        dep_link_mat, dep_type_mat = gen_dep_matrix(relations, heads)
        scores = self.model.predict([token_np, segment_np, word_id_np, dep_type_mat, dep_link_mat, word2char_matrix])[0]
        predict_ids = np.argmax(scores, axis=-1).tolist()
        arguments, starting = [], False
        for i, label in enumerate(predict_ids):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    arguments.append([id2label[(label - 1) // 2], [i]])
                elif starting:
                    arguments[-1][1].append(i)
                else:
                    starting = False
            else:
                starting = False
        res = defaultdict(list)
        # 解码
        for role, arg_ids in arguments:
            res[role].append(tokenizer.decode([token_ids[ind] for ind in arg_ids]))
        return res

    def evaluate(self):
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for text, arguments, dep_feature in self.dev_data:
            pred_arguments = self.extract(text)
            Y += len(pred_arguments)
            Z += len(arguments)
            for k, v in pred_arguments.items():
                if k in arguments:
                    X += critic(''.join(v), arguments[k])
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        with open("./test_result.txt", "w", encoding="utf-8") as fout:
            for text in self.test_data:
                fout.write("{}\t".format(text))
                extract_schema = self.extract(text)
                for label, schema in extract_schema.items():
                    fout.write("{}:{}\t".format(label, schema))
                fout.write("\n")
        logging.info("test file result write in {}".format("test_result.txt"))

    def on_epoch_end(self, epoch, logs={}):
        f1, precision, recall = self.evaluate()
        logging.info("Epoch:{}|||f1:{}--P:{}--R:{}".format(epoch, f1, precision, recall))
        if f1 > self.best_F1:
            self.best_F1 = f1
            self.model.save_weights('./best_model_weights.h5')

class DepdencyDrivenEE(object):

    def __init__(self, args):
        self.mode = args.mode
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.albert_config = './albert/albert_config_small_google.json'
        self.albert_checkpoint = './albert/albert_model.ckpt'
        self.model = self.build_model()

    def build_model(self):
        # 字符特征输入albert层
        char_token = Input(shape=[None, ], dtype="int32", name="char")
        segment_id = Input(shape=[None, ], dtype="int32", name="segment")
        # 词法和句法特征
        word_token = Input(shape=[None, ], dtype="int32", name="word")
        dep_type_matrix = Input(shape=[None, None, ], dtype="int32", name="type_matrix")
        dep_link_matrix = Input(shape=[None, None, ], dtype="float32", name="link_matrix")
        # 字词联合信息
        word2char_matrix = Input(shape=[None, None, ], dtype="float32", name="word2char_matrix")
        albert_layer = build_transformer_model(
            self.albert_config,
            self.albert_checkpoint,
            model = "albert"
        )
        for layer in albert_layer.layers:
            layer.trainable = False
        words_embedder = Embedding(np.shape(weights)[0],
                                   np.shape(weights)[1],
                                   trainable=True,
                                   mask_zero=True,
                                   name="words_embedding_layer")
        dep_embedder = Embedding(30,
                                 np.shape(weights)[1],
                                 trainable=True,
                                 mask_zero=True,
                                 name="deps_embedding_layer")
        dep_gcn = DepGcn(256, name = "dep_gcn_layer")
        crf = CRF(num_labels, sparse_target=True, name="crf_layer")


        word_emb = words_embedder(word_token)
        dep_emb = dep_embedder(dep_type_matrix)
        word_emb_dep = dep_gcn([word_emb, dep_link_matrix, dep_emb])

        # forward
        char_emb  = albert_layer([char_token, segment_id])
        char_emb_from_word = Lambda (lambda x: tf.matmul(x[0], x[1], transpose_b=False))(
            [word2char_matrix, word_emb_dep])
        concat_emb = Concatenate()([char_emb_from_word, char_emb])
        x = Dense(num_labels)(concat_emb)
        final_output = crf(x)
        model = Model(inputs=[char_token, segment_id, word_token, dep_type_matrix, dep_link_matrix, word2char_matrix], outputs = [final_output])
        if self.mode == "train":
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss=crf.loss_function)
            model.summary()
            logging.info("model build success in train mode")
            return model
        else:
            model.load_weights("./best_model_weights.h5")
            logging.info("model build success in inference mode")
            return model

    def train(self, args):
        train_data = dataset(args.train_file)
        train_generator = data_generator(train_data, self.batch_size)
        debug_callback = Debug(args)
        self.model.fit_generator(
            train_generator.forfit(),
            epochs=self.epochs,
            steps_per_epoch=len(train_generator),
            verbose=1,
            callbacks=[debug_callback]
        )


    def predict(self, text):
        token_ids, segment_ids = tokenizer.encode(list(text))
        token_np = np.array([token_ids])
        segment_np = np.array([segment_ids])
        char_ls = [tokenizer.decode([token_id]) for token_id in token_ids[1:-1]]
        word_ls, postags, arcs = ltp_parse(text)
        word_id_np = np.array([word_vocabulary.get(word, 1) for word in word_ls])
        relations = [arc.relation for arc in arcs]
        heads = [arc.head for arc in arcs]
        word2char_matrix = gen_word_to_char_matrix(word_ls, char_ls)
        dep_link_mat, dep_type_mat = gen_dep_matrix(relations, heads)
        predict_ids = list()
        scores = self.model.predict([token_np, segment_np, word_id_np, dep_type_mat, dep_link_mat, word2char_matrix])[0]
        predict_index_list = np.argmax(scores, axis=-1)
        predict_ids = predict_index_list.tolist()
        arguments, starting = [], False
        for i, label in enumerate(predict_ids):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    arguments.append([id2label[(label - 1) // 2], [i]])
                elif starting:
                    arguments[-1][1].append(i)
                else:
                    starting = False
            else:
                starting = False
        res = defaultdict(list)
        for l, arg_ids in arguments:
            res[l].append(tokenizer.decode([token_ids[ind] for ind in arg_ids]))
        return res

    def export(self):
        tf.saved_model.save(self.model, "./serving")

def fit_device(args):
    if args.device == "gpu":
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    tf.compat.v1.disable_eager_execution()
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        type=str,
        default="train",
        help='train or inference')

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='train epochs')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='train batch_size')

    parser.add_argument(
        '--lr',
        type=float,
        default=0.0025,
        help='train learning_rate')

    parser.add_argument(
        '--device',
        type=str, 
        default="cpu", 
        help='gpu or cpu')

    parser.add_argument(
        '--train_file',
        type=str, 
        default=None, 
        help='corpus for train')
    parser.add_argument(
        '--dev_file',
        type=str, 
        default=None, 
        help='corpus for dev')
    parser.add_argument(
        '--test_file',
        type=str, 
        default=None, 
        help='corpus for test')
    args = parser.parse_args()
    fit_device(args)
    model = DepdencyDrivenEE(args)
    if args.mode == "train":
        model.train(args)
    elif args.mode == "predict":
        while True:
            sent = input("Input Sent:")
            if sent == "q":
                break
            res = model.predict(sent)
            print(res)
        print("exit predict")
    elif args.mode == "export":
        model.export()
    else:
        raise ValueError("mode error must be one of [train, predict, export]")
