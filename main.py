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
logging.info("[Config]label2id && id2label dic init success")
token_dict = load_vocab(dict_path="./albert/vocab.txt")
tokenizer = Tokenizer(token_dict, do_lower_case=False)
logging.info("[Config]albert tokenizer init success")

# albert-tokenizer init
count = 0
dep_map = {}
with open('./label/deptype.txt', 'r', encoding="utf-8") as fin:
    for line in fin:
        dep_type = line.strip("\n")
        dep_map[dep_type] = count
        count += 1
logging.info("[Config]dep type map init success")

# pretrain word2vec init
model = KeyedVectors.load_word2vec_format("/home/EventAlgorithm/hot_event_type_model/pretrained/model/train.w2v.100000.model", binary=False)
word_vocabulary = dict([(k, v.index + 3) for k, v in model.vocab.items()])
word_vocabulary["</p>"] = 0  # 补零
word_vocabulary["</u>"] = 1  # oov
word_vocabulary[" "] = 2  # 空格
weights = np.vstack((model.syn0[-1], model.syn0[-2], model.syn0[-3], model.syn0))
logging.info("[Config]pretrain word2vec dict && weights prepared success")

#############################数据读入模块#############################
def get_sample(filename):
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
            token_ids, segment_ids = tokenizer.encode(text)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)

            # word input && word2char transform matrix
            char_ls = list(text)
            word_ls = dep_feature["word_ls"].split("|")
            word2char_matrix = self.gen_word_to_char_matrix(word_ls, char_ls)
            word_ids = [word_vocabulary.get(word, 1) for word in word_ls]
            batch_word_ids.append(np.array(word_ids))
            batch_word2char_mat.append(word2char_matrix)

            # dep input
            relations = dep_feature["dep_type"].split("|")
            heads = dep_feature["dep_link"].split("|")
            dep_link_mat, dep_type_mat = self.gen_dep_matrix(relations, heads)
            batch_dep_type_mat.append(dep_type_mat)
            batch_dep_link_mat.append(dep_link_mat)

            # label
            labels = [0] * len(token_ids)
            for schema, word in feature.items():
                a_token_ids = tokenizer.encode(word)[0][1:-1]
                start_index = search(a_token_ids, token_ids)
                if start_index != -1:
                    labels[start_index] = label2id[schema] * 2 + 1
                    for i in range(1, len(a_token_ids)):
                        labels[start_index + i] = label2id[schema] * 2 + 2
            batch_labels.append(labels)

            if len(batch_token_ids) == self.batch_size or is_end:
                ## max rows
                max_rows = max([len(i) for i in batch_feature_char])
                ## max cols
                max_cols = max([len(i) for i in batch_feature_word])

                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                batch_word_ids = sequence_padding(batch_word_ids)
                batch_word2char_mat = self.matrix_padding(max_cols, max_rows, batch_word2char_mat)
                batch_dep_link_mat = self.matrix_padding(max_cols, max_cols, batch_dep_link_mat)
                batch_dep_type_mat = self.matrix_padding(max_cols, max_cols, batch_dep_type_mat)

                # crf层要求label dims=3 最后一个维度为1
                batch_label_reshape = batch_labels.reshape(
                    (batch_labels.shape[0], batch_labels.shape[1], 1))
                yield [batch_token_ids, batch_segment_ids, batch_word_ids, batch_dep_type_mat, batch_dep_link_mat, batch_word2char_mat], batch_label_reshape
                batch_token_ids, batch_segment_ids, batch_word_ids, batch_dep_type_mat, batch_dep_link_mat, batch_word2char_mat, batch_labels\
                    = [], [], [], [], [], [], []

    def gen_word_to_char_matrix(self, word_ls, char_ls):
        matrix = np.zeros((len(char_ls), len(word_ls)))
        start_ind = 0
        for ind, word in enumerate(word_ls):
            for j in range(start_ind, start_ind + len(word)):
                matrix[j][ind] = 1
            start_ind += len(word)
        return matrix

    def gen_dep_matrix(self, relations, heads):
        dep_link_matrix = np.eye(len(relations), dtype=float)
        for tail, head in enumerate(heads):
            if head != 0:
                dep_link_matrix[tail][head - 1] = 1.0
                dep_link_matrix[head - 1][tail] = 1.0
        dep_type_matrix = np.zeros(len(relations), len(relations), dtype = int)
        for i in range(len(relations)):
            if heads[i] == 0 and relations[i] == 'HED':
                dep_type_matrix[i][i] = dep_map["HED"]
            else:
                dep_type_matrix[i][heads[i] - 1] = dep_map[relations[i]]
                dep_type_matrix[heads[i] - 1][i] = dep_map[relations[i]]
        return dep_link_matrix, dep_type_matrix

    def matrix_padding(self, max_cols, max_rows, ori_mat_ls):
        res = []
        for mat in ori_mat_ls:
            # col padding
            mat = np.concatenate((mat, np.zeros((len(mat), max_cols - len(mat[0])))), axis = 1)
            # row padding
            mat = np.concatenate((mat, np.zeros((max_rows - len(mat), len(mat[0])))), axis = 0)
            res.append(mat)
        return np.array(res)

#############################训练回调模块#############################
class Debug(Callback):
    def __init__(self):
        self.dev_path = './data/dev.json'
        self.test_path = './data/test.json'
        self.dev_data = get_sample(self.dev_path)
        self.test_data = list()
        with open(self.test_path, 'r', encoding="utf-8") as f:
            for l in f:
                l = l.strip("\n").split("\t")
                text = l[0]
                self.test_data.append(text)
        self.best_F1 = 0.
        print("========Evaluate Callback init success=========")

    def extract_arguments(self, text):
        token_ids, segment_ids = tokenizer.encode(text)
        token_np = np.array([token_ids])
        segment_np = np.array([segment_ids])
        predict_ids = list()
        scores = self.model.predict([token_np, segment_np])[0]
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
        # 返回schema属性和词片段
        res = defaultdict(list)
        for l, arg_ids in arguments:
            res[l].append(tokenizer.decode([token_ids[ind] for ind in arg_ids]))
        return res

    def evaluate(self):
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for text, arguments in self.dev_data:
            pred_arguments = self.extract_arguments(text)
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
        no_schema = 0
        total_num = len(self.test_data)
        with open("./test_result.txt", "w", encoding="utf-8") as fout:
            for text in self.test_data:
                fout.write("{}\t".format(text))
                extract_schema = self.extract_arguments(text)
                if len(extract_schema) == 0:
                    no_schema += 1
                for label, schema in extract_schema.items():
                    fout.write("{}:{}\t".format(label, schema))
                fout.write("\n")
            fout.write("覆盖率：{}\n".format((total_num - no_schema) / total_num))
        print("test file result write in {}".format("test_result.txt"))

    def on_epoch_end(self, epoch, logs={}):
        print("============ Dev Test   ==============")
        f1, precision, recall = self.evaluate()
        print("Epoch:{}|||f1:{}--P:{}--R:{}".format(epoch, f1, precision, recall))
        if f1 > self.best_F1:
            print(
                "=============== BEST f1 change to {} ==================".format(f1))
            self.best_F1 = f1
            self.model.save_weights('./best_model_weights.h5')
        else:
            print("=========epoch {} F1 NOT BETTER============".format(epoch))

class DepdencyDrivenEE(object):

    def __init__(self, mode="train"):
        self.mode = mode
        self.epochs = 5
        self.batch_size = 32
        self.learning_rate = 0.0025
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
        words_embedder = Embedding(10000,
                                   128,
                                   trainable=False,
                                   mask_zero=True,
                                   name="words_embedding_layer")
        dep_embedder = Embedding(30,
                                 128,
                                 trainable=False,
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
            logging.info("===========model build success in train mode============")
            return model
        else:
            model.load_weights("./best_model_weights.h5")
            logging.info("===========model build success in inference mode============")
            return model

    def train(self):
        train_data = get_sample("./data/train.txt")
        train_generator = data_generator(train_data, self.batch_size)
        debug_callback = Debug()
        self.model.fit_generator(
            train_generator.forfit(),
            epochs=self.epochs,
            steps_per_epoch=len(train_generator),
            verbose=1,
            callbacks=[debug_callback]
        )


    def predict(self, text):
        token_ids, segment_ids = tokenizer.encode(text)
        token_np = np.array([token_ids])
        segment_np = np.array([segment_ids])
        predict_ids = list()
        scores = self.model.predict([token_np, segment_np])[0]
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
        # 返回schema属性和词片段
        res = defaultdict(list)
        for l, arg_ids in arguments:
            res[l].append(tokenizer.decode([token_ids[ind] for ind in arg_ids]))
        return res

    def export(self):
        save_model = self.build_model()
        save_model.load_weights("./best_model_weights.h5")
        tf.saved_model.save(save_model, "./serving")
        logging.info("check online model in dir:{}".format("./title_gen_serving"))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--mode',
    #     type=str,
    #     default="train",
    #     help='train or inference')
    # parser.add_argument(
    #     '--device',
    #     type=str, 
    #     default="gpu", 
    #     help='gpu or cpu')
    # args = parser.parse_args()
    # if args.device == "gpu":
    #     gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)
    # tf.compat.v1.disable_eager_execution()
    # ee = DepdencyDrivenEE(args.mode)