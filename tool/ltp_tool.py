import os
import time
import argparse
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
from pyltp import SentenceSplitter, Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller

segmentor = Segmentor()
segmentor.load(os.path.join("/home/ltp_model_3.4.0", "cws.model"))

postagger = Postagger()
postagger.load(os.path.join("/home/ltp_model_3.4.0", "pos.model"))

parser = Parser()
parser.load(os.path.join("/home/ltp_model_3.4.0", "parser.model"))

labeller = SementicRoleLabeller()
labeller.load(os.path.join("/home/ltp_model_3.4.0", 'pisrl.model'))

def ltp_parse(sentence):
    words = segmentor.segment(sentence)
    postags = postagger.postag(words)
    arcs = parser.parse(words, postags)
    return words, postags, arcs

def critic(stringa, stringb):
    res = list(set(list(stringa)).intersection(set(list(stringb))))
    return (2 * len(res)) / (len(stringa) + len(stringb))

def search(pattern, sequence):
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1
