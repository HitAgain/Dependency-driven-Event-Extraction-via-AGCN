#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec

# torch version
# def get_attention(self, val_out, dep_embed, adj):
#     batch_size, max_len, feat_dim = val_out.shape
#     val_us = val_out.unsqueeze(dim=2)
#     val_us = val_us.repeat(1,1,max_len,1)
#     val_cat = torch.cat((val_us, dep_embed), -1)
#     atten_expand = (val_cat.float() * val_cat.float().transpose(1,2))
#     attention_score = torch.sum(atten_expand, dim=-1)
#     attention_score = attention_score / feat_dim ** 0.5
#     # softmax
#     exp_attention_score = torch.exp(attention_score)
#     exp_attention_score = torch.mul(exp_attention_score.float(), adj.float())
#     sum_attention_score = torch.sum(exp_attention_score, dim=-1).unsqueeze(dim=-1).repeat(1,1,max_len)
#     attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)
#     return attention_score

class DepAttention(tf.keras.layers.Layer):

    def __init__(self):
        super(DepAttention, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(DepAttention, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        val_out, dep_embed, adj = inputs[0] ,inputs[1], inputs[2]
        val_us = tf.expand_dims(val_out, axis = 2)
        val_us = tf.tile(val_us, [1, 1, tf.shape(val_out)[1], 1])
        val_cat = tf.concat([val_us, dep_embed], axis = -1)
        val_cat_T = tf.transpose(val_cat, perm=[0,2,1,3])
        atten_expand =  tf.multiply(val_cat, val_cat_T)
        attention_score = tf.reduce_sum(atten_expand, axis = -1, keepdims=False)
        attention_score = tf.divide(attention_score, tf.math.sqrt(tf.shape(val_out)[2]))
        # softmax
        exp_attention_score = tf.math.exp(attention_score)
        exp_attention_score = tf.multiply(exp_attention_score, adj)
        sum_attention_score = tf.reduce_sum(exp_attention_score, axis = -1)
        sum_attention_score = tf.expand_dims(sum_attention_score, axis = -1)
        sum_attention_score = tf.tile(sum_attention_score, [1, 1, tf.shape(val_out)[1]])
        res = tf.divide(exp_attention_score,
                        tf.add(sum_attention_score,tf.convert_to_tensor(1e-10)))
        return res

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            mask = [None]
        return mask[0]

    def compute_output_shape(self, input_shape):
        return (None, input_shape[2][1], input_shape[2][2])

    def get_config(self):
        config = super(DepAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DepGcn(tf.keras.layers.Layer):

    def __init__(self, out_feature_dim, bias = True, **kwargs):
        super(DepGcn, self).__init__(**kwargs)
        self.supports_masking = True
        self.out_feature_dim = out_feature_dim
        self.bias = bias
        self.weight, self.bias_weight = None, None

    def build(self, input_shape):
        inp_feature_dim = input_shape[0][-1]
        self.weight = self.add_weight(name="depgcn_weight",
                                      shape=[inp_feature_dim, self.out_feature_dim],
                                      dtype=tf.float32,
                                      #initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed)
                                     )
        if self.bias:
            self.bias_weight = self.add_weight(shape=(self.out_feature_dim,),
                                        #initializer=self.bias_initializer,
                                        #regularizer=self.bias_regularizer,
                                        #constraint=self.bias_constraint,
                                        dtype=tf.float32,
                                        name = "depgcn_bias"
                                    )
        #self.built = True
        super(DepGcn, self).build(input_shape)


    def call(self, inputs, mask=None, **kwargs):
        # dep_embed = [B * SeqLen * SeqLen * DepTypeEmbSize]
        hidden_status, adj, dep_embed = inputs[0], inputs[1], inputs[2]
        #max_len = hidden_status.get_shape().as_list()[1]
        #hidden_dim = hidden_status.get_shape().as_list()[2]
        # [B * SeqLen * 1 * HiddenSize]
        hidden_status_us = tf.expand_dims(hidden_status, axis = 2)
        # [B * SeqLen * SeqLen * HiddenSize]
        hidden_status_us_rp = tf.tile(hidden_status_us, [1, 1, tf.shape(hidden_status)[1], 1])
        # [B * SeqLen * SeqLen * HiddenSize]
        hidden_status_us_rp_sum = tf.add(hidden_status_us_rp, dep_embed)

        adj_us = tf.expand_dims(adj, axis = -1)
        #adj_us_rp = tf.tile(adj_us, [1,1,1,hidden_dim])
        adj_us_rp =  tf.tile(adj_us, [1,1,1,self.out_feature_dim])
        fussion = tf.matmul(hidden_status_us_rp_sum, self.weight)
        fussion_T = tf.transpose(fussion, perm=[0, 2, 1, 3])
        output = tf.multiply(fussion_T, adj_us_rp)
        output_final = tf.reduce_sum(output, axis=2, keepdims=False)

        if self.bias:
            output_final_bias = tf.add(output_final, self.bias_weight)
            return output_final_bias
        else:
            return output_final

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            mask = [None]
        return mask[0]

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], self.out_feature_dim)

    def get_config(self):
        config = super(DepGcn, self).get_config()
        config.update({
            'out_feature_dim': self.out_feature_dim,
            'bias': self.bias
        })
        return dict(list(base_config.items()) + list(config.items()))
