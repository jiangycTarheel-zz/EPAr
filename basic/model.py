import random
import os
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell

from basic.read_data import DataSet
from basic.batcher import get_feed_dict as _get_feed_dict
from my.tensorflow import get_initializer
from my.tensorflow.nn import softsel, get_logits, highway_network, multi_conv1d, reconstruct_batches, span_to_avg_emb, reconstruct_batchesV2, select_topn_doc_idx, reconstruct_batchesV3
from my.tensorflow.rnn import bidirectional_dynamic_rnn
from my.tensorflow.rnn_cell import SwitchableDropoutWrapper, AttentionCell
from reasoning_layers.mac_layer import MACRnn, dynamic_mac_rnn, HierarchicalAttnMACRnn
from reasoning_layers.assembler import BiAttnAssembler
from my.tensorflow.ops import bi_cudnn_rnn_encoder


def get_multi_gpu_models(config, emb_mat=None):
  models = []
  with tf.variable_scope(tf.get_variable_scope()) as vscope:
    for gpu_idx in range(config.num_gpus):
      with tf.name_scope("model_{}".format(gpu_idx)) as scope, tf.device("/{}:{}".format(config.device_type, gpu_idx)):
        if gpu_idx > 0:
          tf.get_variable_scope().reuse_variables()
        model = Model(config, scope, emb_mat, rep=gpu_idx == 0)
        models.append(model)
  return models


class Model(object):
  def __init__(self, config, scope, emb_mat, rep=True):
    self.scope = scope
    self.config = config
    self.emb_mat = emb_mat
    self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                       initializer=tf.constant_initializer(0), trainable=False)

    if config.split_supports is True:
      N, M, JX, JQ, VW, VC, W = \
        config.batch_size, 1, config.max_para_size, \
        config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.max_word_size
      self.x = tf.placeholder('int32', [None, None, None], name='x')
      self.cx = tf.placeholder('int32', [None, None, None, W], name='cx')
      self.x_mask = tf.placeholder('bool', [None, None, None], name='x_mask')
      self.x_sents_len = tf.placeholder('int32', [None, M, 10], name='x_sents_len')
    else:
      # Define forward inputs here
      N, M, JX, JQ, VW, VC, W = \
        config.batch_size, config.max_num_sents, config.max_sent_size, \
        config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.max_word_size
      
      self.x = tf.placeholder('int32', [N, None, None], name='x')
      self.cx = tf.placeholder('int32', [N, None, None, W], name='cx')
      self.x_mask = tf.placeholder('bool', [N, None, None], name='x_mask')

    self.q = tf.placeholder('int32', [N, None], name='q')
    self.cq = tf.placeholder('int32', [N, None, W], name='cq')
    self.q_sub = tf.placeholder('int32', [N, None], name='q_sub')
    self.cq_sub = tf.placeholder('int32', [N, None, W], name='cq_sub')
    self.q_mask = tf.placeholder('bool', [N, None], name='q_mask')
    self.q_sub_mask = tf.placeholder('bool', [N, None], name='q_sub_mask')
    self.y = tf.placeholder('bool', [N, None, None], name='y')
    self.y2 = tf.placeholder('bool', [N, None, None], name='y2')
    self.wy = tf.placeholder('bool', [N, None, None], name='wy')
    self.is_train = tf.placeholder('bool', [], name='is_train')
    self.new_emb_mat = tf.placeholder('float', [None, config.word_emb_size], name='new_emb_mat')
    self.na = tf.placeholder('bool', [N], name='na')

    if (config.reasoning_layer is not None and config.mac_prediction == 'candidates'):
      self.candidate_spans = tf.placeholder('int32', [N, None, None, 2], name='cand_spans')
      self.candidate_span_y = tf.placeholder('int32', [N, None], name='cand_span_y')
      self.num_exceed_cand = tf.placeholder('int32', [N, None, None], name='num_exceed_cand')

    self.x_group = tf.placeholder('int32', [N], name='x_group')  # Define how sentences could be grouped into batch

    if config.supervise_first_doc:
      self.first_doc_ids = tf.placeholder('int32', [N], name='first_doc_ids')

    if config.use_assembler:
      self.selected_sent_ids = tf.placeholder('int32', [config.batch_size, config.num_hops], name='selected_sent_ids')

    self.answer_doc_ids = tf.placeholder('int32', [N, None], name='answer_doc_ids')
    self.answer_word_ids = tf.placeholder('int32', [N, None], name='answer_word_ids')
    
    self.period_id = None

    # Define misc
    self.tensor_dict = {}

    # Forward outputs / loss inputs
    self.logits = None
    self.yp = None
    self.var_list = None
    self.na_prob = None

    # Loss outputs
    self.loss = None

    self._build_forward()
    self._build_loss()
    self.var_ema = None
    if rep:
      self._build_var_ema()
      if config.mode == 'train':
        self._build_ema()

    self.summary = tf.summary.merge_all()
    self.summary = tf.summary.merge(tf.get_collection("summaries", scope=self.scope))


  def _build_forward(self):
    config = self.config

    N, M, JX, JQ, VW, VC, d, W = \
      config.batch_size, config.max_num_sents, config.max_sent_size, \
      config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, \
      config.max_word_size
    if config.split_supports:
      M = 1
    JX = tf.shape(self.x)[2]
    JQ = tf.shape(self.q)[1]
    JQ_sub = tf.shape(self.q_sub)[1]

    M = tf.shape(self.x)[1]
    dc, dw, dco = config.char_emb_size, config.word_emb_size, config.char_out_size

    with tf.variable_scope("emb"):
      if config.use_char_emb:
        with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
          char_emb_mat = tf.get_variable("char_emb_mat", shape=[VC, dc], dtype='float')

        with tf.variable_scope("char"):
          Acx = tf.nn.embedding_lookup(char_emb_mat, self.cx)  # [N, M, JX, W, dc]
          Acq = tf.nn.embedding_lookup(char_emb_mat, self.cq)  # [N, JQ, W, dc]      
          
          Acx = tf.reshape(Acx, [-1, JX, W, dc])
          Acq = tf.reshape(Acq, [-1, JQ, W, dc])
          if config.get_query_subject:
            Acq_sub = tf.nn.embedding_lookup(char_emb_mat, self.cq_sub)  # [N, JQ, W, dc]
            Acq_sub = tf.reshape(Acq_sub, [-1, JQ_sub, W, dc])

          filter_sizes = list(map(int, config.out_channel_dims.split(',')))
          heights = list(map(int, config.filter_heights.split(',')))
          assert sum(filter_sizes) == dco, (filter_sizes, dco)
          with tf.variable_scope("conv"):
            xx = multi_conv1d(Acx, filter_sizes, heights, "VALID",  self.is_train, config.keep_prob, scope="xx")
            if config.share_cnn_weights:
              tf.get_variable_scope().reuse_variables()
              qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="xx")
              if config.get_query_subject:
                qq_sub = multi_conv1d(Acq_sub, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="xx")
              
            else:
              qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="qq")
              if config.get_query_subject:
                qq_sub = multi_conv1d(Acq_sub, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="qq")

            xx = tf.reshape(xx, [-1, M, JX, dco])
            qq = tf.reshape(qq, [-1, JQ, dco])
            if config.get_query_subject:
              qq_sub = tf.reshape(qq_sub, [-1, JQ_sub, dco])


      if config.use_word_emb:
        with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
          if config.mode == 'train':
            word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[VW, dw], initializer=get_initializer(self.emb_mat))
          else:
            word_emb_mat = tf.get_variable("word_emb_mat", shape=[VW, dw], dtype='float')
          if config.use_glove_for_unk:
            word_emb_mat = tf.concat(axis=0, values=[word_emb_mat, self.new_emb_mat])

        with tf.name_scope("word"):
          Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [N, M, JX, d]
          Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [N, JQ, d]

          if config.get_query_subject:
            Aq_sub = tf.nn.embedding_lookup(word_emb_mat, self.q_sub)
            self.tensor_dict['q_sub'] = Aq_sub
          
          self.tensor_dict['x'] = Ax
          self.tensor_dict['q'] = Aq
        if config.use_char_emb:
          xx = tf.concat(axis=3, values=[xx, Ax])  # [N, M, JX, di]
          qq = tf.concat(axis=2, values=[qq, Aq])  # [N, JQ, di]
          if config.get_query_subject:
            qq_sub = tf.concat(axis=2, values=[qq_sub, Aq_sub])

        else:
          xx = Ax
          qq = Aq
          if config.get_query_subject:
            qq_sub = Aq_sub

    # highway network
    if config.highway:
      with tf.variable_scope("highway"):
        xx = highway_network(xx, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train, input_keep_prob=config.highway_keep_prob)
        tf.get_variable_scope().reuse_variables()
        qq = highway_network(qq, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train, input_keep_prob=config.highway_keep_prob)
        if config.get_query_subject:
          qq_sub = highway_network(qq_sub, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train, input_keep_prob=config.highway_keep_prob)

    self.tensor_dict['xx'] = xx
    self.tensor_dict['qq'] = qq

    cell_fw = BasicLSTMCell(d, state_is_tuple=True)
    cell_bw = BasicLSTMCell(d, state_is_tuple=True)
    d_cell_fw = SwitchableDropoutWrapper(cell_fw, self.is_train, input_keep_prob=config.input_keep_prob)
    d_cell_bw = SwitchableDropoutWrapper(cell_bw, self.is_train, input_keep_prob=config.input_keep_prob)
    cell2_fw = BasicLSTMCell(d, state_is_tuple=True)
    cell2_bw = BasicLSTMCell(d, state_is_tuple=True)
    d_cell2_fw = SwitchableDropoutWrapper(cell2_fw, self.is_train, input_keep_prob=config.input_keep_prob)
    d_cell2_bw = SwitchableDropoutWrapper(cell2_bw, self.is_train, input_keep_prob=config.input_keep_prob)
    cell3_fw = BasicLSTMCell(d, state_is_tuple=True)
    cell3_bw = BasicLSTMCell(d, state_is_tuple=True)
    d_cell3_fw = SwitchableDropoutWrapper(cell3_fw, self.is_train, input_keep_prob=config.input_keep_prob)
    d_cell3_bw = SwitchableDropoutWrapper(cell3_bw, self.is_train, input_keep_prob=config.input_keep_prob)
    cell4_fw = BasicLSTMCell(d, state_is_tuple=True)
    cell4_bw = BasicLSTMCell(d, state_is_tuple=True)
    d_cell4_fw = SwitchableDropoutWrapper(cell4_fw, self.is_train, input_keep_prob=config.input_keep_prob)
    d_cell4_bw = SwitchableDropoutWrapper(cell4_bw, self.is_train, input_keep_prob=config.input_keep_prob)
    x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 2)  # [N, M]
    q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]
    q_sub_len = tf.reduce_sum(tf.cast(self.q_sub_mask, 'int32'), 1)  # [N]

    with tf.variable_scope("prepro"):
      
      if config.cudnn_rnn:
        if config.reasoning_layer == 'mac_rnn' and config.use_control_unit is False:
          with tf.variable_scope('u1'):
            u_bod, _ = bi_cudnn_rnn_encoder('lstm', config.hidden_size, 1, 1-config.input_keep_prob, qq, q_len-q_sub_len, self.is_train)
            u_st = zhong_selfatt(tf.expand_dims(u_bod, axis=1), config.hidden_size*2, seq_len=q_len-q_sub_len, transform='squeeze')
            tf.get_variable_scope().reuse_variables()
            u, _ = bi_cudnn_rnn_encoder('lstm', config.hidden_size, 1, 1-config.input_keep_prob, qq, q_len, self.is_train)
        else: # go to this case if answer_state_update_rule == 'bi-attn'
          with tf.variable_scope('u1'):
            u, _ = bi_cudnn_rnn_encoder('lstm', config.hidden_size, 1, 1-config.input_keep_prob, qq, q_len, self.is_train)
            if config.reasoning_layer == 'mac_rnn':
              u_st = zhong_selfatt(tf.expand_dims(u, axis=1), config.hidden_size*2, seq_len=q_len, transform='squeeze')
         
        q_sub_st = None 
        if config.share_lstm_weights:
          with tf.variable_scope('u1', reuse=True):
            h, _ = bi_cudnn_rnn_encoder('lstm', config.hidden_size, 1, 1-config.input_keep_prob, tf.squeeze(xx, axis=1), tf.squeeze(x_len, axis=1), self.is_train)
            h = tf.expand_dims(h, axis=1)
            if config.reasoning_layer == 'mac_rnn':
              h_st = zhong_selfatt(h, config.hidden_size*2, seq_len=tf.squeeze(x_len, axis=1), transform='squeeze')
            else: # Need a dumy h_st
              h_st = tf.reduce_mean(tf.squeeze(h, axis=1), axis=1)
            if config.get_query_subject:
              q_sub, _ = bi_cudnn_rnn_encoder('lstm', config.hidden_size, 1, 1-config.input_keep_prob, qq_sub, q_sub_len, self.is_train)
              q_sub_st = zhong_selfatt(tf.expand_dims(q_sub, axis=1), config.hidden_size*2, seq_len=q_sub_len, transform='squeeze')
           
      else:
        if config.reasoning_layer == 'mac_rnn' and config.use_control_unit is False:
          # If control_unit is False, only encode the query body
          (fw_u, bw_u), (fw_u_f_st, bw_u_f_st) = bidirectional_dynamic_rnn(d_cell_fw, d_cell_bw, qq, q_len - q_sub_len, dtype='float', scope='u1')  # [N, J, d], [N, d]  
          u_st = tf.concat(axis=1, values=[fw_u_f_st.c, bw_u_f_st.c])
          #if config.bidaf:
          (fw_u, bw_u), _ = bidirectional_dynamic_rnn(d_cell_fw, d_cell_bw, qq, q_len, dtype='float', scope='u1')  # [N, J, d], [N, d]  
          u = tf.concat(axis=2, values=[fw_u, bw_u])
        else: # go to this case if answer_state_update_rule == 'bi-attn'
          (fw_u, bw_u), (fw_u_f_st, bw_u_f_st) = bidirectional_dynamic_rnn(d_cell_fw, d_cell_bw, qq, q_len, dtype='float', scope='u1')  # [N, J, d], [N, d]  
          u = tf.concat(axis=2, values=[fw_u, bw_u])

        if config.share_lstm_weights:
          tf.get_variable_scope().reuse_variables()
          (fw_h, bw_h), (fw_h_f_st, bw_h_f_st) = bidirectional_dynamic_rnn(cell_fw, cell_bw, xx, x_len, dtype='float', scope='u1')  # [N, M, JX, 2d]
          h = tf.concat(axis=3, values=[fw_h, bw_h])  # [N, M, JX, 2d]
          h_st = tf.concat(axis=1, values=[fw_h_f_st.c, bw_h_f_st.c])  # [N, M, 2d]
          if config.get_query_subject:
            _, (fw_u2_f_st, bw_u2_f_st) = bidirectional_dynamic_rnn(cell_fw, cell_bw, qq_sub, q_sub_len, dtype='float', scope='u1')  # [N, M, JX, 2d]
            q_sub_st = tf.concat(axis=1, values=[fw_u2_f_st.c, bw_u2_f_st.c])  # [N, M, 2d]
          else:
            q_sub_st = None
        else:
          (fw_h, bw_h), (fw_h_f_st, bw_h_f_st) = bidirectional_dynamic_rnn(cell_fw, cell_bw, xx, x_len, dtype='float', scope='h1')  # [N, M, JX, 2d]
          h = tf.concat(axis=3, values=[fw_h, bw_h])  # [N, M, JX, 2d]
          h_st = tf.concat(axis=1, values=[fw_h_f_st.c, bw_h_f_st.c])  # [N, M, 2d]
          if config.get_query_subject:
            tf.get_variable_scope().reuse_variables()
            _, (fw_u2_f_st, bw_u2_f_st) = bidirectional_dynamic_rnn(cell_fw, cell_bw, qq_sub, q_sub_len, dtype='float', scope='u1')  # [N, M, JX, 2d]
            q_sub_st = tf.concat(axis=2, values=[fw_u2_f_st.c, bw_u2_f_st.c])  # [N, M, 2d]
          else:
            q_sub_st = None
      self.tensor_dict['u'] = u
      self.tensor_dict['h'] = h

    with tf.variable_scope("main"):
      context_dim = config.hidden_size * 2
      # Reconstruct before bidaf because otherwise we need to build a larger query tensor.
      if config.split_supports: # Reconstruct batches into [N, M, JX, 2d]
        if config.select_top_n_doc > 0:
          first_n_doc_idx = select_topn_doc_idx(N, config.select_top_n_doc, self.x_group)
          h_plus_one = tf.concat([h, tf.expand_dims(tf.zeros_like(h[0], tf.float32), axis=0)], axis=0)
          h_st_plus_one = tf.concat([h_st, tf.expand_dims(tf.zeros_like(h_st[0], tf.float32), axis=0)], axis=0)
          x_len_plus_one = tf.concat([x_len, tf.expand_dims(tf.zeros_like(x_len[0], tf.int32), axis=0)], axis=0)
          x_mask_plus_one = tf.concat([self.x_mask, tf.expand_dims(tf.zeros_like(self.x_mask[0], tf.bool), axis=0)], axis=0)
          top_n_h = tf.gather(h_plus_one, first_n_doc_idx)
          top_n_h_st = tf.gather(h_st_plus_one, first_n_doc_idx)
          top_n_x_len = tf.gather(x_len_plus_one, first_n_doc_idx)
          top_n_x_mask = tf.gather(x_mask_plus_one, first_n_doc_idx)
        
        if config.hierarchical_attn is False:
          h, x_len, x_mask = reconstruct_batches(h, x_len, self.x_group, target_batch_size=N, \
            max_para_size=config.max_para_size, model=self)
        else:
          if config.bidaf:
            context_dim = config.hidden_size * 4
            # Augment query to match 
            batch_nums = []
            for i in range(config.batch_size):
              batch_nums = tf.concat([batch_nums, tf.tile([i], [self.x_group[i]])], axis=0)

            u_tiled = tf.gather(u, batch_nums)
            q_mask_tiled = tf.gather(self.q_mask, batch_nums)
            h = attention_layer(config, self.is_train, h, u_tiled, h_mask=self.x_mask, u_mask=q_mask_tiled, scope="p0", tensor_dict=self.tensor_dict)
            W = tf.get_variable('W', [160, 80])
            b = tf.get_variable('b', [80])
            h = tf.einsum('ijkl,lm->ijkm',h,W) + b
          h_reconstruct, _, _ = reconstruct_batches(h, x_len, self.x_group, target_batch_size=N, \
            max_para_size=config.max_para_size, model=self, emb_dim=context_dim)
          
          if config.select_top_n_doc > 1:
            top_n_x_group = []
            for i in range(N):
              to_append = tf.cond(self.x_group[i] > config.select_top_n_doc, lambda: config.select_top_n_doc, lambda: self.x_group[i])
              top_n_x_group.append(to_append)
            top_n_x_group = tf.stack(top_n_x_group)
            h, p_st, x_mask, pdoc_mask, self.x_sents_len_reconstruct = reconstruct_batchesV2(top_n_h, top_n_h_st, top_n_x_mask, top_n_x_group, self.x_sents_len, target_batch_size=N, \
              max_para_size=config.max_para_size, model=self)
          else:  
            h, p_st, x_mask, pdoc_mask, self.x_sents_len_reconstruct = reconstruct_batchesV2(h, h_st, self.x_mask, self.x_group, self.x_sents_len, target_batch_size=N, \
              max_para_size=config.max_para_size, model=self)
        if config.select_top_n_doc > 0:
          x_len = top_n_x_len
      else:
        x_mask = self.x_mask

      if config.bidaf and config.hierarchical_attn is False:
        context_dim = config.hidden_size * 8
        if config.use_control_unit is False and config.reasoning_layer == 'mac_rnn':
          if config.select_top_n_doc > 0:
            p0 = attention_layer(config, self.is_train, top_n_h, u, h_mask=top_n_x_mask, u_mask=self.q_mask, scope="p0", tensor_dict=self.tensor_dict)
          else:
            p0 = attention_layer(config, self.is_train, h, u, h_mask=x_mask, u_mask=self.q_mask, scope="p0", tensor_dict=self.tensor_dict)
        else:
          if config.select_top_n_doc > 0:
            p0 = attention_layer(config, self.is_train, top_n_h, u, h_mask=top_n_x_mask, u_mask=self.q_mask, scope="p0", tensor_dict=self.tensor_dict)
          else:
            p0 = attention_layer(config, self.is_train, h, u, h_mask=x_mask, u_mask=self.q_mask, scope="p0", tensor_dict=self.tensor_dict)
      else:
        p0 = h

      first_cell_fw = d_cell2_fw
      second_cell_fw = d_cell3_fw
      first_cell_bw = d_cell2_bw
      second_cell_bw = d_cell3_bw

      if config.reasoning_layer == 'mac_rnn':
        query_dim = config.hidden_size * 2

        if config.hierarchical_attn:
          mac_rnn_cell = HierarchicalAttnMACRnn(config.batch_size, context_dim, query_dim, num_hops=config.num_hops, reuse_cell=config.reuse_cell, \
            is_train=self.is_train, use_control_unit=config.use_control_unit, mode=config.mode, read_strategy=config.mac_read_strategy, \
            output_unit_type=config.mac_output_unit, answer_state_update_rule=config.mac_answer_state_update_rule, reasoning_unit=config.mac_reasoning_unit, \
            memory_state_update_rule=config.mac_memory_state_update_rule, \
            answer_doc_ids=self.answer_doc_ids if config.supervise_final_doc or (config.oracle is not None) else None, \
            sents_len=self.x_sents_len_reconstruct, oracle=config.oracle, \
            input_keep_prob=config.input_keep_prob, \
            attention_cell_dropout=config.attention_cell_dropout, read_topk_docs=config.read_topk_docs)
          self.mac_rnn_cell = mac_rnn_cell

          if config.mac_prediction == 'candidates': 
            cand_emb, cand_mask = span_to_avg_emb(self.candidate_spans, h_reconstruct, config.batch_size, self)
            g1 = dynamic_mac_rnn(mac_rnn_cell, p0, u, q_len, x_mask, self.q_mask, q_sub_st=q_sub_st, context_st=p_st, query_st=u_st, cdoc_mask=pdoc_mask, candidates=cand_emb, cand_mask=cand_mask)
            
            self.doc_attn_logits = mac_rnn_cell.doc_attn_logits_lst
            self.word_attn_logits = mac_rnn_cell.word_attn_logits_lst 
            self.doc_labels = mac_rnn_cell.doc_attn
            self.g1 = g1
            self.cand_mask = cand_mask
            self.cand_emb = cand_emb
            self.pdoc_mask = pdoc_mask
            self.p_st = p_st
            logits = get_logits([g1], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob, mask=cand_mask, \
              is_train=self.is_train, func=config.answer_func, scope='logits1')
            
            JX = tf.shape(g1)[2]
            self.JX = JX
            self.g1_shape=tf.shape(g1)
            flat_logits = tf.reshape(logits, [config.batch_size, M * JX])
            flat_yp = tf.nn.softmax(flat_logits)  # [-1, M*JX]
            yp = tf.reshape(flat_yp, [config.batch_size, M, JX])
            self.logits = flat_logits
            self.yp = yp

            if config.use_assembler or config.attn_visualization:
              self.yp_list = []
              self.logits_list = []
              for i in range(config.num_hops):
                logits = get_logits([mac_rnn_cell.answer_list[i]], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob, mask=cand_mask, \
                  is_train=self.is_train, func=config.answer_func, scope='logits1', reuse=True)
                flat_logits = tf.reshape(logits, [config.batch_size, M * JX])
                flat_yp = tf.nn.softmax(flat_logits)
                yp = tf.reshape(flat_yp, [config.batch_size, M, JX])
                self.yp_list.append(yp)
                self.logits_list.append(flat_logits)

            if config.use_assembler:
              if config.assembler_type == 'BiAttn':
                self.assembler = BiAttnAssembler(config, self.is_train, self, context_dim=context_dim)
                self.assembler.build_forward(p0, x_mask, u, u_st, self.q_mask, cand_emb, cand_mask)
              else:
                raise NotImplementedError
          
            return
          else:
            raise NotImplementedError
        else:
          mac_rnn_cell = MACRnn(config.batch_size, p0.get_shape()[-1], u.get_shape()[-1], num_hops=config.num_hops, prediction=config.mac_prediction, \
            reuse_cell=config.reuse_cell, is_train=self.is_train, use_control_unit=config.use_control_unit, mode=config.mode)
          if config.mac_prediction == 'candidates':
            cand_emb, cand_mask = span_to_avg_emb(self.candidate_spans, p0, config.batch_size, self)
            g1 = dynamic_mac_rnn(mac_rnn_cell, p0, u, q_len, x_mask, self.q_mask, candidates=cand_emb, cand_mask=cand_mask, q_sub_st=q_sub_st)
            self.g1 = g1
            self.cand_mask = cand_mask
            self.cand_emb = cand_emb
            logits = get_logits([g1], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob, mask=cand_mask, \
              is_train=self.is_train, func=config.answer_func, scope='logits1')
            
            JX = tf.shape(g1)[2]
            flat_logits = tf.reshape(logits, [config.batch_size, M * JX])
            flat_yp = tf.nn.softmax(flat_logits)  # [-1, M*JX]
            yp = tf.reshape(flat_yp, [config.batch_size, M, JX])
            self.logits = flat_logits
            self.yp = yp
            return

          elif config.mac_prediction == 'span-dual':
            g1, g2 = dynamic_mac_rnn(mac_rnn_cell, p0, qq, q_len)
            if config.split_supports is True:
              M=1
              JX=config.max_para_size
              N=config.batch_size

            logits = get_logits([g1], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                mask=x_mask, is_train=self.is_train, func=config.answer_func, scope='logits1')
            logits2 = get_logits([g2], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                mask=x_mask, is_train=self.is_train, func=config.answer_func, scope='logits2')
          else:
            assert config.mac_prediction == 'span-single'
            g1, logits = dynamic_mac_rnn(mac_rnn_cell, p0, qq, q_len, x_mask, self.q_mask)
            if config.split_supports is True:
              M=1
              JX=config.max_para_size
              N=config.batch_size
            a1i = softsel(tf.reshape(g1, [N, M * JX, 80]), tf.reshape(logits, [N, M * JX]))
            a1i = tf.tile(tf.expand_dims(tf.expand_dims(a1i, 1), 1), [1, M, JX, 1])
      
      else:
        if config.cudnn_rnn:
          with tf.variable_scope('g0'):
            g0, _ = bi_cudnn_rnn_encoder('lstm', config.hidden_size, 1, 1-config.input_keep_prob, tf.squeeze(p0, axis=1), tf.squeeze(x_len, axis=1), self.is_train)
            g0 = tf.expand_dims(g0, axis=1)
        else:
          (fw_g0, bw_g0), _ = bidirectional_dynamic_rnn(first_cell_fw, first_cell_bw, p0, x_len, dtype='float', scope='g0')  # [N, M, JX, 2d]
          g0 = tf.concat(axis=3, values=[fw_g0, bw_g0])
        if config.cudnn_rnn:
          with tf.variable_scope('g1'):
            g1, _ = bi_cudnn_rnn_encoder('lstm', config.hidden_size, 1, 1-config.input_keep_prob, tf.squeeze(g0, axis=1), tf.squeeze(x_len, axis=1), self.is_train)
            g1 = tf.expand_dims(g1, axis=1)
        else:
          (fw_g1, bw_g1), (fw_g1_f_st, bw_g1_f_st) = bidirectional_dynamic_rnn(second_cell_fw, second_cell_bw, g0, x_len, dtype='float', scope='g1')  # [N, M, JX, 2d]
          g1 = tf.concat(axis=3, values=[fw_g1, bw_g1])

        if config.reasoning_layer == 'bidaf' and config.mac_prediction == 'candidates':

          
          logits = get_logits([g1], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob, mask=x_mask, is_train=self.is_train, scope='a_state_logits')
          probs = tf.nn.softmax(logits)
          a_state = tf.einsum('ijkl,ijk->ijl', h, probs)
          a_state = tf.squeeze(a_state, axis=1)
          cand_emb, cand_mask = span_to_avg_emb(self.candidate_spans, h, config.batch_size, self)
          cand_emb = tf.squeeze(cand_emb, axis=1)
          cand_dim = config.hidden_size * 2
          with tf.variable_scope('output_unit'):
            num_cand = tf.shape(cand_emb)[1]
            similarity = tf.einsum('ik,ijk->ijk', a_state, cand_emb)
            M = tf.tile(tf.expand_dims(a_state, axis=1), [1, num_cand, 1])
            W1 = tf.get_variable('W1', [3*cand_dim, 2*cand_dim])
            b1 = tf.get_variable('b1', [2*cand_dim])
            W2 = tf.get_variable('W2', [2*cand_dim, cand_dim])
            b2 = tf.get_variable('b2', [cand_dim])
            concat_in = tf.concat(axis=-1, values=[tf.reshape(M, [-1, cand_dim]), tf.reshape(cand_emb, [-1, cand_dim]), tf.reshape(similarity, [-1, cand_dim])])
            output = tf.matmul(tf.nn.relu(tf.matmul(concat_in, W1) + b1), W2) + b2
            g1 = tf.expand_dims(tf.reshape(output, [self.config.batch_size, -1, 40]), axis=1)

          logits = get_logits([g1], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob, mask=cand_mask, \
            is_train=self.is_train, func=config.answer_func, scope='logits1')
          JX = tf.shape(g1)[2]
          flat_logits = tf.reshape(logits, [config.batch_size, JX])
          flat_yp = tf.nn.softmax(flat_logits)  # [-1, M*JX]
          yp = tf.reshape(flat_yp, [config.batch_size, 1, JX])
          self.logits = flat_logits
          self.yp = yp
          return

        logits = get_logits([g1, p0], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                mask=x_mask, is_train=self.is_train, func=config.answer_func, scope='logits1')
        
        if config.split_supports is True:
          M=1
          JX=config.max_para_size
          N=config.batch_size
        a1i = softsel(tf.reshape(g1, [N, M * JX, 2 * d]), tf.reshape(logits, [N, M * JX]))
        a1i = tf.tile(tf.expand_dims(tf.expand_dims(a1i, 1), 1), [1, M, JX, 1])

      if config.reasoning_layer is None or config.mac_prediction == 'span-single':
        if config.cudnn_rnn:
          with tf.variable_scope('g2'):
            g2_in = tf.squeeze(tf.concat(axis=3, values=[p0, g1, a1i, g1 * a1i]), axis=1)
            g2, _ = bi_cudnn_rnn_encoder('lstm', config.hidden_size, 1, 1-config.input_keep_prob, g2_in, tf.squeeze(x_len, axis=1), self.is_train)
            g2 = tf.expand_dims(g2, axis=1)
        else:
          (fw_g2, bw_g2), _ = bidirectional_dynamic_rnn(d_cell4_fw, d_cell4_bw, tf.concat(axis=3, values=[p0, g1, a1i, g1 * a1i]),
                                                        x_len, dtype='float', scope='g2')  # [N, M, JX, 2d]
        g2 = tf.concat(axis=3, values=[fw_g2, bw_g2])
        logits2 = get_logits([g2, p0], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                   mask=x_mask,
                   is_train=self.is_train, func=config.answer_func, scope='logits2')

      flat_logits = tf.reshape(logits, [-1, M * JX])
      flat_yp = tf.nn.softmax(flat_logits)  # [-1, M*JX]
      flat_logits2 = tf.reshape(logits2, [-1, M * JX])
      flat_yp2 = tf.nn.softmax(flat_logits2)

      if config.na:
        na_bias = tf.get_variable("na_bias", shape=[], dtype='float')
        na_bias_tiled = tf.tile(tf.reshape(na_bias, [1, 1]), [N, 1])  # [N, 1]
        concat_flat_logits = tf.concat(axis=1, values=[na_bias_tiled, flat_logits])
        concat_flat_yp = tf.nn.softmax(concat_flat_logits)
        na_prob = tf.squeeze(tf.slice(concat_flat_yp, [0, 0], [-1, 1]), [1])
        flat_yp = tf.slice(concat_flat_yp, [0, 1], [-1, -1])

        concat_flat_logits2 = tf.concat(axis=1, values=[na_bias_tiled, flat_logits2])
        concat_flat_yp2 = tf.nn.softmax(concat_flat_logits2)
        na_prob2 = tf.squeeze(tf.slice(concat_flat_yp2, [0, 0], [-1, 1]), [1])  # [N]
        flat_yp2 = tf.slice(concat_flat_yp2, [0, 1], [-1, -1])

        self.concat_logits = concat_flat_logits
        self.concat_logits2 = concat_flat_logits2
        self.na_prob = na_prob * na_prob2

      yp = tf.reshape(flat_yp, [-1, M, JX])
      yp2 = tf.reshape(flat_yp2, [-1, M, JX])
      wyp = tf.nn.sigmoid(logits2)

      self.logits = flat_logits
      self.logits2 = flat_logits2
      self.yp = yp
      self.yp2 = yp2
      self.wyp = wyp


  def _build_loss(self):
    config = self.config
    JX = tf.shape(self.x)[2]
    #
    N = config.batch_size
    if config.split_supports is True:
      M = 1
      JX = config.max_para_size
    else:
      M = tf.shape(self.x)[1]

    JQ = tf.shape(self.q)[1]

    loss_mask = tf.reduce_max(tf.cast(self.q_mask, 'float'), 1)
    if config.wy:
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=tf.reshape(self.logits2, [-1, M, JX]), labels=tf.cast(self.wy, 'float'))  # [N, M, JX]
      num_pos = tf.reduce_sum(tf.cast(self.wy, 'float'))
      num_neg = tf.reduce_sum(tf.cast(self.x_mask, 'float')) - num_pos
      damp_ratio = num_pos / num_neg
      dampened_losses = losses * (
        (tf.cast(self.x_mask, 'float') - tf.cast(self.wy, 'float')) * damp_ratio + tf.cast(self.wy, 'float'))
      new_losses = tf.reduce_sum(dampened_losses, [1, 2])
      ce_loss = tf.reduce_mean(loss_mask * new_losses)
      tf.add_to_collection('losses', ce_loss)
    else:
      if config.reasoning_layer is not None and config.mac_prediction == 'candidates':
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.logits, labels=tf.cast(tf.reshape(self.candidate_span_y, [config.batch_size]), 'int32'))  
        ce_loss = tf.reduce_mean(loss_mask * losses)
        tf.add_to_collection('losses', ce_loss) 
      else:
        if config.na:
          na = tf.reshape(self.na, [-1, 1])
          concat_y = tf.concat(axis=1, values=[na, tf.reshape(self.y, [-1, M * JX])])
          losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.concat_logits, labels=tf.cast(concat_y, 'float'))
          concat_y2 = tf.concat(axis=1, values=[na, tf.reshape(self.y2, [-1, M * JX])])
          losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.concat_logits2, labels=tf.cast(concat_y2, 'float'))
        else:
          losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=tf.cast(tf.reshape(self.y, [-1, M * JX]), 'float'))
          losses2 = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits2, labels=tf.cast(tf.reshape(self.y2, [-1, M * JX]), 'float'))
        ce_loss = tf.reduce_mean(loss_mask * losses)
        ce_loss2 = tf.reduce_mean(loss_mask * losses2)
        tf.add_to_collection('losses', ce_loss)
        tf.add_to_collection("losses", ce_loss2)

    self.loss = tf.add_n(tf.get_collection('losses', scope=self.scope), name='loss')
    self.ansProp_loss = tf.add_n(tf.get_collection('losses', scope=self.scope), name='ansProp_loss')
    self.docExpl_ansProp_loss = self.ansProp_loss

    tf.summary.scalar(self.loss.op.name, self.loss)
    tf.add_to_collection('ema/scalar', self.loss)

    if config.supervise_first_doc:
      doc_first_attn_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.doc_attn_logits[0], labels=self.first_doc_ids)
      doc_first_attn_loss = tf.reduce_mean(doc_first_attn_loss, name='doc_first_attn_loss')
      tf.summary.scalar('doc_first_attn_loss', doc_first_attn_loss)
      tf.add_to_collection('ema/scalar', doc_first_attn_loss)
      self.loss = self.loss + config.first_attn_loss_coeff * doc_first_attn_loss
      self.docExpl_loss = config.first_attn_loss_coeff * doc_first_attn_loss
    else:
      self.docExpl_loss = 0.

    if config.supervise_final_doc:
      answer_doc_ids = tf.squeeze(tf.slice(self.answer_doc_ids, [0, 0], [-1, 1]), axis=1)
      answer_word_ids = tf.squeeze(tf.slice(self.answer_word_ids, [0, 0], [-1, 1]), axis=1)

      if config.mac_read_strategy=='one_doc_per_it_and_repeat_2nd_step':
        doc_attn_logits = self.doc_attn_logits[1]

        if config.mac_memory_state_update_rule is None:
          batch_nums = tf.range(0, limit=N)
          doc_indices = tf.stack([batch_nums, answer_doc_ids], axis=1)
          word_attn_logits = tf.gather_nd(self.word_attn_logits[1], doc_indices)
        else:
          word_attn_logits = self.word_attn_logits[1]
      else:
        doc_attn_logits = self.doc_attn_logits[-1]
        if config.mac_memory_state_update_rule is None:
          batch_nums = tf.range(0, limit=N)
          doc_indices = tf.stack([batch_nums, answer_doc_ids], axis=1)
          word_attn_logits = tf.gather_nd(self.word_attn_logits[-1], doc_indices)
        else:
          word_attn_logits = self.word_attn_logits[-1]

      doc_final_attn_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=doc_attn_logits, labels=answer_doc_ids)
      
      word_attn_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=word_attn_logits, labels=answer_word_ids)
      
      doc_final_attn_loss = tf.reduce_mean(doc_final_attn_loss, name='doc_final_attn_loss')
      word_attn_loss = tf.reduce_mean(word_attn_loss, name='word_attn_loss')
      
      tf.summary.scalar('doc_final_attn_loss', doc_final_attn_loss)
      tf.summary.scalar('word_attn_loss', word_attn_loss)
      
      tf.add_to_collection('ema/scalar', word_attn_loss)
      tf.add_to_collection('ema/scalar', doc_final_attn_loss)

      self.docExpl_loss += config.attn_loss_coeff * (doc_final_attn_loss + word_attn_loss)
      self.loss = self.loss + config.attn_loss_coeff * doc_final_attn_loss + config.attn_loss_coeff * word_attn_loss
      self.docExpl_ansProp_loss += self.docExpl_loss
      tf.summary.scalar('total_loss', self.loss)

    if config.use_assembler:
      assembler_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.assembler.logits, labels=tf.cast(tf.reshape(self.candidate_span_y, [config.batch_size]), 'int32'))  
      self.assembler_loss = tf.reduce_mean(loss_mask * assembler_losses, name='assembler_loss')
      self.loss += config.assembler_loss_coeff * self.assembler_loss
      tf.summary.scalar('assembler_loss', self.assembler_loss)
      tf.add_to_collection('ema/scalar', self.assembler_loss)


  def _build_ema(self):
    self.ema = tf.train.ExponentialMovingAverage(self.config.decay)
    ema = self.ema
    tensors = tf.get_collection("ema/scalar", scope=self.scope) + tf.get_collection("ema/vector", scope=self.scope)
    ema_op = ema.apply(tensors)
    for var in tf.get_collection("ema/scalar", scope=self.scope):
      ema_var = ema.average(var)
      tf.summary.scalar(ema_var.op.name, ema_var)
    for var in tf.get_collection("ema/vector", scope=self.scope):
      ema_var = ema.average(var)
      tf.summary.histogram(ema_var.op.name, ema_var)
   
    with tf.control_dependencies([ema_op]):
      self.loss = tf.identity(self.loss)


  def _build_var_ema(self):
    self.var_ema = tf.train.ExponentialMovingAverage(self.config.var_decay)
    ema = self.var_ema
    ema_op = ema.apply(tf.trainable_variables())        
    with tf.control_dependencies([ema_op]):
      self.loss = tf.identity(self.loss)

  
  def get_loss(self):
    return self.loss


  def get_global_step(self):
    return self.global_step


  def get_var_list(self, model_name):
    if model_name == 'expl+prop':
      self.var_list = [var for var in tf.trainable_variables() if 'assembler' not in var.name]
    elif model_name == 'expl+prop_only':
      self.var_list = [var for var in tf.trainable_variables() if 'MACRnn' in var.name or 'main/logits1' in var.name]
    elif model_name == 'assembler':
      self.var_list = [var for var in tf.trainable_variables() if 'MACRnn' not in var.name \
      and 'main/logits1' not in var.name]
    elif model_name == 'assembler_only':
      self.var_list = [var for var in tf.trainable_variables() if 'assembler' in var.name]
    elif model_name == 'model_network' or model_name == 'all':
      self.var_list = [var for var in tf.trainable_variables()]
    else:
      raise NotImplementedError
    assert len(self.var_list) > 0
    return self.var_list


  def get_feed_dict(self, batch, is_train, supervised=True):
    return _get_feed_dict(self, batch, is_train, supervised)


def bi_attention(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
  with tf.variable_scope(scope or "bi_attention"):
    JX = tf.shape(h)[2]
    M = tf.shape(h)[1]
    JQ = tf.shape(u)[1]
    h_aug = tf.tile(tf.expand_dims(h, 3), [1, 1, 1, JQ, 1])
    u_aug = tf.tile(tf.expand_dims(tf.expand_dims(u, 1), 1), [1, M, JX, 1, 1])
    if h_mask is None:
      hu_mask = None
    else:
      h_mask_aug = tf.tile(tf.expand_dims(h_mask, 3), [1, 1, 1, JQ])
      u_mask_aug = tf.tile(tf.expand_dims(tf.expand_dims(u_mask, 1), 1), [1, M, JX, 1])
      hu_mask = h_mask_aug & u_mask_aug

    u_logits = get_logits([h_aug, u_aug], None, True, wd=config.wd, mask=hu_mask,
                is_train=is_train, func=config.logit_func, scope='u_logits')  # [N, M, JX, JQ]
    u_a = softsel(u_aug, u_logits)  # [N, M, JX, d]
    h_a = softsel(h, tf.reduce_max(u_logits, 3))  # [N, M, d]
    h_a = tf.tile(tf.expand_dims(h_a, 2), [1, 1, JX, 1])

    if tensor_dict is not None:
      a_u = tf.nn.softmax(u_logits)  # [N, M, JX, JQ]
      a_h = tf.nn.softmax(tf.reduce_max(u_logits, 3))
      tensor_dict['a_u'] = a_u
      tensor_dict['a_h'] = a_h
      variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
      for var in variables:
        tensor_dict[var.name] = var

    return u_a, h_a


def attention_layer(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
  with tf.variable_scope(scope or "attention_layer"):
    JX = tf.shape(h)[2]
    M = tf.shape(h)[1]
    JQ = tf.shape(u)[1]
    if config.q2c_att or config.c2q_att:
      u_a, h_a = bi_attention(config, is_train, h, u, h_mask=h_mask, u_mask=u_mask, tensor_dict=tensor_dict)
    if not config.c2q_att:
      u_a = tf.tile(tf.expand_dims(tf.expand_dims(tf.reduce_mean(u, 1), 1), 1), [1, M, JX, 1])
    if config.q2c_att:
      p0 = tf.concat(axis=3, values=[h, u_a, h * u_a, h * h_a])
    else:
      p0 = tf.concat(axis=3, values=[h, u_a, h * u_a])
    return p0


def zhong_selfatt(U, dim, mask=None, seq_len=None, transform=None, scope=None, reuse=None):
  if mask is None:
    assert seq_len is not None
    mask = tf.expand_dims(tf.sequence_mask(seq_len, tf.shape(U)[1]), axis=1)

  with tf.variable_scope(scope or 'zhong_selfAttention', reuse=reuse):
    W1 = tf.get_variable("W1", [dim, dim])
    b1 = tf.get_variable("b1", [dim,])
    W2 = tf.get_variable("W2", [dim, 1])
    b2 = tf.get_variable("b2", [1,])
    layer1_output = tf.nn.tanh(tf.einsum('ijkl,lt->ijkt', U, W1) + b1)
    logits = tf.nn.tanh(tf.squeeze(tf.einsum('ijkl,lt->ijkt', layer1_output, W2) + b2, axis=-1))
    masked_logits = logits * tf.cast(mask, dtype='float')
    att = tf.nn.softmax(masked_logits)
    output = tf.einsum("ijkl,ijk->ijl", U, att)
    if transform == 'expand':
      output = tf.expand_dims(output, axis=1)
    elif transform == 'squeeze':
      output = tf.squeeze(output, axis=1)
  return output
