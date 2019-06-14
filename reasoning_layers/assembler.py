import tensorflow as tf
import numpy as np
import re

from my.tensorflow.nn import get_logits, linear_logits, softsel
from my.tensorflow.rnn import bidirectional_dynamic_rnn
from tensorflow.contrib.rnn import BasicLSTMCell
from my.tensorflow.rnn_cell import SwitchableDropoutWrapper
from qangaroo.utils import get_word_span
from reasoning_layers.utils import biattention_layer


class Assembler(object):
  def __init__(self, config, is_train, model, context_dim=40):
    self.config = config
    self.is_train = is_train
    self.model = model
    self.context_dim = context_dim

  def build_forward(self, **kwargs):
    return None

  def get_var_list(self):
    var_list = [var for var in tf.trainable_variables() if 'assembler' in var.name]
    assert len(var_list) > 0
    return var_list


class BiAttnAssembler(Assembler):
  def __init__(self, config, is_train, model, context_dim=40):
    super(BiAttnAssembler, self).__init__(config, is_train, model, context_dim)

  def build_forward(self, k, c_mask, query, query_st, q_mask, cand_emb, cand_mask, drop_one_doc=False, dropped_hops=None, reuse=False):
    self.selected_sent_ids = self.model.selected_sent_ids
    cand_emb = tf.squeeze(cand_emb, axis=1)
    config = self.config
    model = self.model
    context_dim = self.context_dim
    x_acc_sents_len = model.x_sents_len_reconstruct
    self.x_acc_sents_len = x_acc_sents_len
    zero_prepend_x_acc_sents_len = tf.concat([tf.expand_dims(tf.zeros_like(x_acc_sents_len[:, :, 0]), axis=2), x_acc_sents_len], axis=2)
    with tf.variable_scope('assembler', reuse=reuse):
      
      if config.assembler_biattn_w_first_doc:
        assert config.assembler_merge_query_st, ("If biattn with first doc, then has to merge query information in another way.")
        
        for hop in range(config.num_hops - 1):
          doc_indices = model.mac_rnn_cell.doc_indices_history[hop + 1]
          doc2 = tf.gather_nd(k, doc_indices)
          doc2_mask = tf.cast(tf.gather_nd(tf.cast(c_mask, 'float32'), doc_indices), 'bool')
          #doc2_first_sent_len = tf.gather_nd(x_sents_len[:, :, 0], doc_indices)
          selected_sent_indices = tf.concat([doc_indices, tf.expand_dims(self.selected_sent_ids[:, hop + 1], axis=1)], axis=-1)
          self.selected_sent_indices = selected_sent_indices
          doc2_selected_sent_end = tf.gather_nd(x_acc_sents_len, selected_sent_indices)
          doc2_selected_sent_start = tf.gather_nd(zero_prepend_x_acc_sents_len, selected_sent_indices)
          self.doc2_selected_sent_start = doc2_selected_sent_start
          self.doc2_selected_sent_end = doc2_selected_sent_end
          self.doc_lens = []
          self.prev_doc_lens = []
          self.new_doc_lens = []
          self.padding_lens = []
          if hop == 0:
            self.concat_selected_doc = []
            for i in range(config.batch_size):
              new_doc = tf.slice(doc2[i], [doc2_selected_sent_start[i], 0], [doc2_selected_sent_end[i] - doc2_selected_sent_start[i], -1])
              self.concat_selected_doc.append(new_doc)
            self.concat_selected_doc_len = doc2_selected_sent_end - doc2_selected_sent_start
          else:
            for i in range(config.batch_size):
              prev_doc = tf.slice(self.concat_selected_doc[i], [0, 0], [self.concat_selected_doc_len[i], -1])
              new_doc = tf.slice(doc2[i], [doc2_selected_sent_start[i], 0], [doc2_selected_sent_end[i] - doc2_selected_sent_start[i], -1])
              padding_len = tf.reduce_max(self.concat_selected_doc_len + doc2_selected_sent_end - doc2_selected_sent_start) - self.concat_selected_doc_len[i] - (doc2_selected_sent_end[i] - doc2_selected_sent_start[i])
              padding = tf.zeros([padding_len, context_dim])
              self.concat_selected_doc[i] = tf.concat([prev_doc, new_doc, padding], axis=0)
              self.doc_lens.append(tf.shape(tf.concat([prev_doc, new_doc, padding], axis=0))[0])
              self.prev_doc_lens.append(tf.shape(prev_doc)[0])
              self.new_doc_lens.append(tf.shape(new_doc)[0])
              self.padding_lens.append(padding_len)
            self.concat_selected_doc_len += (doc2_selected_sent_end - doc2_selected_sent_start)
        
        max_len = tf.reduce_max(self.concat_selected_doc_len)
        self.concat_selected_doc_mask = []
        for i in range(config.batch_size):
          concat_selected_doc_mask_i = tf.concat(values=[tf.ones([self.concat_selected_doc_len[i]]), tf.zeros([max_len-self.concat_selected_doc_len[i]])], axis=0)
          self.concat_selected_doc_mask.append(concat_selected_doc_mask_i)
        
        self.concat_selected_doc = tf.stack(self.concat_selected_doc, axis=0)
        self.concat_selected_doc_mask = tf.cast(tf.stack(self.concat_selected_doc_mask, axis=0), 'bool')

        first_doc_indices = model.mac_rnn_cell.doc_indices_history[0]
        doc1 = tf.gather_nd(k, first_doc_indices)
        doc1_mask = tf.cast(tf.gather_nd(tf.cast(c_mask, 'float32'), first_doc_indices), 'bool')
        p0 = biattention_layer(self.is_train, self.concat_selected_doc, doc1, h_mask=self.concat_selected_doc_mask, u_mask=doc1_mask)
      else:
        if drop_one_doc:
          num_docs_to_concat = config.num_hops - 1 
        else:
          num_docs_to_concat = (config.num_hops+2 if config.mac_read_strategy=='one_doc_per_it_and_repeat_for_3_hops' else (config.num_hops - 1)*2) if config.assembler_repeat_first_doc else config.num_hops
        for hop in range(num_docs_to_concat):
          if config.assembler_repeat_first_doc:
            if drop_one_doc:
              raise NotImplementedError
            if config.mac_read_strategy=='one_doc_per_it_and_repeat_for_3_hops':
              doc_indices = model.mac_rnn_cell.doc_indices_history[0] if (hop%4==1 or hop==0) else model.mac_rnn_cell.doc_indices_history[hop - int(hop/5)]
            else:
              doc_indices = model.mac_rnn_cell.doc_indices_history[0] if hop%2 == 0 else model.mac_rnn_cell.doc_indices_history[int(hop/2) + 1]
          else:
            if drop_one_doc:
              doc_indices = tf.squeeze(tf.slice(tf.stack(model.mac_rnn_cell.doc_indices_history), [dropped_hops[hop], 0, 0], [1, -1, -1]), axis=0)
            else:
              doc_indices = model.mac_rnn_cell.doc_indices_history[hop]
          doc2 = tf.gather_nd(k, doc_indices)
          doc2_mask = tf.cast(tf.gather_nd(tf.cast(c_mask, 'float32'), doc_indices), 'bool')
          #doc2_first_sent_len = tf.gather_nd(x_sents_len[:, :, 0], doc_indices)
          if config.assembler_repeat_first_doc:
            if config.mac_read_strategy=='one_doc_per_it_and_repeat_for_3_hops':
              if hop%4==1 or hop==0:
                selected_sent_indices = tf.concat([doc_indices, tf.expand_dims(self.selected_sent_ids[:, 0], axis=1)], axis=-1)
              else:
                selected_sent_indices = tf.concat([doc_indices, tf.expand_dims(self.selected_sent_ids[:, hop - int(hop/5)], axis=1)], axis=-1)
            else:
              if hop%2 == 0:
                print("concat first doc")
                selected_sent_indices = tf.concat([doc_indices, tf.expand_dims(self.selected_sent_ids[:, 0], axis=1)], axis=-1)
              else:
                print("concat second doc")
                selected_sent_indices = tf.concat([doc_indices, tf.expand_dims(self.selected_sent_ids[:, int(hop/2) + 1], axis=1)], axis=-1)
          else:
            selected_sent_indices = tf.concat([doc_indices, tf.expand_dims(self.selected_sent_ids[:, hop], axis=1)], axis=-1)
          self.selected_sent_indices = selected_sent_indices
          doc2_selected_sent_end = tf.gather_nd(x_acc_sents_len, selected_sent_indices)
          doc2_selected_sent_start = tf.gather_nd(zero_prepend_x_acc_sents_len, selected_sent_indices)
          self.doc2_selected_sent_start = doc2_selected_sent_start
          self.doc2_selected_sent_end = doc2_selected_sent_end
          
          if hop == 0:
            self.doc_lens = []
            self.prev_doc_lens = []
            self.new_doc_lens = []
            self.padding_lens = []
            self.concat_selected_doc = []
            for i in range(config.batch_size):
              new_doc = tf.slice(doc2[i], [doc2_selected_sent_start[i], 0], [doc2_selected_sent_end[i] - doc2_selected_sent_start[i], -1])
              self.concat_selected_doc.append(new_doc)
            self.concat_selected_doc_len = doc2_selected_sent_end - doc2_selected_sent_start
          else:
            for i in range(config.batch_size):
              prev_doc = tf.slice(self.concat_selected_doc[i], [0, 0], [self.concat_selected_doc_len[i], -1])
              new_doc = tf.slice(doc2[i], [doc2_selected_sent_start[i], 0], [doc2_selected_sent_end[i] - doc2_selected_sent_start[i], -1])
              padding_len = tf.reduce_max(self.concat_selected_doc_len + doc2_selected_sent_end - doc2_selected_sent_start) - self.concat_selected_doc_len[i] - (doc2_selected_sent_end[i] - doc2_selected_sent_start[i])
              padding = tf.zeros([padding_len, context_dim])
              self.concat_selected_doc[i] = tf.concat([prev_doc, new_doc, padding], axis=0)
              self.doc_lens.append(tf.shape(tf.concat([prev_doc, new_doc, padding], axis=0))[0])
              self.prev_doc_lens.append(tf.shape(prev_doc)[0])
              self.new_doc_lens.append(tf.shape(new_doc)[0])
              self.padding_lens.append(padding_len)
            self.concat_selected_doc_len += (doc2_selected_sent_end - doc2_selected_sent_start)
        
        max_len = tf.reduce_max(self.concat_selected_doc_len)
        self.concat_selected_doc_mask = []
        for i in range(config.batch_size):
          concat_selected_doc_mask_i = tf.concat(values=[tf.ones([self.concat_selected_doc_len[i]]), tf.zeros([max_len-self.concat_selected_doc_len[i]])], axis=0)
          self.concat_selected_doc_mask.append(concat_selected_doc_mask_i)
        
        self.concat_selected_doc = tf.stack(self.concat_selected_doc, axis=0)
        self.concat_selected_doc_mask = tf.cast(tf.stack(self.concat_selected_doc_mask, axis=0), 'bool')
        p0 = biattention_layer(self.is_train, self.concat_selected_doc, query, h_mask=self.concat_selected_doc_mask, u_mask=q_mask)
        
      p0 = tf.squeeze(p0, axis=1)
      
      if config.assembler_bidaf_layer > 1:
        with tf.variable_scope("layer_1"): 
          cell_fw = BasicLSTMCell(40, state_is_tuple=True)
          cell_bw = BasicLSTMCell(40, state_is_tuple=True)
          cell_fw = SwitchableDropoutWrapper(cell_fw, self.is_train, input_keep_prob=config.input_keep_prob)
          cell_bw = SwitchableDropoutWrapper(cell_bw, self.is_train, input_keep_prob=config.input_keep_prob)
        
          (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, p0, self.concat_selected_doc_len, dtype='float') 
          x = tf.concat(axis=2, values=[fw_h, bw_h])

        with tf.variable_scope("layer_2"):
          cell2_fw = BasicLSTMCell(40, state_is_tuple=True)
          cell2_bw = BasicLSTMCell(40, state_is_tuple=True)
          cell2_fw = SwitchableDropoutWrapper(cell2_fw, self.is_train, input_keep_prob=config.input_keep_prob)
          cell2_bw = SwitchableDropoutWrapper(cell2_bw, self.is_train, input_keep_prob=config.input_keep_prob)
        
          (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell2_fw, cell2_bw, x, self.concat_selected_doc_len, dtype='float') 
          x = tf.concat(axis=2, values=[fw_h, bw_h])
      else:
        cell_fw = BasicLSTMCell(40, state_is_tuple=True)
        cell_bw = BasicLSTMCell(40, state_is_tuple=True)
        cell_fw = SwitchableDropoutWrapper(cell_fw, self.is_train, input_keep_prob=config.input_keep_prob)
        cell_bw = SwitchableDropoutWrapper(cell_bw, self.is_train, input_keep_prob=config.input_keep_prob)
      
        (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, p0, self.concat_selected_doc_len, dtype='float') 
        x = tf.concat(axis=2, values=[fw_h, bw_h])

      logits = linear_logits([x], True, input_keep_prob=config.input_keep_prob, mask=self.concat_selected_doc_mask, is_train=self.is_train, scope='a_state_logits')
      probs = tf.nn.softmax(logits)
      new_ans = tf.einsum('ijk,ij->ik', self.concat_selected_doc, probs)
      if config.assembler_merge_query_st:
        W_c = tf.get_variable('W_c', [40, 40])
        b_c = tf.get_variable('b_c', [40])
        c_proj = tf.matmul(query_st, W_c) + b_c

        W1 = tf.get_variable('W1', [3*40, 2*40])
        b1 = tf.get_variable('b1', [2*40])
        W2 = tf.get_variable('W2', [2*40, 40])
        b2 = tf.get_variable('b2', [40])
        concat_in = tf.concat(axis=-1, values=[new_ans, c_proj, new_ans*c_proj])
        a_state = tf.matmul(tf.nn.relu(tf.matmul(concat_in, W1) + b1), W2) + b2
      else:
        a_state = new_ans
      g1 = tf.expand_dims(self.output_unit(cand_emb, a_state), axis=1)
      self.g1 = g1
      self.cand_mask = cand_mask
      self.cand_emb = cand_emb
      logits = linear_logits([g1], True, is_train=self.is_train, input_keep_prob=config.input_keep_prob, mask=cand_mask, scope='g1_logits')
      
      JX = tf.shape(g1)[2]
      self.g1_shape=tf.shape(g1)
      flat_logits = tf.reshape(logits, [config.batch_size, JX])
      flat_yp = tf.nn.softmax(flat_logits)  # [-1, M*JX]
      yp = tf.reshape(flat_yp, [config.batch_size, 1, JX])
      self.logits = flat_logits
      self.yp = yp

 
  def output_unit(self, candidates, a_state, reuse=False):
    with tf.variable_scope('output_unit', reuse=reuse):
      context_dim = self.context_dim
      cand_dim = context_dim
      #cand_dim = candidates.get_shape()[-1]
      num_cand = tf.shape(candidates)[1]
      similarity = tf.einsum('ik,ijk->ijk', a_state, candidates)
      M = tf.tile(tf.expand_dims(a_state, axis=1), [1, num_cand, 1])
      W1 = tf.get_variable('W1', [3*cand_dim, 2*cand_dim])
      b1 = tf.get_variable('b1', [2*cand_dim])
      W2 = tf.get_variable('W2', [2*cand_dim, cand_dim])
      b2 = tf.get_variable('b2', [cand_dim])
      concat_in = tf.concat(axis=-1, values=[tf.reshape(M, [-1, cand_dim]), tf.reshape(candidates, [-1, cand_dim]), tf.reshape(similarity, [-1, cand_dim])])
      output = tf.matmul(tf.nn.relu(tf.matmul(concat_in, W1) + b1), W2) + b2
      g1 = tf.reshape(output, [self.config.batch_size, -1, context_dim])
      return g1


  def get_sentence_ids(self, sess, cand_word, x, feed_dict, handle, model_id=0):
    config = self.config
    ensemble_yps = []
    qsub_topk_ids, qsub_topk_probs, qsub_all_probs, yp, yp_list, doc_lst, sents_len = sess.partial_run(handle, [self.model.mac_rnn_cell.qsub_topk_ids, self.model.mac_rnn_cell.qsub_topk_probs, \
      self.model.mac_rnn_cell.qsub_all_probs, self.model.yp, self.model.yp_list, self.model.mac_rnn_cell.doc_attn, self.model.x_sents_len_reconstruct], feed_dict=feed_dict if model_id==0 else None)
    for i in range(config.num_hops):
      #yp = sess.run(self.yp_list[i+1], feed_dict=feed_dict)
      ensemble_yps.append(yp_list[i])
    ensemble_yps = np.array(ensemble_yps)
    answer_qsub_probs = np.zeros([config.batch_size, config.num_hops])
    sentence_ids = np.zeros([config.batch_size, config.num_hops])
    answer_cand_ids = []
    for i in range(config.batch_size):
      answer_cand_id = []
      answer_cand_ids.append(answer_cand_id)
      for j in range(len(ensemble_yps)):
        if j == 0:
          continue
        my_answer_id = np.argmax(ensemble_yps[j,i], axis=-1)[0]
        answer_cand_id.append(my_answer_id)
        my_answer_phrase = cand_word[i][my_answer_id]
        doc_selected = x[i][doc_lst[j][i][0]]

        if my_answer_phrase.lower() not in ' '.join(doc_selected).lower():
          continue
        else:
          answer_spans = []
          answer_word_ids = []
          next_answer_start = 0
          next_answer_stop = 0
          context = ' '.join(x[i][doc_lst[j][i][0]])
          xi = x[i][doc_lst[j][i][0]]
          next_context = context[next_answer_stop:]
          while True:
            next_answer_start, next_answer_stop = compute_answer_span(next_context, my_answer_phrase)
            next_context = next_context[next_answer_stop:]
            if next_answer_start is not None:
              if len(answer_spans) > 0:
                answer_spans.append((next_answer_start + answer_spans[-1][1], next_answer_stop + answer_spans[-1][1]))
              else:
                answer_spans.append((next_answer_start, next_answer_stop))
            else:
              break

            if len(answer_spans) > 1:
              next_yi0, next_yi1 = get_word_span(context, xi, next_answer_start + answer_spans[-2][1], next_answer_stop + answer_spans[-2][1])
            else:
              next_yi0, next_yi1 = get_word_span(context, xi, next_answer_start, next_answer_stop)
            answer_word_ids += list(range(next_yi0[0], next_yi1[0] + 1))
          
          answer_word_qsub_probs = [qsub_all_probs[j][i][answer_word_id] for answer_word_id in answer_word_ids]
          answer_qsub_probs[i,j] = np.amax(answer_word_qsub_probs)
          top_answer_word_id = answer_word_ids[np.argmax(answer_word_qsub_probs)]
          sentence_id = 0
          for kid, k in enumerate(sents_len[i, doc_lst[j][i][0]]):
            if k > top_answer_word_id:
              if kid == 0:
                sentence_id = kid
                break
              elif sents_len[i, doc_lst[j][i][0]][kid-1] <= top_answer_word_id:
                sentence_id = kid
                break
          sentence_ids[i, j] = sentence_id


    return sentence_ids, handle, answer_cand_ids, doc_lst


def compute_answer_span(context, answer):

  answer = answer.replace(' – ',' ').lower()
  context = context.lower()
  try:
    a = re.search(r'({})'.format(answer), context)
  except:
    print(answer)
    print(context)
    return None, None
  if a is None:
    return None, None
  start = a.start()
  end = start + len(answer)
  return start, end
