import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from my.tensorflow.nn import linear_logits, get_logits, softsel
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from my.tensorflow.rnn import bidirectional_dynamic_rnn
from my.tensorflow.rnn_cell import SwitchableDropoutWrapper, AttentionCell
from tensorflow.contrib.rnn import BasicLSTMCell
from reasoning_layers.utils import biattention_layer

def dynamic_mac_rnn(cell, context, query, q_len, c_mask, q_mask, q_sub_st=None, context_st=None, query_st=None, cdoc_mask=None, candidates=None, cand_mask=None, greedy_read=False):
  if cdoc_mask is None:
    assert context_st is None 
    return cell.apply(context, query, q_len, c_mask, q_mask, q_sub_st=q_sub_st, candidates=candidates, cand_mask=cand_mask)
  else:
    assert context_st is not None and q_sub_st is not None
    assert isinstance(cell, HierarchicalAttnMACRnn)
    return cell.apply(context, context_st, query, query_st, q_sub_st, q_len, c_mask, cdoc_mask, q_mask, candidates=candidates, cand_mask=cand_mask, greedy_read=greedy_read)


class MACRnn(object):
  """
  This class implements a standard MAC RNN (https://arxiv.org/abs/1803.03067) adapted for multi-hop qa.

  """
  def __init__(self, batch_size, context_dim, query_dim, hidden_dim=80, num_hops=6, bidirectional_input_unit=False, prediction='span-single', \
    reuse_cell=True, is_train=None, use_control_unit=True, mode="train", output_unit_type='similarity', reasoning_unit='answer_unit', \
    answer_state_update_rule='mlp'):
    """
    num_hops: the number of mac cell chained together, or number of reasoning steps.
    bidriectional_input_unit: use bi-lstm for input unit. Default to false to save memory.
    prediction: prediction layer. Could be 'span-single/dual', 'candidates'
    reuse_cell: use one single cell for all reasoning steps. (not sure what Hudson and Mannning did.)
    """    
    self.batch_size = batch_size
    self.hidden_dim = hidden_dim
    self.context_dim = context_dim
    self.query_dim = query_dim
    self.num_hops = num_hops
    self.bidirectional_input_unit = bidirectional_input_unit
    self.prediction = prediction
    self.reuse_cell = reuse_cell
    self.is_train = is_train
    self.use_control_unit = use_control_unit
    self.mode = mode
    self.output_unit_type = output_unit_type
    self.reasoning_unit = reasoning_unit
    self.answer_state_update_rule = answer_state_update_rule
    self.top_attn = []

  def apply(self, context, query, q_len, c_mask, q_mask, candidates=None, cand_mask=None, q_sub_st=None):
    batch_size = self.batch_size
    hidden_dim = self.hidden_dim
    query_dim = self.query_dim
    reuse_cell = self.reuse_cell
    context = tf.squeeze(context, axis=1)
    if candidates is not None:
      candidates = tf.squeeze(candidates, axis=1)
    c_state = tf.zeros((batch_size, hidden_dim))
    m_state = tf.zeros((batch_size, hidden_dim))

    with tf.variable_scope('MACRnn'):
      query, q_rep = self.MACInputUnit(query, q_len)
      
      c_history = []
      m_history = []

      for i in range(self.num_hops):
        if reuse_cell:
          scope_str = 'MACRnn-layer-%d' % 0
          c_state, m_state = self.MACCell(i, query, q_rep, context, c_mask, q_mask, c_history, m_history, \
            c_state, m_state, scope_str, reuse=(i!=0))          
        else:
          scope_str = 'MACRnn-layer-%d' % i
          c_state, m_state = self.MACCell(i, query, q_rep, context, c_mask, q_mask, c_history, m_history, \
            c_state, m_state, scope_str, reuse=False)
        
        c_history.append(c_state)
        m_history.append(m_state)

      if self.prediction == 'candidates':
        g1 = self.MACOutputUnit(m_state, context, candidates)
        return tf.expand_dims(g1, axis=1)
      elif self.prediction == 'span-dual':
        g1, g2 = self.MACOutputUnit(m_state, context)
        return tf.expand_dims(g1, axis=1), tf.expand_dims(g2, axis=1)
      else:
        assert self.prediction == 'span-single'
        g1, logits = self.MACOutputUnit(m_state, context)
        return tf.expand_dims(g1, axis=1), logits

  def MACInputUnit(self, query, query_len, reuse=False):
    """
    Inputs: encodede query and length.
    Outputs: query encoded by another lstm, and the final state of this lstm as 
             a fixed-size representation of this query.
    """
    with tf.variable_scope('input_unit', initializer=tf.random_uniform_initializer, reuse=reuse):
      hidden_dim = self.hidden_dim
      
      if self.bidirectional_input_unit is True:
        cell_fw = tf.contrib.rnn.LSTMCell(hidden_dim, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(hidden_dim, state_is_tuple=True)
        (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, query, \
          dtype=tf.float32, sequence_length=query_len, swap_memory=True)
        query_embed = tf.concat(axis=2, values=encoder_outputs)
        query_rep = tf.concat([fw_st.c, bw_st.c], axis=1)
        W_emb = tf.get_variable('W_emb', [2*hidden_dim, hidden_dim])
        b_emb = tf.get_variable('b_emb', [hidden_dim])
        W_rep = tf.get_variable('W_rep', [2*hidden_dim, hidden_dim])
        b_rep = tf.get_variable('b_rep', [hidden_dim])
        query_embed = tf.einsum('ijk,kl->ijl', query_embed, W_emb) + b_emb
        query_rep = tf.matmul(query_rep, W_rep) + b_rep
      else:
        cell_fw = tf.contrib.rnn.LSTMCell(hidden_dim, state_is_tuple=True)
        query_embed, final_st = tf.nn.dynamic_rnn(cell_fw, query, dtype=tf.float32, \
          sequence_length=query_len)
        query_rep = final_st.c
      return query_embed, query_rep

  def MACCell(self, layer: int, cw, q, k, c_mask, q_mask, c_history, m_history, c_state, m_state, scope_str, reuse=False):
    hidden_dim = self.hidden_dim
    context_dim = self.context_dim
    query_dim = self.query_dim
    def control_unit():
      with tf.variable_scope('control_unit'):
        W_cq = tf.get_variable('W_cq', [2*hidden_dim, hidden_dim])
        b_cq = tf.get_variable('b_cq', [hidden_dim])
        cq = tf.matmul(tf.concat([c_state, q], axis=1), W_cq) + b_cq

        W_ca = tf.get_variable('W_ca', [hidden_dim, 1])
        b_ca = tf.get_variable('b_ca', [1])
        ca = tf.squeeze(tf.einsum('ijk,kl->ijl', tf.einsum('ik,ijk->ijk', cq, cw), W_ca), axis=2) + b_ca
        cv = tf.nn.softmax(ca)
      return tf.einsum('ijk,ij->ik', cw, cv)

    def read_unit(new_c_state):
      """
      Does not include the I' in the original MAC paper.
      """
      with tf.variable_scope('read_unit'):
        W_m = tf.get_variable('W_m', [hidden_dim, hidden_dim])
        b_m = tf.get_variable('b_m', [hidden_dim])
        W_k = tf.get_variable('W_k', [context_dim, hidden_dim])
        b_k = tf.get_variable('b_k', [hidden_dim])
        I = tf.einsum('il,ijl->ijl', tf.matmul(m_state, W_m) + b_m, tf.einsum('ijk,kl->ijl', k, W_k) + b_k) # [batch_size, context_len, hidden_dim]

        
        W_ra = tf.get_variable('W_ra', [hidden_dim, 1])
        b_ra = tf.get_variable('b_ra', [1])
        ra = tf.squeeze(tf.einsum('ijk,kl->ijl', tf.einsum('ik,ijk->ijk', new_c_state, I), W_ra), axis=2) + b_ra
        rv = tf.nn.softmax(ra)
      return tf.einsum('ijk,ij->ik', k, rv)

    def write_unit(r, new_c_state):
      with tf.variable_scope('write_unit'):
        W_m = tf.get_variable('W_m', [context_dim + hidden_dim, hidden_dim])
        b_m = tf.get_variable('b_m', [hidden_dim])
        m_prev = tf.matmul(tf.concat([r, m_state], axis=1), W_m) + b_m

        if layer > 0 or self.reuse_cell:
          W_c = tf.get_variable('W_c', [hidden_dim, 1])
          b_c = tf.get_variable('b_c', [1])
          #sa = tf.nn.softmax(tf.squeeze(tf.einsum('ijk,kl->ijl', tf.multiply(new_c_state, c_history), W_c), axis=2))
          W_s = tf.get_variable('W_s', [hidden_dim, hidden_dim])

        W_p = tf.get_variable('W_p', [hidden_dim, hidden_dim])
        b = tf.get_variable('b', [hidden_dim])
        
        if layer > 0:
          sa = tf.nn.softmax(tf.squeeze(tf.einsum('ijk,kl->ijl', tf.einsum('ik,ijk->ijk', new_c_state, c_history), W_c) + b_c, axis=2))       
          m_sa = tf.einsum('ijk,ij->ik', m_history, sa)
          m_prime = tf.matmul(m_sa, W_s) + tf.matmul(m_prev, W_p) + b
        else:
          m_prime = tf.matmul(m_prev, W_p) + b

        W_c_2 = tf.get_variable('W_c_2', [hidden_dim, 1])
        b_c_2 = tf.get_variable('b_c_2', [1])
        c_prime = tf.matmul(new_c_state, W_c_2) + b_c_2
        
        return tf.nn.sigmoid(c_prime) * m_state + (1 - tf.nn.sigmoid(c_prime)) * m_prime
    
    if layer > 0:
      c_history = tf.stack(c_history, axis=1)
      m_history = tf.stack(m_history, axis=1)
    
    with tf.variable_scope(scope_str, reuse=reuse) as scope:
      new_c_state = control_unit()
      new_m_state = write_unit(read_unit(new_c_state), new_c_state)
      
    return new_c_state, new_m_state

  def MACOutputUnit(self, m_state, context, candidates=None, query=None, reuse=False):
    hidden_dim = self.hidden_dim
    context_dim = self.context_dim
    with tf.variable_scope('output_unit', reuse=reuse):
      if self.prediction == 'candidates':
        assert candidates is not None
        cand_dim = context_dim
        #cand_dim = candidates.get_shape()[-1]
        if self.output_unit_type == 'similarity':
          W_m = tf.get_variable('W_m', [hidden_dim, hidden_dim])
          b_m = tf.get_variable('b_m', [hidden_dim])
          M = tf.matmul(m_state, W_m) + b_m

          W_k = tf.get_variable('W_k', [cand_dim, hidden_dim])
          b_k = tf.get_variable('b_k', [hidden_dim])
          I = tf.einsum('ijk,kl->ijl', candidates, W_k) + b_k

          g1 = tf.einsum('ik,ijk->ijk', M, I)
        
        elif self.output_unit_type == 'nested-triplet-mlp':
          num_cand = tf.shape(candidates)[1]
          if self.reasoning_unit == 'bi-attn' or self.reasoning_unit == 'attention-lstm' or self.reasoning_unit == 'concat_first_sent' or self.reasoning_unit == 'concat_full_doc':
            similarity = tf.einsum('ik,ijk->ijk', m_state, candidates)
            M = tf.tile(tf.expand_dims(m_state, axis=1), [1, num_cand, 1])
            W1 = tf.get_variable('W1', [3*cand_dim, 2*cand_dim])
            b1 = tf.get_variable('b1', [2*cand_dim])
            W2 = tf.get_variable('W2', [2*cand_dim, cand_dim])
            b2 = tf.get_variable('b2', [cand_dim])
            concat_in = tf.concat(axis=-1, values=[tf.reshape(M, [-1, cand_dim]), tf.reshape(candidates, [-1, cand_dim]), tf.reshape(similarity, [-1, cand_dim])])
            output = tf.matmul(tf.nn.relu(tf.matmul(concat_in, W1) + b1), W2) + b2
          else:
            W_k = tf.get_variable('W_k', [cand_dim, hidden_dim])
            b_k = tf.get_variable('b_k', [hidden_dim])
            similarity = tf.einsum('ik,ijk->ijk', m_state, tf.einsum('ijk,kl->ijl', candidates, W_k)) + b_k
            M = tf.tile(tf.expand_dims(m_state, axis=1), [1, num_cand, 1])
            W1 = tf.get_variable('W1', [2*hidden_dim + cand_dim, hidden_dim])
            b1 = tf.get_variable('b1', [hidden_dim])
            W2 = tf.get_variable('W2', [hidden_dim, 40])
            b2 = tf.get_variable('b2', [40])
            concat_in = tf.concat(axis=-1, values=[tf.reshape(M, [-1, hidden_dim]), tf.reshape(candidates, [-1, cand_dim]), tf.reshape(similarity, [-1, hidden_dim])])
            output = tf.matmul(tf.nn.relu(tf.matmul(concat_in, W1) + b1), W2) + b2
          g1 = tf.reshape(output, [self.batch_size, -1, context_dim])

        elif self.output_unit_type == 'triplet-mlp':
          assert query is not None
          assert self.reasoning_unit == 'None' or self.reasoning_unit is None
          num_cand = tf.shape(candidates)[1]
          query_dim = self.query_dim
          W_q = tf.get_variable('W_q', [query_dim, hidden_dim])
          b_q = tf.get_variable('b_q', [hidden_dim])
          query = tf.matmul(query, W_q) + b_q
          query = tf.tile(tf.expand_dims(query, axis=1), [1, num_cand, 1])

          W_k = tf.get_variable('W_k', [cand_dim, hidden_dim])
          b_k = tf.get_variable('b_k', [hidden_dim])
          similarity = tf.einsum('ik,ijk->ijk', m_state, tf.einsum('ijk,kl->ijl', candidates, W_k)) + b_k
          M = tf.tile(tf.expand_dims(m_state, axis=1), [1, num_cand, 1])
          W1 = tf.get_variable('W1', [3*hidden_dim + cand_dim, hidden_dim])
          b1 = tf.get_variable('b1', [hidden_dim])
          W2 = tf.get_variable('W2', [hidden_dim, 40])
          b2 = tf.get_variable('b2', [40])
          concat_in = tf.concat(axis=-1, values=[tf.reshape(query, [-1, hidden_dim]), tf.reshape(M, [-1, hidden_dim]), tf.reshape(candidates, [-1, cand_dim]), tf.reshape(similarity, [-1, hidden_dim])])
          output = tf.matmul(tf.nn.relu(tf.matmul(concat_in, W1) + b1), W2) + b2
          g1 = tf.reshape(output, [self.batch_size, -1, 40])
        else:
          raise NotImplementedError

        return g1
      else:
        W_m = tf.get_variable('W_m', [hidden_dim, hidden_dim])
        b_m = tf.get_variable('b_m', [hidden_dim])
        W_k = tf.get_variable('W_k', [context_dim, hidden_dim])
        b_k = tf.get_variable('b_k', [hidden_dim])
        I = tf.einsum('ijk,kl->ijl', context, W_k) + b_k
        M = tf.matmul(m_state, W_m) + b_m
        
        g1 = tf.einsum('ik,ijk->ijk', M, I)

        if self.prediction == 'span-dual':
          p2 = tf.concat([I, g1], axis=2)
          W_p = tf.get_variable('W_p', [2*hidden_dim, hidden_dim])
          b_p = tf.get_variable('b_p', [hidden_dim])
          I_prime = tf.einsum('ijk,kl->ijl', p2, W_p) + b_p
          g2 = tf.einsum('ik,ijk->ijk', M, I_prime)
          
          return g1, g2

        else:
          W_ra = tf.get_variable('W_ra', [hidden_dim, 1])
          b_ra = tf.get_variable('b_ra', [1])
          ra = tf.squeeze(tf.einsum('ijk,kl->ijl', g1, W_ra), axis=2) + b_ra
          return g1, ra  
      

class HierarchicalAttnMACRnn(MACRnn):
  def __init__(self, batch_size, context_dim, query_dim, hidden_dim=80, num_hops=6, bidirectional_input_unit=False, prediction='candidates', input_keep_prob=0.8, reuse_cell=True, \
    is_train=None, use_control_unit=True, mode="train", read_strategy='full', output_unit_type='similarity', reasoning_unit='answer_unit', \
    memory_state_update_rule=None, answer_state_update_rule='mlp', attention_style='similarity', \
    answer_doc_ids=None, sents_len=None, oracle=None, reinforce=False, attention_cell_dropout=False, \
    read_topk_docs=0):
    """
    num_hops: the number of mac cell chained together, or number of reasoning steps.
    bidriectional_input_unit: use bi-lstm for input unit. Default to false to save memory.
    prediction: prediction layer. Could be 'span-single/dual', 'candidates'
    reuse_cell: use one single cell for all reasoning steps. (not sure what Hudson and Mannning did.)
    """    
    assert prediction == "candidates"
    assert reuse_cell == True
    super(HierarchicalAttnMACRnn, self).__init__(batch_size, context_dim, query_dim, hidden_dim, num_hops, \
      bidirectional_input_unit, prediction, reuse_cell, is_train, use_control_unit, mode, output_unit_type, \
      reasoning_unit, answer_state_update_rule)
    self.input_keep_prob = input_keep_prob
    self.top_doc_attn = []
    self.top_attn_prob = []
    self.doc_attn = []
    self.read_strategy = read_strategy
    self.rv_doc_history = []
    self.doc_indices_history = []
    self.attention_style = attention_style
    self.memory_state_update_rule = memory_state_update_rule
    self.oracle = oracle
    if self.oracle is not None:
      assert answer_doc_ids is not None
      self.answer_doc_ids = answer_doc_ids
    self.sents_len = sents_len
    self.answer_list = []
    self._c_state = tf.placeholder('float', [batch_size, query_dim], name='_c_state')
    self._m_state = tf.placeholder('float', [batch_size, hidden_dim], name='_m_state')
    self._a_state = tf.placeholder('float', [batch_size, hidden_dim], name='_a_state')
    self._c_history = tf.placeholder('float', [batch_size, None, query_dim], name='_c_history')
    self._m_history = tf.placeholder('float', [batch_size, None, hidden_dim], name='_m_history')
    self.reinforce = reinforce
    self.attention_cell_dropout = attention_cell_dropout
    self.read_topk_docs = read_topk_docs


  def apply(self, context, context_st, query, query_st, q_sub_st, q_len, c_mask, cdoc_mask, q_mask, candidates, cand_mask, greedy_read=False, reuse=False):
    batch_size = self.batch_size
    hidden_dim = self.hidden_dim
    query_dim = self.query_dim
    self.docs_len = tf.reduce_sum(tf.cast(c_mask, 'int32'), 2)
    candidates = tf.squeeze(candidates, axis=1)
    c_state = tf.zeros((batch_size, query_dim))
    m_state = tf.zeros((batch_size, hidden_dim))
    a_state = tf.zeros((batch_size, hidden_dim))

    with tf.variable_scope('MACRnn'):
      
      with tf.variable_scope('q_sub_proj'):
        W = tf.get_variable('W', [query_dim, hidden_dim])
        b = tf.get_variable('b', [hidden_dim])
        m_state = tf.matmul(q_sub_st, W) + b

      self.c_history = []
      self.m_history = []
      self.a_history = []
      self.doc_attn_logits_lst = []
      self.word_attn_logits_lst = []
      self.doc_attn_weights_lst = []

      cell = tf.contrib.rnn.GRUCell(hidden_dim)
      self.cell = cell

      for i in range(self.num_hops):
        scope_str = 'MACRnn-layer-%d' % 0

        if self.read_strategy == 'one_doc_per_it_and_repeat_2nd_step' and i > 1:
          m_state = self.m_history[0]
          a_state = self.a_history[0]

        c_state, m_state, a_state, doc_attn_logits, doc_attn_weights, word_attn_logits = self.HierarchicalAttnMACCell(i, cell, query, query_st, q_sub_st, context, context_st, c_mask, cdoc_mask, \
          q_mask, self.c_history, self.m_history, c_state, m_state, a_state, scope_str, reuse=(reuse or i!=0), greedy_read=greedy_read)

        self.doc_attn_logits_lst.append(doc_attn_logits)
        self.word_attn_logits_lst.append(word_attn_logits)
        self.doc_attn_weights_lst.append(doc_attn_weights)
        self.c_history.append(c_state)
        self.m_history.append(m_state)
       

        if (self.reasoning_unit == 'concat_first_sent' or self.reasoning_unit == 'concat_full_doc') and i == self.num_hops - 1:
          with tf.variable_scope("concat_read_lstm", reuse=False):
            max_len = tf.reduce_max(self.concat_selected_doc_len)
            self.concat_selected_doc_mask = []
            for k in range(self.batch_size):
              concat_selected_doc_mask_k = tf.concat(values=[tf.ones([self.concat_selected_doc_len[k]]), tf.zeros([max_len-self.concat_selected_doc_len[k]])], axis=0)
              self.concat_selected_doc_mask.append(concat_selected_doc_mask_k)
            
            self.concat_selected_doc = tf.stack(self.concat_selected_doc, axis=0)
            self.concat_selected_doc_mask = tf.cast(tf.stack(self.concat_selected_doc_mask, axis=0), 'bool')
            p0 = biattention_layer(self.is_train, self.concat_selected_doc, query, h_mask=self.concat_selected_doc_mask, u_mask=q_mask)
            p0 = tf.squeeze(p0, axis=1)
            cell_fw = BasicLSTMCell(40, state_is_tuple=True)
            cell_bw = BasicLSTMCell(40, state_is_tuple=True)
            cell_fw = SwitchableDropoutWrapper(cell_fw, self.is_train, input_keep_prob=self.input_keep_prob)
            cell_bw = SwitchableDropoutWrapper(cell_bw, self.is_train, input_keep_prob=self.input_keep_prob)
            
            (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell_fw, cell_bw, p0, self.concat_selected_doc_len, dtype='float') 
            x = tf.concat(axis=2, values=[fw_h, bw_h])
            logits = linear_logits([x], True, input_keep_prob=self.input_keep_prob, mask=self.concat_selected_doc_mask, is_train=self.is_train, scope='logits1')
            probs = tf.nn.softmax(logits)
            doc_rep = tf.einsum('ijk,ij->ik', self.concat_selected_doc, probs)
            a_state = doc_rep

        self.a_history.append(a_state)

      if self.oracle == 'extra':
        scope_str = 'MACRnn-layer-%d' % 0
        if self.read_strategy == 'one_doc_per_it_and_repeat_2nd_step' and i > 1:
          m_state = self.m_history[0]
          a_state = self.a_history[0]
        _, _, a_state, _, _, _ = self.HierarchicalAttnMACCell(self.num_hops, cell, query, query_st, q_sub_st, context, context_st, c_mask, cdoc_mask, \
          q_mask, self.c_history, self.m_history, c_state, m_state, a_state, scope_str, reuse=True, greedy_read=greedy_read)
        
      if self.prediction == 'candidates':        
        if self.output_unit_type == 'triplet-mlp':
          g1 = self.MACOutputUnit(a_state, context, candidates, query=query)
          if (self.reasoning_unit != 'concat_first_sent' and self.reasoning_unit != 'concat_full_doc') and (self.reasoning_unit != 'attention-lstm' or self.read_strategy != 'one_doc_per_it'):
            for i in range(self.num_hops):
              gi = self.MACOutputUnit(self.a_history[i], context, candidates, query=query)
              self.answer_list.append(tf.expand_dims(gi, axis=1))
        else:
          g1 = self.MACOutputUnit(a_state, context, candidates)
          if (self.reasoning_unit != 'concat_first_sent' and self.reasoning_unit != 'concat_full_doc') and (self.reasoning_unit != 'attention-lstm' or self.read_strategy != 'one_doc_per_it'):
            for i in range(self.num_hops):
              gi = self.MACOutputUnit(self.a_history[i], context, candidates, reuse=True)
              self.answer_list.append(tf.expand_dims(gi, axis=1))

        return tf.expand_dims(g1, axis=1)
      else:
        raise NotImplementedError

  def initialize_state(self, q_sub):
    with tf.variable_scope('initial_m'):
      W = tf.get_variable('W', [self.hidden_dim*2, self.hidden_dim])
      b = tf.get_variable('b', [self.hidden_dim])
      new_state = tf.matmul(q_sub, W) + b
    return new_state


  def HierarchicalAttnMACCell(self, layer: int, cell, cw, cw_st, q_sub_st, k, k_st, c_mask, cdoc_mask, q_mask, c_history, m_history, c_state, m_state, a_state, scope_str, \
    reuse=False, out_of_graph=False, greedy_read=False):
    """
    The 2nd implementation based on MAC Cell with hierarchical attention. 
    The read unit does not depend on c_state any more.
    Added a_state.
    Input: k [N, M, JX, context_dim]
    """
    hidden_dim = self.hidden_dim
    context_dim = self.context_dim
    query_dim = self.query_dim
    

    def control_unit():
      with tf.variable_scope('control_unit'):
        W_cq = tf.get_variable('W_cq', [query_dim + hidden_dim, query_dim])
        b_cq = tf.get_variable('b_cq', [query_dim])
        cq = tf.matmul(tf.concat([c_state, m_state], axis=1), W_cq) + b_cq
        pre_ca = tf.einsum('ik,ijk->ijk', cq, cw)
        ca = linear_logits([pre_ca], True, input_keep_prob=self.input_keep_prob, is_train=self.is_train, mask=q_mask)
        cv = tf.nn.softmax(ca)
      return tf.einsum('ijk,ij->ik', cw, cv)

    def read_unit(m_state):
      with tf.variable_scope('read_unit'):
        W_cm = tf.get_variable('W_cm', [hidden_dim, hidden_dim])
        b_cm = tf.get_variable('b_cm', [hidden_dim])
        cm_state = tf.matmul(m_state, W_cm) + b_cm

        if layer > 1 and self.read_strategy == 'one_doc_per_it_and_repeat_2nd_step':
          ra_doc = self.doc_attn_logits_lst[1]
          rv_doc = self.doc_attn_weights_lst[1]
        else:
          W_k2 = tf.get_variable('W_k2', [query_dim, hidden_dim])
          b_k2 = tf.get_variable('b_k2', [hidden_dim])
          I_doc = tf.einsum('ijk,kl->ijl', k_st, W_k2) + b_k2  # [N, M, hidden_dim]
          pre_ra_doc = tf.einsum('ik,ijk->ijk', cm_state, I_doc)

          if self.attention_style == 'Bahdanau':
            W_b2 = tf.get_variable('W_b2', [hidden_dim, hidden_dim])
            b_b2 = tf.get_variable('b_b2', [hidden_dim])
            shape_1 = tf.shape(I_doc)[1]
            tiled_cm_state = tf.tile(tf.expand_dims(cm_state, axis=1), [1, shape_1, 1])
            concat_in = tf.reshape(tiled_cm_state, [-1, hidden_dim]) + tf.reshape(I_doc, [-1, hidden_dim]) + tf.reshape(pre_ra_doc, [-1, hidden_dim])
            pre_ra_doc = tf.matmul(concat_in, W_b2) + b_b2
            pre_ra_doc = tf.reshape(pre_ra_doc, [-1, shape_1, hidden_dim])

          ra_doc = linear_logits([pre_ra_doc], True, is_train=self.is_train, input_keep_prob=self.input_keep_prob, mask=cdoc_mask, scope='logits2')
          rv_doc = tf.nn.softmax(ra_doc)  # document-level attention weight

        # Word-level attention
        if self.memory_state_update_rule is None:
          W_k = tf.get_variable('W_k', [context_dim, hidden_dim])
          b_k = tf.get_variable('b_k', [hidden_dim])
          I_word = tf.einsum('ijkl,lm->ijkm', k, W_k) + b_k
          pre_ra_word = tf.einsum('il,ijkl->ijkl', cm_state, I_word)

          if self.attention_style == 'Bahdanau':
            W_b = tf.get_variable('W_b', [hidden_dim, hidden_dim])
            b_b = tf.get_variable('b_b', [hidden_dim])
            shape_1 = tf.shape(I_word)[1]
            shape_2 = tf.shape(I_word)[2]
            tiled_cm_state = tf.tile(tf.expand_dims(tf.expand_dims(cm_state, axis=1), axis=1), [1, shape_1, shape_2, 1])
            concat_in = tf.reshape(tiled_cm_state, [-1, hidden_dim]) + tf.reshape(I_word, [-1, hidden_dim]) + tf.reshape(pre_ra_word, [-1, hidden_dim])
            pre_ra_word = tf.matmul(concat_in, W_b) + b_b
            pre_ra_word = tf.reshape(pre_ra_word, [-1, shape_1, shape_2, hidden_dim])

          ra_word = linear_logits([pre_ra_word], True, is_train=self.is_train, input_keep_prob=self.input_keep_prob, mask=c_mask, scope='logits1')
          rv_word = tf.nn.softmax(ra_word)  # word-level attention weight
          r_doc = tf.einsum('ijkl,ijk->ijl', k, rv_word) # [N, M, context_dim]

        doc_indices = None
        if self.read_strategy == 'one_doc_per_it' or self.read_strategy == 'one_doc_per_it_and_mask_all_read' or self.read_strategy == 'one_doc_per_it_and_mask_read_pairs' \
        or self.read_strategy == 'one_doc_per_it_and_repeat_2nd_step':
          if out_of_graph or layer > 0:
            if self.read_strategy == 'one_doc_per_it_and_mask_read_pairs':
              prev_read = self.doc_attn[layer-1]

              doc_idx = tf.expand_dims(tf.stack(self.doc_attn, axis=1), axis=2)
              shape = tf.shape(rv_doc)
              updates = tf.negative(tf.ones([self.batch_size, layer]))
              batch_nums = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(0, limit=self.batch_size), axis=1), axis=1), [1, layer, 1])
              indices = tf.concat([batch_nums, doc_idx], axis=2) # [batch_size, layer, 2]
            elif self.read_strategy == 'one_doc_per_it':
              if out_of_graph:
                doc_idx = tf.stack(self.doc_attn, axis=1)[:, layer-1]
              else:
                doc_idx = self.doc_attn[layer-1]
              shape = tf.shape(rv_doc)
              updates = tf.negative(tf.ones([self.batch_size]))
              batch_nums = tf.expand_dims(tf.range(0, limit=self.batch_size), axis=1)
              indices = tf.concat([batch_nums, tf.reshape(doc_idx, [self.batch_size, 1])], axis=1)
            elif self.read_strategy == 'one_doc_per_it_and_mask_all_read' or self.read_strategy == 'one_doc_per_it_and_repeat_2nd_step':
              #if self.mode == 'train':
              doc_idx = tf.stack(self.doc_attn, axis=1)
              # else:
              #   doc_idx = tf.expand_dims(tf.stack(self.doc_attn, axis=1), axis=2)
              shape = tf.shape(rv_doc)
              updates = tf.negative(tf.ones([self.batch_size, layer]))
              batch_nums = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(0, limit=self.batch_size), axis=1), axis=1), [1, layer, 1])
              indices = tf.concat([batch_nums, doc_idx], axis=2) # [batch_size, layer, 2]
              updates_2 = tf.ones([self.batch_size, layer]) * 1e-30
              very_small_number = tf.scatter_nd(indices, updates_2, shape)
            
            mask = tf.scatter_nd(indices, updates, shape)
            mask = mask + 1
            rv_doc = rv_doc * mask

            if self.read_strategy == 'one_doc_per_it_and_mask_all_read' or self.read_strategy == 'one_doc_per_it_and_repeat_2nd_step':
              rv_doc = rv_doc + very_small_number
          
          if self.mode == 'test':
            if self.oracle == 'final' and layer == self.num_hops - 1:
              new_doc_idx = tf.slice(self.answer_doc_ids, [0, 0], [-1, 1])
            else:
              new_doc_idx = tf.expand_dims(tf.argmax(tf.log(rv_doc), axis=1), axis=-1)
          elif self.mode == 'train':
            if (self.oracle == 'final' and layer == self.num_hops - 1) or (self.oracle == 'extra' and layer == self.num_hops):
              new_doc_idx = tf.slice(self.answer_doc_ids, [0, 0], [-1, 1])
            else:
              if self.read_topk_docs > 0:
                topk_doc_mask_1 = tf.ones([self.batch_size, tf.minimum(tf.shape(rv_doc)[1], self.read_topk_docs)])
                topk_doc_mask_0 = tf.zeros([self.batch_size, tf.maximum(tf.shape(rv_doc)[1]-self.read_topk_docs, 0)])
                topk_doc_mask = tf.concat([topk_doc_mask_1, topk_doc_mask_0], axis=1)
                rv_doc = rv_doc * topk_doc_mask

              if (greedy_read or self.read_strategy == 'one_doc_per_it_and_repeat_2nd_step') and \
              self.reinforce is False:
                new_doc_idx = tf.expand_dims(tf.argmax(tf.log(rv_doc), axis=1), axis=-1)
              else:
                new_doc_idx = tf.multinomial(tf.log(rv_doc), 1)
                #new_doc_idx = tf.argmax(tf.log(rv_doc), axis=1)
          else:
            raise NotImplementedError
          new_doc_idx = tf.cast(new_doc_idx, 'int32')

          shape = tf.shape(rv_doc)
          updates = tf.ones([self.batch_size])
          batch_nums = tf.expand_dims(tf.range(0, limit=self.batch_size), axis=1)
          doc_indices = tf.concat([batch_nums, tf.cast(tf.reshape(new_doc_idx, [self.batch_size, 1]), 'int32')], axis=1)
          
          if self.memory_state_update_rule == 'bi-attn':
            selected_doc = tf.gather_nd(k, indices)
            selected_mask = tf.gather_nd(c_mask, indices)
            p0 = biattention_layer(self.is_train, selected_doc, cw, h_mask=selected_mask, u_mask=q_mask)
            p0 = tf.squeeze(p0, axis=1)
            W_p0 = tf.get_variable('W_p0', [hidden_dim*2, hidden_dim])
            b_p0 = tf.get_variable('b_p0', [hidden_dim])
            I_word = tf.einsum('ijk,km->ijm', p0, W_p0) + b_p0
            pre_ra_word = tf.einsum('ik,ijk->ijk', cm_state, I_word)
            
            ra_word = linear_logits([pre_ra_word], True, is_train=self.is_train, input_keep_prob=self.input_keep_prob, mask=selected_mask, scope='logits1')
            rv_word = tf.nn.softmax(ra_word)  # word-level attention weight
            r_doc = tf.einsum('ikl,ik->il', p0, rv_word) # [N, M, context_dim]
            r = r_doc # No need to apply doc_mask again.
          else:
            r = tf.gather_nd(r_doc, doc_indices)
          print('one_doc_per_it')
        elif self.read_strategy == 'mask_previous_max':
          if layer > 0:
            doc_idx = self.doc_attn[layer-1]
            shape = tf.shape(rv_doc)
            updates = tf.negative(tf.ones([self.batch_size]))
            batch_nums = tf.expand_dims(tf.range(0, limit=self.batch_size), axis=1)
            indices = tf.concat([batch_nums, tf.cast(tf.reshape(doc_idx, [self.batch_size, 1]), 'int32')], axis=1)
            mask = tf.scatter_nd(indices, updates, shape)
            mask = mask + 1
            #self.mask = mask
            rv_doc = rv_doc * mask

          new_doc_idx = tf.argmax(tf.log(rv_doc), axis=1)
          r = tf.einsum('ijk,ij->ik', r_doc, rv_doc)
        else:
          assert self.read_strategy == 'full'
          new_doc_idx = tf.argmax(tf.log(rv_doc), axis=1)
          r = tf.einsum('ijk,ij->ik', r_doc, rv_doc)
        
        if out_of_graph is False:
          self.doc_attn.append(new_doc_idx)
          self.rv_doc_history.append(rv_doc)
          self.doc_indices_history.append(doc_indices)

        _, topk_docs = tf.nn.top_k(rv_doc, 3)
        topk_words_prob, topk_words = tf.nn.top_k(rv_word[:,topk_docs[0, 0]], 20)
        if out_of_graph is False:
          self.top_doc_attn.append(topk_docs)
          self.top_attn.append(topk_words)
          self.top_attn_prob.append(topk_words_prob)
      return r, ra_doc, rv_doc, ra_word

    def write_unit(r, new_c_state, c_history, m_history, query=None):
      with tf.variable_scope('write_unit'):
        doc_indices = self.doc_indices_history[layer]
        new_m_state, output = cell(r, m_state)
        if self.reasoning_unit == 'answer_unit':
          
          W_c = tf.get_variable('W_c', [query_dim, hidden_dim])
          b_c = tf.get_variable('b_c', [hidden_dim])
          c_proj = tf.matmul(new_c_state, W_c) + b_c

          W1 = tf.get_variable('W1', [3*hidden_dim, 2*hidden_dim])
          b1 = tf.get_variable('b1', [2*hidden_dim])
          W2 = tf.get_variable('W2', [2*hidden_dim, hidden_dim])
          b2 = tf.get_variable('b2', [hidden_dim])
          concat_in = tf.concat(axis=-1, values=[output, c_proj, output*c_proj])
          new_ans = tf.matmul(tf.nn.relu(tf.matmul(concat_in, W1) + b1), W2) + b2
          
          
          if self.answer_state_update_rule == 'bi-attn':
            assert query is not None
            selected_doc = tf.einsum('ijkl,ij->ikl', k, doc_mask)
            selected_mask = tf.cast(tf.einsum('ijk,ij->ik', tf.cast(c_mask, 'float32'), doc_mask), 'bool')
            p0 = biattention_layer(self.is_train, selected_doc, query, h_mask=selected_mask, u_mask=q_mask)
            p0 = tf.squeeze(p0, axis=1)
            
            logits = linear_logits([selected_doc, p0], True, is_train=self.is_train, input_keep_prob=self.input_keep_prob, mask=selected_mask)
            weights = tf.nn.softmax(logits)
            new_ans_2 = tf.einsum('ijk,ij->ik', selected_doc, weights)
            W_a = tf.get_variable('W_a', [self.context_dim, hidden_dim])
            b_a = tf.get_variable('b_a', [hidden_dim])
            new_ans_2 = tf.matmul(new_ans_2, W_a) + b_a
            new_ans = tf.concat([new_ans, new_ans_2], axis=-1)
            W_a2 = tf.get_variable('W_a2', [hidden_dim * 2, hidden_dim])
            b_a2 = tf.get_variable('b_a2', [hidden_dim])
            new_ans = tf.matmul(new_ans, W_a2) + b_a2
          else:
            assert self.answer_state_update_rule == 'mlp'

          W_g = tf.get_variable('W_g', [hidden_dim, 1])
          b_g = tf.get_variable('b_g', [1])
          gate = tf.matmul(output*c_proj, W_g) + b_g
          new_a_state = tf.sigmoid(gate) * new_ans + (1-tf.sigmoid(gate)) * a_state
        elif self.reasoning_unit == 'mlp':
          c_proj = new_c_state

          W1 = tf.get_variable('W1', [3*query_dim, 3*query_dim])
          b1 = tf.get_variable('b1', [3*query_dim])
          W2 = tf.get_variable('W2', [3*query_dim, hidden_dim])
          b2 = tf.get_variable('b2', [hidden_dim])
          # concat_in = tf.concat(axis=-1, values=[output, c_proj, output*c_proj])
          concat_in = tf.concat(axis=-1, values=[r, c_proj, r*c_proj])
          new_a_state = tf.matmul(tf.nn.relu(tf.matmul(concat_in, W1) + b1), W2) + b2
        elif self.reasoning_unit == 'bi-attn':
          c_proj = new_c_state
          #selected_doc = tf.einsum('ijkl,ij->ikl', k, doc_mask)
          selected_doc = tf.gather_nd(k, doc_indices)
          #selected_mask = tf.cast(tf.einsum('ijk,ij->ik', tf.cast(c_mask, 'float32'), doc_mask), 'bool')
          selected_mask = tf.cast(tf.gather_nd(tf.cast(c_mask, 'float32'), doc_indices), 'bool')
          p0 = biattention_layer(self.is_train, selected_doc, query, h_mask=selected_mask, u_mask=q_mask)
          p0 = tf.squeeze(p0, axis=1)
          
          logits = linear_logits([selected_doc, p0], True, is_train=self.is_train, input_keep_prob=self.input_keep_prob, mask=selected_mask)
          weights = tf.nn.softmax(logits)
          new_a_state = tf.einsum('ijk,ij->ik', selected_doc, weights)
        elif self.reasoning_unit == 'concat_first_sent' or self.reasoning_unit == 'concat_full_doc':
          doc2 = tf.gather_nd(k, doc_indices)
          doc2_mask = tf.cast(tf.gather_nd(tf.cast(c_mask, 'float32'), doc_indices), 'bool')
          if self.reasoning_unit == 'concat_first_sent':
            doc2_first_sent_len = tf.gather_nd(self.sents_len[:, :, 0], doc_indices)
          else:
            doc2_first_sent_len = tf.gather_nd(self.docs_len, doc_indices)
          if layer == 0:
            print(doc2.get_shape())
            print(tf.reshape(tf.slice(doc2, [0, 0, 0], [-1, tf.reduce_max(doc2_first_sent_len), -1]), [self.batch_size, -1, context_dim]).get_shape())
            
            self.concat_selected_doc = tf.unstack(tf.reshape(tf.slice(doc2, [0, 0, 0], [-1, tf.reduce_max(doc2_first_sent_len), -1]), [self.batch_size, -1, context_dim]), axis=0)
            assert len(self.concat_selected_doc) == self.batch_size, (len(self.concat_selected_doc))
            self.concat_selected_doc_len = doc2_first_sent_len
          else:
            for i in range(self.batch_size):
              prev_doc = tf.slice(self.concat_selected_doc[i], [0, 0], [self.concat_selected_doc_len[i], -1])
              new_doc = tf.slice(doc2[i], [0, 0], [doc2_first_sent_len[i], -1])
              padding_len = tf.reduce_max(self.concat_selected_doc_len + doc2_first_sent_len) - self.concat_selected_doc_len[i] - doc2_first_sent_len[i]
              padding = tf.zeros([padding_len, context_dim])
              self.concat_selected_doc[i] = tf.concat([prev_doc, new_doc, padding], axis=0)
            self.concat_selected_doc_len += doc2_first_sent_len
          new_a_state = None
        elif self.reasoning_unit == 'attention-lstm':
          if layer > 0:
            if self.read_strategy == 'one_doc_per_it_and_repeat_2nd_step':
              doc1_indices = self.doc_indices_history[0]
            else:
              doc1_indices = self.doc_indices_history[layer-1]
            doc1 = tf.gather_nd(k, doc1_indices)
            doc1_mask = tf.cast(tf.gather_nd(tf.cast(c_mask, 'float32'), doc1_indices), 'bool')
          else:
            doc1 = cw
            doc1_mask = q_mask
          if self.read_strategy == 'one_doc_per_it' and (layer < self.num_hops - 1 and layer > 0):
            new_a_state = None
          else:
            doc1_len = tf.reduce_sum(tf.cast(doc1_mask, 'int32'), axis=-1)

            doc2 = tf.gather_nd(k, doc_indices)
            doc2_mask = tf.cast(tf.gather_nd(tf.cast(c_mask, 'float32'), doc_indices), 'bool')
            doc2_len = tf.reduce_sum(tf.cast(doc2_mask, 'int32'), axis=-1)

            lstm_cell = BasicLSTMCell(hidden_dim, state_is_tuple=True)

            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
              num_units=hidden_dim,
              memory=doc1,
              memory_sequence_length=doc1_len)
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(lstm_cell, attention_mechanism, output_attention=False)
  
            if self.attention_cell_dropout:
              attention_cell = tf.contrib.rnn.DropoutWrapper(attention_cell, input_keep_prob=self.input_keep_prob)
                 
            decoder_initial_state = attention_cell.zero_state(
              dtype=tf.float32, batch_size=self.batch_size)

            lstm_output, _ = tf.nn.dynamic_rnn( cell=attention_cell,
                                                inputs=doc2,
                                                sequence_length=doc2_len,
                                                initial_state=decoder_initial_state,
                                                dtype=tf.float32)            
            W_x = tf.get_variable('W_x', [hidden_dim, context_dim])
            b_x = tf.get_variable('b_x', [context_dim])
            #x = tf.reshape(tf.einsum('ijk,kl->ijl', lstm_output, W_x) + b_x, [config.batch_size, self.tree_width, -1, d])
            x = tf.einsum('ijk,kl->ijl', lstm_output, W_x) + b_x

            similarity_with_q_sub = tf.einsum('ijk,ik->ijk', x, q_sub_st)
            similarity_with_q_bod = tf.einsum('ijk,ik->ijk', x, cw_st)

            doc2_mask = tf.reshape(doc2_mask, [self.batch_size, -1])
            logits_q_sub = linear_logits([similarity_with_q_sub], True, input_keep_prob=self.input_keep_prob, mask=doc2_mask, \
              is_train=self.is_train, scope='logits1')
            logits_q_bod = linear_logits([similarity_with_q_bod], True, input_keep_prob=self.input_keep_prob, mask=doc2_mask, \
              is_train=self.is_train, scope='logits2')
            similarity_w_qsub_probs = tf.nn.softmax(logits_q_sub)
            similarity_w_qbod_probs = tf.nn.softmax(logits_q_bod)
            similarity_probs = (similarity_w_qsub_probs + similarity_w_qbod_probs) / 2
            doc_rep = tf.einsum('ijk,ij->ik', doc2, similarity_probs)
            new_a_state = doc_rep
            
            qsub_topk_probs, qsub_topk_ids = tf.nn.top_k(similarity_w_qsub_probs, 10)
            qbod_topk_probs, qbod_topk_ids = tf.nn.top_k(similarity_w_qbod_probs, 10)
            if layer > 0:
              self.qsub_topk_ids.append(qsub_topk_ids)
              self.qsub_topk_probs.append(qsub_topk_probs)
              self.qbod_topk_ids.append(qbod_topk_ids)
              self.qbod_topk_probs.append(qbod_topk_probs)
              self.qsub_all_probs.append(similarity_w_qsub_probs)
            else:
              self.qsub_topk_ids = [qsub_topk_ids]
              self.qsub_topk_probs = [qsub_topk_probs]
              self.qbod_topk_ids = [qbod_topk_ids]
              self.qbod_topk_probs = [qbod_topk_probs]
              self.qsub_all_probs = [similarity_w_qsub_probs]
        elif self.reasoning_unit == 'None' or self.reasoning_unit is None:
          new_a_state = output
        else:
          raise NotImplementedError
        return new_m_state, new_a_state
    
    if out_of_graph is False and layer > 0:
      c_history = tf.stack(c_history, axis=1)
      m_history = tf.stack(m_history, axis=1)
    
    with tf.variable_scope(scope_str, reuse=reuse) as scope:
      if self.use_control_unit:
        new_c_state = control_unit()
      else:
        new_c_state = cw_st
      
      # Read unit
      r, ra_doc, rv_doc, ra_word = read_unit(m_state)
      # Write unit
      new_m_state, new_a_state = write_unit(r, new_c_state, c_history, m_history, cw)

    return new_c_state, new_m_state, new_a_state, ra_doc, rv_doc, ra_word
