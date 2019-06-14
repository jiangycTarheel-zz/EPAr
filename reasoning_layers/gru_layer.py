import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from my.tensorflow.nn import linear_logits

def dynamic_gru_rnn(cell, context, query, c_len, q_len, c_mask, q_mask, candidates=None, cand_mask=None):
  out = cell.apply(context, query, c_len, q_len, c_mask, q_mask, candidates=candidates, cand_mask=cand_mask)
  return out


class GRURnn(object):
  """
  This class implements a GRU RNN (https://arxiv.org/abs/1803.03067) adapted for multi-hop qa.

  """
  def __init__(self, batch_size, context_dim, query_dim, hidden_dim=80, length=6, \
    bidirectional_input_unit=False, reuse_cell=True, is_train=None):
    """
    length: the number of mac cell chained together, or number of reasoning steps.
    bidriectional_input_unit: use bi-lstm for input unit. Default to false to save memory.
    prediction: prediction layer. Could be 'span-single/dual', 'candidates'
    reuse_cell: use one single cell for all reasoning steps. (not sure what Hudson and Mannning did.)
    """    
    self.batch_size = batch_size
    self.hidden_dim = hidden_dim
    self.context_dim = context_dim
    self.query_dim = query_dim
    self.length = length
    self.bidirectional_input_unit = bidirectional_input_unit
    #self.prediction = prediction
    self.reuse_cell = reuse_cell
    self.is_train = is_train

  def apply(self, context, query, c_len, q_len, c_mask, q_mask, candidates=None, cand_mask=None):
    batch_size = self.batch_size
    hidden_dim = self.hidden_dim
    reuse_cell = self.reuse_cell
    context = tf.squeeze(context, axis=1)
    if candidates is not None:
      candidates = tf.squeeze(candidates, axis=1)
    s_state = tf.zeros((batch_size, hidden_dim))
    #m_state = tf.zeros((batch_size, hidden_dim))

    with tf.variable_scope('GRURnn'):
      query, q_rep = self.query_emb_unit(query, q_len)
      gru_cell = tf.contrib.rnn.GRUCell(hidden_dim)

      for i in range(self.length):
        if reuse_cell:    
          scope_str = 'GRURnn-layer-%d' % 0
          gru_input = self.GRUInputUnit(i, query, context, c_mask, q_mask, s_state, scope_str, reuse=(i!=0))
          output, _ = gru_cell(gru_input, s_state)
          s_state = output
        else:
          raise NotImplementedError
          # scope_str = 'GRURnn-layer-%d' % i
          # s_state = self.GRUCell(i, query, context, s_state, c_mask, q_mask, scope_str, reuse=False)

      g1 = self.GRUOutputUnit(s_state, candidates)
      return tf.expand_dims(g1, axis=1)

  def query_emb_unit(self, query, query_len):
    """
    Inputs: encodede query and length.
    Outputs: query encoded by another lstm, and the final state of this lstm as 
             a fixed-size representation of this query.
    """
    with tf.variable_scope('input_unit', initializer=tf.random_uniform_initializer):
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

  def GRUOutputUnit(self, s_state, candidates):
    hidden_dim = self.hidden_dim
    context_dim = self.context_dim
    with tf.variable_scope('output_unit'):
      W_m = tf.get_variable('W_m', [hidden_dim, hidden_dim])
      b_m = tf.get_variable('b_m', [hidden_dim])
      M = tf.matmul(s_state, W_m) + b_m

      W_k = tf.get_variable('W_k', [context_dim, hidden_dim])
      b_k = tf.get_variable('b_k', [hidden_dim])
      I = tf.einsum('ijk,kl->ijl', candidates, W_k) + b_k
      g1 = tf.einsum('ik,ijk->ijk', M, I)
      return g1

  def GRUInputUnit(self, layer: int, cw, k, c_mask, q_mask, s_state, scope_str, reuse=False):
    hidden_dim = self.hidden_dim
    context_dim = self.context_dim

    def q_attn_read(s, q, q_mask):
      with tf.variable_scope('q_attn_read'):
        A_q = tf.get_variable('A_q', [hidden_dim, hidden_dim])
        a_q = tf.get_variable('a_q', [hidden_dim])
        _s = tf.matmul(s, A_q) + a_q
        q_it = tf.einsum('ijk,ik->ij', q, _s)
        q_it = tf.nn.softmax(q_it)
        q_t = tf.einsum('ijk,ij->ik', q, q_it)
        return q_t

    def c_attn_read(s, c, c_mask, q_t):
      with tf.variable_scope('c_attn_read'):
        A_c = tf.get_variable('A_c', [hidden_dim*2, context_dim])
        a_c = tf.get_variable('a_c', [context_dim])
        _s = tf.matmul(tf.concat([s, q_t], axis=-1), A_c) + a_c
        c_it = tf.einsum('ijk,ik->ij', c, _s)
        c_it = tf.nn.softmax(c_it)
        c_t = tf.einsum('ijk,ij->ik', c, c_it)
        
        # Cast the context embedding to the size of hidden_dim.
        W_c = tf.get_variable('W_c', [context_dim, hidden_dim])
        b_c = tf.get_variable('b_c', [hidden_dim])
        c_t = tf.matmul(c_t, W_c) + b_c
        return c_t

    def gate(x, y, s, reuse):
      with tf.variable_scope('gate', reuse=reuse):
        W1 = tf.get_variable('W1', [hidden_dim*4, hidden_dim*2])
        b1 = tf.get_variable('b1', [hidden_dim*2])
        gate_in = tf.concat([s, x, y, x*y], axis=-1)
        _x = tf.nn.relu(tf.matmul(gate_in, W1) + b1)
        W2 = tf.get_variable('W2', [hidden_dim*2, hidden_dim])
        b2 = tf.get_variable('b2', [hidden_dim])
        out = tf.matmul(_x, W2) + b2
        return out

    with tf.variable_scope('GRU_input', reuse=reuse):
      q_t = q_attn_read(s_state, cw, q_mask)
      c_t = c_attn_read(s_state, k, c_mask, q_t)
      r_q = gate(q_t, c_t, s_state, reuse=reuse)
      r_c = gate(c_t, q_t, s_state, reuse=True)
      gru_input = tf.concat([r_q*q_t, r_c*c_t], axis=-1)

      return gru_input
