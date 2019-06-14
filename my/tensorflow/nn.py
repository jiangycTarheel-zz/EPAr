#from tensorflow.python.ops.rnn_cell_impl import _linear
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear

from tensorflow.python.util import nest
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

from my.tensorflow import flatten, reconstruct, add_wd, exp_mask

def reconstruct_batchesV3(x_new, x, cx, x_mask, group, target_batch_size, max_para_size, model):
  """
  For reconstruction of raw input (not embeddings).
  """
  out_x_new = []
  out_x = []
  out_cx = []
  out_xmask = []

  start_batch = 0
  #emb_dim = 160
  #max_doc_num = tf.shape(p0)[1]
  max_doc_num = tf.reduce_max(group)

  def outer_while_body(j, x_new, x, cs, x_mask, start_batch, out_x_newi, out_xi, out_cxi, out_xmaski, group_len):
    # with tf.control_dependencies([tf.assert_equal(tf.shape(p0)[0], tf.shape(p_mask)[0])]):
    out_x_newi = tf.cond(j < group_len, lambda: out_x_newi.write(j, x_new[start_batch+j]), lambda: out_x_newi.write(j, tf.zeros_like(x_new[0])))
    out_xi = tf.cond(j < group_len, lambda: out_xi.write(j, x[start_batch+j, 0]), lambda: out_xi.write(j, tf.zeros_like(x[0, 0])))
    out_cxi = tf.cond(j < group_len, lambda: out_cxi.write(j, cx[start_batch+j, 0]), lambda: out_cxi.write(j, tf.zeros_like(cx[0, 0])))
    out_xmaski = tf.cond(j < group_len, lambda: out_xmaski.write(j, x_mask[start_batch+j, 0]), lambda: out_xmaski.write(j, tf.zeros_like(x_mask[0, 0])))
    
    return tf.add(j, 1), x_new, x, cx, x_mask, start_batch, out_x_newi, out_xi, out_cxi, out_xmaski, group_len

  with tf.control_dependencies([tf.assert_equal(tf.shape(x)[0], tf.reduce_sum(group)), tf.assert_equal(tf.shape(x)[0], tf.shape(x_mask)[0])]):
    for i in range(target_batch_size):
      out_x_newi = tensor_array_ops.TensorArray(dtype=tf.int32, size=max_doc_num, dynamic_size=False, infer_shape=True)
      out_xi = tensor_array_ops.TensorArray(dtype=tf.int32, size=max_doc_num, dynamic_size=False, infer_shape=True)
      out_cxi = tensor_array_ops.TensorArray(dtype=tf.int32, size=max_doc_num, dynamic_size=False, infer_shape=True)
      out_xmaski = tensor_array_ops.TensorArray(dtype=tf.bool, size=max_doc_num, dynamic_size=False, infer_shape=True)
      
      j, x_new, x, cx, x_mask, start_batch, out_x_newi, out_xi, out_cxi, out_xmaski, group_len = control_flow_ops.while_loop(
        cond=lambda j, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10: j < max_doc_num, 
        body=outer_while_body,
        loop_vars=[tf.constant(0, dtype=tf.int32), x_new, x, cx, x_mask, start_batch, out_x_newi, out_xi, out_cxi, out_xmaski, group[i]])

      start_batch += group[i]
      out_x_new.append(out_x_newi.stack())
      out_x.append(out_xi.stack())
      out_cx.append(out_cxi.stack())
      out_xmask.append(out_xmaski.stack())

  out_x_new = tf.stack(out_x_new, axis=0)
  out_x = tf.stack(out_x, axis=0)
  out_cx = tf.stack(out_cx, axis=0)
  out_xmask = tf.stack(out_xmask, axis=0)

  return out_x_new, out_x, out_cx, out_xmask



def combine_docs(doc_1, doc_2, doc_1_len, doc_2_len, batch_size, wd, emb_dim):
  combined_docs = []
  max_len = tf.maximum(doc_1_len + doc_2_len)
  for i in range(batch_size):
    combined_doc = []
    #combined_docs.append(combined_doc)
    doc_1_slice = tf.slice(doc_1[i], [0, 0], [doc_1_len[i], -1])
    for j in range(wd):
      doc_2_slice = tf.slice(doc_2[i,j], [0, 0], [doc_2_len[i][j], -1])
      zero_pad = tf.zeros([max_len - doc_1_len[i] - doc_2_len[i][j], emb_dim])
      new_doc = tf.concat([doc_1_slice, doc_2_slice, zero_pad], axis=0)
      combined_doc.append(new_doc)

    combined_doc = tf.stack(combined_doc, axis=0)
    combined_docs.append(combined_doc)

  combined_docs = tf.stack(combined_docs, axis=0)
  return combined_docs


def select_topn_doc_idx(batch_size, topn, x_group):
  size = tf.reduce_sum([tf.minimum(x_group[i], topn) for i in range(batch_size)])
  first_n_doc_idx = tensor_array_ops.TensorArray(dtype=tf.int32, size=size, dynamic_size=False, infer_shape=True)
  
  def while_body(j, i, first_n_doc_idx, x_group, counter):
    to_append = tf.cond(tf.reduce_sum(x_group) > tf.reduce_sum(x_group[:i]) + j, lambda: tf.reduce_sum(x_group[:i])+j, lambda: tf.reduce_sum(x_group[:i]))
    first_n_doc_idx = first_n_doc_idx.write(counter, to_append)
    return j + 1, i, first_n_doc_idx, x_group, counter + 1

  counter = 0
  for i in range(batch_size):
    # if i == 0:
    #   for j in range(topn):
    #     first_n_doc_idx = first_n_doc_idx.write(j)
    # else:
    j, _, first_n_doc_idx, _, counter = control_flow_ops.while_loop(
      cond=lambda j, _1, _2, _3, _4: j < tf.minimum(x_group[i], topn), 
      body=while_body,
      loop_vars=[tf.constant(0, dtype=tf.int32), i, first_n_doc_idx, x_group, counter])

  first_n_doc_idx = first_n_doc_idx.stack()
  return first_n_doc_idx


def span_to_avg_emb(original_span, original_context, batch_size, model):
  spans = tf.unstack(tf.squeeze(original_span, axis=1))
  contexts = tf.unstack(tf.squeeze(original_context, axis=1))

  def while_body(j, avg_emb_ta, mask_ta, span, context):
    avg_emb = tf.cond(span[j,1]-span[j,0]>0, lambda: tf.reduce_mean(context[span[j,0]:span[j,1], :], axis=-2), \
      lambda: tf.zeros_like(context[span[j,0]]))
    
    avg_emb_ta = avg_emb_ta.write(j, avg_emb)
    mask_ta = mask_ta.write(j, (span[j,1]-span[j,0]>0))
    return j + 1, avg_emb_ta, mask_ta, span, context

  candidate_emb = []
  mask = []
  for i in range(batch_size):
    span = spans[i]
    context = contexts[i]

    avg_emb_i = tensor_array_ops.TensorArray(dtype=tf.float32, size=tf.shape(span)[0], dynamic_size=False, infer_shape=True)
    mask_i = tensor_array_ops.TensorArray(dtype=tf.bool, size=tf.shape(span)[0], dynamic_size=False, infer_shape=True)

    j, avg_emb_i, mask_i, span, context = control_flow_ops.while_loop(
      cond=lambda j, _1, _2, _3, _4: j < tf.shape(span)[0], body=while_body, \
      loop_vars=[tf.constant(0, dtype=tf.int32), avg_emb_i, mask_i, span, context])

    candidate_emb.append(avg_emb_i.stack())
    mask.append(mask_i.stack())

  candidate_emb = tf.expand_dims(tf.stack(candidate_emb, axis=0), axis=1)
  mask = tf.expand_dims(tf.stack(mask, axis=0), axis=1)
  #model.cand_mask = mask
  #model.candidate_emb = candidate_emb
  return candidate_emb, mask


def reconstruct_batchesV2(p0, p_st, p_mask, group, sents_len, target_batch_size, max_para_size, model):
  """
  For hierarchical mac cell.
  """
  out_p0 = []
  out_pst = []
  out_pmask = []
  out_pdocmask = []
  out_sents_len = []
  start_batch = 0
  #emb_dim = 160
  #max_doc_num = tf.shape(p0)[1]
  max_doc_num = tf.reduce_max(group)

  def outer_while_body(j, p0, p_st, p_mask, sents_len, start_batch, out_p0i, out_psti, out_pmaski, out_pdocmaski, out_sents_leni, group_len):
    # with tf.control_dependencies([tf.assert_equal(tf.shape(p0)[0], tf.shape(p_mask)[0])]):
    out_p0i = tf.cond(j < group_len, lambda: out_p0i.write(j, p0[start_batch+j, 0]), lambda: out_p0i.write(j, tf.zeros_like(p0[0, 0])))
    out_psti = tf.cond(j < group_len, lambda: out_psti.write(j, p_st[start_batch+j]), lambda: out_psti.write(j, tf.zeros_like(p_st[0])))
    out_pmaski = tf.cond(j < group_len, lambda: out_pmaski.write(j, p_mask[start_batch+j, 0]), lambda: out_pmaski.write(j, tf.zeros_like(p_mask[0, 0])))
    out_pdocmaski = tf.cond(j < group_len, lambda: out_pdocmaski.write(j, True), lambda: out_pdocmaski.write(j, False))
    out_sents_leni = tf.cond(j < group_len, lambda: out_sents_leni.write(j, sents_len[start_batch+j, 0]), lambda: out_sents_leni.write(j, tf.zeros_like(sents_len[0, 0])))
    # k, p0, j, start_batch, out_i, mask_i, group = control_flow_ops.while_loop(
    #   cond=lambda k, _1, _2, _3, _4, _5: k < x_len[start_batch+j, 0],
    #   body=inner_while_body,
    #   loop_vars=[tf.constant(0, dtype=tf.int32), p0, j, start_batch, \
    #   out_i, mask_i, group])
    return tf.add(j, 1), p0, p_st, p_mask, sents_len, start_batch, out_p0i, out_psti, out_pmaski, out_pdocmaski, out_sents_leni, group_len

  with tf.control_dependencies([tf.assert_equal(tf.shape(p0)[0], tf.reduce_sum(group)), tf.assert_equal(tf.shape(p0)[0], tf.shape(p_mask)[0])]):
    for i in range(target_batch_size):
      out_p0i = tensor_array_ops.TensorArray(dtype=tf.float32, size=max_doc_num, dynamic_size=False, infer_shape=True)
      out_psti = tensor_array_ops.TensorArray(dtype=tf.float32, size=max_doc_num, dynamic_size=False, infer_shape=True)
      out_pmaski = tensor_array_ops.TensorArray(dtype=tf.bool, size=max_doc_num, dynamic_size=False, infer_shape=True)
      out_pdocmaski = tensor_array_ops.TensorArray(dtype=tf.bool, size=max_doc_num, dynamic_size=False, infer_shape=True)
      out_sents_leni = tensor_array_ops.TensorArray(dtype=tf.int32, size=max_doc_num, dynamic_size=False, infer_shape=True)
      j, p0, p_st, p_mask, sents_len, start_batch, out_p0i, out_psti, out_pmaski, out_pdocmaski, out_sents_leni, group_len = control_flow_ops.while_loop(
        cond=lambda j, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11: j < max_doc_num, 
        body=outer_while_body,
        loop_vars=[tf.constant(0, dtype=tf.int32), p0, p_st, p_mask, sents_len, start_batch, out_p0i, out_psti, out_pmaski, out_pdocmaski, out_sents_leni, group[i]])

      start_batch += group[i]
      out_p0.append(out_p0i.stack())
      out_pst.append(out_psti.stack())
      out_pmask.append(out_pmaski.stack())
      out_pdocmask.append(out_pdocmaski.stack())
      out_sents_len.append(out_sents_leni.stack())

  out_p0 = tf.stack(out_p0, axis=0)
  out_pst = tf.stack(out_pst, axis=0)
  out_pmask = tf.stack(out_pmask, axis=0)
  out_pdocmask = tf.stack(out_pdocmask, axis=0)
  out_sents_len = tf.stack(out_sents_len, axis=0)

  return out_p0, out_pst, out_pmask, out_pdocmask, out_sents_len


def reconstruct_batches(p0, x_len, group, target_batch_size, max_para_size, model, emb_dim=40):
  out = []
  mask = []
  start_batch = 0
  M = tf.shape(p0)[1]
  #emb_dim = 160
  #d = tf.shape(p0)[3]
  
  def inner_while_body(k, tracker, p0, j, start_batch, out_i, mask_i):
    out_i = out_i.write(tracker, p0[start_batch+j, 0, k])
    mask_i = mask_i.write(tracker, True)
    return k + 1, tracker+1, p0, j, start_batch, out_i, mask_i

  def outer_while_body(j, len_tracker, p0, x_len, start_batch, out_i, mask_i):
    k, len_tracker, p0, j, start_batch, out_i, mask_i = control_flow_ops.while_loop(
      cond=lambda k, _1, _2, _3, _4, _5, _6: k < x_len[start_batch+j, 0],
      body=inner_while_body,
      loop_vars=[tf.constant(0, dtype=tf.int32), len_tracker, p0, j, start_batch, \
      out_i, mask_i])
    return j + 1, len_tracker, p0, x_len, start_batch, out_i, mask_i

  def outer_while_body_2(j, p0, out_i, mask_i, emb_dim):
    out_i = out_i.write(j, tf.zeros(shape=[emb_dim], dtype=tf.float32))
    mask_i = mask_i.write(j, False)
    return j + 1, p0, out_i, mask_i, emb_dim

  sent_len = []
  for i in range(target_batch_size):
    out_i = tensor_array_ops.TensorArray(dtype=tf.float32, size=max_para_size, dynamic_size=False, infer_shape=True)
    mask_i = tensor_array_ops.TensorArray(dtype=tf.bool, size=max_para_size, dynamic_size=False, infer_shape=True)
    
    j, len_i, p0, x_len, start_batch, out_i, mask_i = control_flow_ops.while_loop(
      cond=lambda j, _1, _2, _3, _4, _5, _6: j < group[i], 
      body=outer_while_body,
      loop_vars=[tf.constant(0, dtype=tf.int32), tf.constant(0, dtype=tf.int32), p0, \
      x_len, start_batch, out_i, mask_i])

    j, p0, out_i, mask_i, emb_dim = control_flow_ops.while_loop(
      cond=lambda j, _1, _2, _3, _4: j < max_para_size,
      body=outer_while_body_2,
      loop_vars=[len_i, p0, out_i, mask_i, emb_dim])
    
    start_batch += group[i]
    out.append(out_i.stack())
    mask.append(mask_i.stack())
    sent_len.append(len_i)
  #assert len(out) == target_batch_size
  
  sent_len = tf.expand_dims(tf.stack(sent_len, axis=0), axis=1)
  model.recon_x_len = sent_len
  out = tf.expand_dims(tf.stack(out, axis=0), axis=1)
  mask = tf.expand_dims(tf.stack(mask, axis=0), axis=1)
  #assert out.get_shape()[0] == target_batch_size, (target_batch_size)
  #out.set_shape([target_batch_size, 1, None, emb_dim])
  #mask.set_shape([target_batch_size, 1, max_para_size])
  return out, sent_len, mask

def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
       is_train=None):
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  flat_args = [flatten(arg, 1) for arg in args]
  if input_keep_prob < 1.0:
    assert is_train is not None
    flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)
           for arg in flat_args]
  with tf.variable_scope(scope or 'Linear'):
    flat_out = _linear(flat_args, output_size, bias, bias_initializer=tf.constant_initializer(bias_start))
  out = reconstruct(flat_out, args[0], 1)
  if squeeze:
    out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])
  if wd:
    add_wd(wd)

  return out


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
  with tf.name_scope(name or "dropout"):
    if keep_prob < 1.0:
      d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
      out = tf.cond(is_train, lambda: d, lambda: x)
      return out
    return x


def softmax(logits, mask=None, scope=None):
  with tf.name_scope(scope or "Softmax"):
    if mask is not None:
      logits = exp_mask(logits, mask)
    flat_logits = flatten(logits, 1)
    flat_out = tf.nn.softmax(flat_logits)
    out = reconstruct(flat_out, logits, 1)

    return out


def softsel(target, logits, mask=None, scope=None):
  """

  :param target: [ ..., J, d] dtype=float
  :param logits: [ ..., J], dtype=float
  :param mask: [ ..., J], dtype=bool
  :param scope:
  :return: [..., d], dtype=float
  """
  with tf.name_scope(scope or "Softsel"):
    a = softmax(logits, mask=mask)
    target_rank = len(target.get_shape().as_list())
    out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
    return out


def double_linear_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
  with tf.variable_scope(scope or "Double_Linear_Logits"):
    first = tf.tanh(linear(args, size, bias, bias_start=bias_start, scope='first',
                 wd=wd, input_keep_prob=input_keep_prob, is_train=is_train))
    second = linear(first, 1, bias, bias_start=bias_start, squeeze=True, scope='second',
            wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
    if mask is not None:
      second = exp_mask(second, mask)
    return second


def linear_logits(args, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None, reuse=False):
  with tf.variable_scope(scope or "Linear_Logits", reuse=reuse):
    logits = linear(args, 1, bias, bias_start=bias_start, squeeze=True, scope='first',
            wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
    if mask is not None:
      logits = exp_mask(logits, mask)
    return logits


def sum_logits(args, mask=None, name=None):
  with tf.name_scope(name or "sum_logits"):
    if args is None or (nest.is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
      args = [args]
    rank = len(args[0].get_shape())
    logits = sum(tf.reduce_sum(arg, rank-1) for arg in args)
    if mask is not None:
      logits = exp_mask(logits, mask)
    return logits


def get_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None, func=None, reuse=False):
  if func is None:
    func = "sum"
  if func == 'sum':
    return sum_logits(args, mask=mask, name=scope)
  elif func == 'linear':
    return linear_logits(args, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
               is_train=is_train, reuse=reuse)
  elif func == 'double':
    return double_linear_logits(args, size, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                  is_train=is_train)
  elif func == 'dot':
    assert len(args) == 2
    arg = args[0] * args[1]
    return sum_logits([arg], mask=mask, name=scope)
  elif func == 'mul_linear':
    assert len(args) == 2
    arg = args[0] * args[1]
    return linear_logits([arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
               is_train=is_train)
  elif func == 'proj':
    assert len(args) == 2
    d = args[1].get_shape()[-1]
    proj = linear([args[0]], d, False, bias_start=bias_start, scope=scope, wd=wd, input_keep_prob=input_keep_prob,
            is_train=is_train)
    return sum_logits([proj * args[1]], mask=mask)
  elif func == 'tri_linear':
    assert len(args) == 2
    new_arg = args[0] * args[1]
    return linear_logits([args[0], args[1], new_arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
               is_train=is_train)
  else:
    raise Exception()


def highway_layer(arg, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
  with tf.variable_scope(scope or "highway_layer"):
    d = arg.get_shape()[-1]
    trans = linear([arg], d, bias, bias_start=bias_start, scope='trans', wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
    trans = tf.nn.relu(trans)
    gate = linear([arg], d, bias, bias_start=bias_start, scope='gate', wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
    gate = tf.nn.sigmoid(gate)
    out = gate * trans + (1 - gate) * arg
    return out


def highway_network(arg, num_layers, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
  with tf.variable_scope(scope or "highway_network"):
    prev = arg
    cur = None
    for layer_idx in range(num_layers):
      cur = highway_layer(prev, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx), wd=wd,
                input_keep_prob=input_keep_prob, is_train=is_train)
      prev = cur
    return cur


def conv1d(in_, filter_size, height, padding, is_train=None, keep_prob=1.0, scope=None):
  with tf.variable_scope(scope or "conv1d"):
    num_channels = in_.get_shape()[-1]
    filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype='float')
    bias = tf.get_variable("bias", shape=[filter_size], dtype='float')
    strides = [1, 1, 1, 1]
    if is_train is not None and keep_prob < 1.0:
      in_ = dropout(in_, keep_prob, is_train)
    xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias  # [N*M, JX, W/filter_stride, d]
    out = tf.reduce_max(tf.nn.relu(xxc), 2)  # [-1, JX, d]
    return out


def multi_conv1d(in_, filter_sizes, heights, padding, is_train=None, keep_prob=1.0, scope=None):
  with tf.variable_scope(scope or "multi_conv1d"):
    assert len(filter_sizes) == len(heights)
    outs = []
    for filter_size, height in zip(filter_sizes, heights):
      if filter_size == 0:
        continue
      out = conv1d(in_, filter_size, height, padding, is_train=is_train, keep_prob=keep_prob, scope="conv1d_{}".format(height))
      outs.append(out)
    concat_out = tf.concat(axis=2, values=outs)
    return concat_out
