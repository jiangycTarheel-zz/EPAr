import tensorflow as tf
import math
from basic.model import Model
from my.tensorflow import average_gradients
import numpy as np

class Trainer(object):
  def __init__(self, config, model):
    assert isinstance(model, Model)
    self.config = config
    self.model = model
    self.opt = tf.train.AdamOptimizer(config.init_lr)
    self.loss = model.get_loss()
    self.var_list = model.get_var_list()
    self.global_step = model.get_global_step()
    self.summary = model.summary
    self.grads = self.opt.compute_gradients(self.loss, var_list=self.var_list)
    self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

  def get_train_op(self):
    return self.train_op

  def step(self, sess, batch, get_summary=False):
    assert isinstance(sess, tf.Session)
    _, ds = batch
    feed_dict = self.model.get_feed_dict(ds, True)
    
    if get_summary:
      loss, summary, train_op = \
        sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
    else:
      loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
      summary = None
    return loss, summary, train_op


class MultiGPUTrainer(object):
  def __init__(self, config, models):
    model = models[0]
    assert isinstance(model, Model)
    self.config = config
    self.model = model
    self.opt = tf.train.AdamOptimizer(config.init_lr)
    self.var_list = model.get_var_list('model_network')
    self.global_step = model.get_global_step()
    self.summary = model.summary
    self.models = models
    losses, grads_list = [], []

    for gpu_idx, model in enumerate(models):
      with tf.name_scope("grads_{}".format(gpu_idx)), tf.device("/{}:{}".format(config.device_type, gpu_idx)):
        loss = model.get_loss()
        grads = self.opt.compute_gradients(loss, var_list=self.var_list)
        losses.append(loss)
        grads_list.append(grads)

      self.loss = tf.add_n(losses)/len(losses)
      self.grads = average_gradients(grads_list)

      grad_vars = [x[1] for x in self.grads]
      gradients = [x[0] for x in self.grads]  
      clipped, _ = tf.clip_by_global_norm(gradients, 2)

      self.train_op = self.opt.apply_gradients(zip(clipped, grad_vars), global_step=self.global_step)

      with tf.control_dependencies([self.train_op]):
        self.dummy = tf.constant(0, name='dummy')
        
        
  def setup_session_partial_run(self, sess, fetches, feeds):
    sess.partial_run_setup(fetches, feeds)


  def step(self, sess, batches, get_summary=False):
    partial_run = False
    assert isinstance(sess, tf.Session)
    feed_dict = {}
    for batch, model in zip(batches, self.models):
      _, ds = batch
      feed_dict.update(model.get_feed_dict(ds, True, sess))

    candidate_spans = feed_dict[self.model.candidate_spans]
    candidate_span_y = feed_dict[self.model.candidate_span_y]

    if self.config.use_assembler:
      new_feed_dict = {}
      to_run = []
      feeds = list(feed_dict.keys())
      for mid, model in enumerate(self.models):
        to_run += [model.mac_rnn_cell.qsub_topk_ids, model.mac_rnn_cell.qsub_topk_probs, model.mac_rnn_cell.qsub_all_probs, model.yp, model.yp_list, model.mac_rnn_cell.doc_attn, \
        model.x_sents_len_reconstruct]
        feeds += [model.selected_sent_ids]

      to_run += [self.loss, self.summary, self.dummy] if get_summary else [self.loss, self.dummy]
      handle = sess.partial_run_setup(to_run, feeds)

      for mid, (batch, model) in enumerate(zip(batches, self.models)):
        data_cand_word = batch[1].data['cand_word']
        data_x = batch[1].data['x']
        if len(data_x) < self.config.batch_size:
          data_cand_word = data_cand_word + data_cand_word
          data_x = data_x + data_x
        partial_run = True
        sents_ids, handle, _, _ = model.assembler.get_sentence_ids(sess, data_cand_word, data_x, feed_dict, handle, mid) 
        new_feed_dict[model.selected_sent_ids] = sents_ids

    if get_summary:
      if partial_run:
        loss, summary, train_op = \
          sess.partial_run(handle, [self.loss, self.summary, self.dummy], feed_dict=new_feed_dict)
      else:
        loss, summary, train_op = \
          sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
    else:
      if partial_run:
        loss, train_op = \
          sess.partial_run(handle, [self.loss, self.dummy], feed_dict=new_feed_dict)
        
      else:
        loss, train_op = \
          sess.run([self.loss, self.train_op], feed_dict=feed_dict)      
      summary = None
    if math.isnan(loss):
      logits, g1, cand_mask, cand_emb = sess.run([self.model.logits, self.model.g1, self.model.cand_mask, self.model.cand_emb], feed_dict)
      print(logits)
      print(candidate_spans[0])
      print(candidate_span_y)
      print("mask: ")
      print(cand_mask[0])
      print("cand_emb: ")
      print(cand_emb[0])
      print(feed_dict[self.model.answer_doc_ids])
      print(feed_dict[self.model.first_doc_ids])
      print(batches[0][1].data['ids'])
      exit()
    return loss, summary, train_op


  def step_docExpl_ansProp(self, sess, batches, get_summary=False):
    assert isinstance(sess, tf.Session)
    feed_dict = {}
    for batch, model in zip(batches, self.models):
      _, ds = batch
      feed_dict.update(model.get_feed_dict(ds, True, sess))

    partial_run = False
    
    if self.config.use_assembler is not None:
      new_feed_dict = {}
      to_run = []
      feeds = list(feed_dict.keys())
      for mid, model in enumerate(self.models):
        to_run += [model.mac_rnn_cell.qsub_topk_ids, model.mac_rnn_cell.qsub_topk_probs, model.mac_rnn_cell.qsub_all_probs, model.yp, model.yp_list, model.mac_rnn_cell.doc_attn, \
        model.x_sents_len_reconstruct]
        feeds += [model.selected_sent_ids]

      to_run += [self.loss_1, self.summary, self.dummy_1] if get_summary else [self.loss_1, self.dummy_1]
      handle = sess.partial_run_setup(to_run, feeds)

      for mid, (batch, model) in enumerate(zip(batches, self.models)):
        data_cand_word = batch[1].data['cand_word']
        data_x = batch[1].data['x']
        if len(data_x) < self.config.batch_size:
          data_cand_word = data_cand_word + data_cand_word
          data_x = data_x + data_x
        partial_run = True

        sents_ids, handle, _, _ = model.assembler.get_sentence_ids(sess, data_cand_word, data_x, feed_dict, handle, mid)        
        new_feed_dict[model.selected_sent_ids] = sents_ids
    
    if get_summary:
      if partial_run:
        loss, summary, train_op = sess.partial_run(handle, [self.loss_1, self.summary, self.dummy_1], feed_dict=new_feed_dict)
      else:
        loss, summary, train_op = sess.run([self.loss_1, self.summary, self.train_op_1], feed_dict=feed_dict)
    else:
      if partial_run:
        loss, train_op = sess.partial_run(handle, [self.loss_1, self.dummy_1], feed_dict=new_feed_dict)
      else:
        loss, train_op = sess.run([self.loss_1, self.train_op_1], feed_dict=feed_dict)      
      summary = None

    if math.isnan(loss):
      logits, g1, cand_mask, cand_emb = sess.run([self.model.logits, self.model.g1, self.model.cand_mask, self.model.cand_emb], feed_dict)
      print(logits)
      print(candidate_spans[0])
      print(candidate_span_y)
      print("mask: ")
      print(cand_mask[0])
      print("cand_emb: ")
      print(cand_emb[0])
      print(feed_dict[self.model.answer_doc_ids])
      print(feed_dict[self.model.first_doc_ids])
      print(batches[0][1].data['ids'])
      exit()
    return loss, summary, train_op


  def step_assembler(self, sess, batches, get_summary=False):
    partial_run = False
    assert isinstance(sess, tf.Session)
    feed_dict = {}
    for batch, model in zip(batches, self.models):
      _, ds = batch
      feed_dict.update(model.get_feed_dict(ds, True, sess))

    candidate_spans = feed_dict[self.model.candidate_spans]
    candidate_span_y = feed_dict[self.model.candidate_span_y]

    if self.config.use_assembler:
      new_feed_dict = {}
      to_run = []
      feeds = list(feed_dict.keys())
      for mid, model in enumerate(self.models):
        to_run += [model.mac_rnn_cell.qsub_topk_ids, model.mac_rnn_cell.qsub_topk_probs, model.mac_rnn_cell.qsub_all_probs, model.yp, model.yp_list, model.mac_rnn_cell.doc_attn, \
        model.x_sents_len_reconstruct]
        feeds += [model.selected_sent_ids]

      to_run += [self.loss_2, self.summary, self.dummy] if get_summary else [self.loss_2, self.dummy]
      handle = sess.partial_run_setup(to_run, feeds)

      for mid, (batch, model) in enumerate(zip(batches, self.models)):
        data_cand_word = batch[1].data['cand_word']
        data_x = batch[1].data['x']
        if len(data_x) < self.config.batch_size:
          data_cand_word = data_cand_word + data_cand_word
          data_x = data_x + data_x
        partial_run = True

        sents_ids, handle, _, _ = model.assembler.get_sentence_ids(sess, data_cand_word, data_x, feed_dict, handle, mid)        
        new_feed_dict[model.selected_sent_ids] = sents_ids
    
    if get_summary:
      if partial_run:
        loss, summary, train_op = \
          sess.partial_run(handle, [self.loss_2, self.summary, self.dummy], feed_dict=new_feed_dict)
      else:
        loss, summary, train_op = \
          sess.run([self.loss_2, self.summary, self.train_op], feed_dict=feed_dict)
    else:
      if partial_run:
        loss, train_op = \
          sess.partial_run(handle, [self.loss_2, self.dummy], feed_dict=new_feed_dict)
        
      else:
        loss, train_op = \
          sess.run([self.loss_2, self.train_op], feed_dict=feed_dict)      
      summary = None
    if math.isnan(loss):
      logits, g1, cand_mask, cand_emb = sess.run([self.model.logits, self.model.g1, self.model.cand_mask, self.model.cand_emb], feed_dict)
      print(logits)
      print(candidate_spans[0])
      print(candidate_span_y)
      print("mask: ")
      print(cand_mask[0])
      print("cand_emb: ")
      print(cand_emb[0])
      print(feed_dict[self.model.answer_doc_ids])
      print(feed_dict[self.model.first_doc_ids])
      print(batches[0][1].data['ids'])
      exit()
    return loss, summary, train_op


  