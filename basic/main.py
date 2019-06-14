import argparse
import json
import math
import os
import shutil
from pprint import pprint

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import csv

from basic.evaluator import ForwardEvaluator, MultiGPUF1Evaluator, MultiGPUF1CandidateEvaluator, MultiGPUF1CandidateDocSelEvaluator
from basic.graph_handler import GraphHandler
from basic.model import get_multi_gpu_models
from basic.trainer import MultiGPUTrainer
from basic.read_data import read_data, get_qangaroo_data_filter, update_config
from my.tensorflow import get_num_params


def main(config):
  set_dirs(config)
  with tf.device(config.device):
    if config.mode == 'train':
      _train(config)
    elif config.mode == 'test':
      _test(config)
    else:
      raise ValueError("invalid value for 'mode': {}".format(config.mode))


def set_dirs(config):
  # create directories
  assert config.load or config.mode == 'train', "config.load must be True if not training"
  if not config.load and os.path.exists(config.out_dir):
    raise Exception("no_load is set to True, but the out_dir already exists")
    #shutil.rmtree(config.out_dir)

  config.save_dir = os.path.join(config.out_dir, "save")
  config.log_dir = os.path.join(config.out_dir, "log")
  config.eval_dir = os.path.join(config.out_dir, "eval")
  config.answer_dir = os.path.join(config.out_dir, "answer")
  if not os.path.exists(config.out_dir):
    os.makedirs(config.out_dir)
  if not os.path.exists(config.save_dir):
    os.mkdir(config.save_dir)
  if not os.path.exists(config.log_dir):
    os.mkdir(config.log_dir)
  if not os.path.exists(config.answer_dir):
    os.mkdir(config.answer_dir)
  if not os.path.exists(config.eval_dir):
    os.mkdir(config.eval_dir)



def _train(config):
  if config.dataset == 'qangaroo':
    data_filter = get_qangaroo_data_filter(config)
  else:
    raise NotImplementedError

  train_data = read_data(config, 'train', config.load, data_filter=data_filter)
  dev_data = read_data(config, 'dev', True, data_filter=data_filter)
  update_config(config, [train_data, dev_data])
  
  word2vec_dict = train_data.shared['lower_word2vec'] if config.lower_word else train_data.shared['word2vec']
  word2idx_dict = train_data.shared['word2idx']
  idx2vec_dict = {word2idx_dict[word]: vec for word, vec in word2vec_dict.items() if word in word2idx_dict}
  emb_mat = np.array([idx2vec_dict[idx] if idx in idx2vec_dict
            else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
            for idx in range(config.word_vocab_size)])
  
  # construct model graph and variables (using default graph)
  pprint(config.__flags, indent=2)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  with sess.as_default():
    models = get_multi_gpu_models(config, emb_mat)
    model = models[0]
    print("num params: {}".format(get_num_params()))
    trainer = MultiGPUTrainer(config, models)
    if config.reasoning_layer is not None and config.mac_prediction == 'candidates':
      evaluator = MultiGPUF1CandidateEvaluator(config, models, tensor_dict=model.tensor_dict if config.vis else None)
    else:
      evaluator = MultiGPUF1Evaluator(config, models, tensor_dict=model.tensor_dict if config.vis else None)
    graph_handler = GraphHandler(config, model)  # controls all tensors and variables in the graph, including loading /saving

    # Variables
    #gpu_options = tf.GPUOptions(allow_growth=True)
    #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    graph_handler.initialize(sess)

  # Begin training
  num_steps = config.num_steps or int(math.ceil(train_data.num_examples / (config.batch_size * config.num_gpus))) * config.num_epochs
  global_step = 0
  
  for batches in tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus,
                           num_steps=num_steps, shuffle=True, cluster=config.cluster), total=num_steps):

    INSUFFICIENT_DATA = False
    for batch in batches:
      _, ds = batch
      if len(ds.data['x']) < config.batch_size:
        INSUFFICIENT_DATA = True
        break
    if INSUFFICIENT_DATA:
      continue

    global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
    get_summary = global_step % config.log_period == 0

    loss, summary, train_op = trainer.step(sess, batches, get_summary=get_summary)
    if get_summary:
      graph_handler.add_summary(summary, global_step)

    # occasional saving
    if global_step % config.save_period == 0:
      graph_handler.save(sess, global_step=global_step)

    if not config.eval:
      continue

    # Occasional evaluation
    if global_step % config.eval_period == 0:
      num_steps = math.ceil(dev_data.num_examples / (config.batch_size * config.num_gpus))
      if 0 < config.val_num_batches < num_steps:
        num_steps = config.val_num_batches
      e_dev = evaluator.get_evaluation_from_batches(
        sess, tqdm(dev_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps))
      graph_handler.add_summaries(e_dev.summaries, global_step)
      e_train = evaluator.get_evaluation_from_batches(
        sess, tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps)
      )
      graph_handler.add_summaries(e_train.summaries, global_step)
      if config.dump_eval:
        graph_handler.dump_eval(e_dev)
      if config.dump_answer:
        graph_handler.dump_answer(e_dev)
  if global_step % config.save_period != 0:
    graph_handler.save(sess, global_step=global_step)


def _test(config):
  if config.save_selected_docs:
    test_data = read_data(config, 'filtered_test', True)
  else:
    test_data = read_data(config, 'test', True)
  update_config(config, [test_data])

  if config.use_glove_for_unk:
    word2vec_dict = test_data.shared['lower_word2vec'] if config.lower_word else test_data.shared['word2vec']
    new_word2idx_dict = test_data.shared['new_word2idx']
    idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
    new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
    #config.new_emb_mat = new_emb_mat

  pprint(config.__flags, indent=2)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  with sess.as_default():
    models = get_multi_gpu_models(config, None)
    model = models[0]
    evaluator = MultiGPUF1CandidateEvaluator(config, models, tensor_dict=models[0].tensor_dict if config.vis else None)
    graph_handler = GraphHandler(config, model)

    graph_handler.initialize(sess)
  num_steps = math.ceil(test_data.num_examples / (config.batch_size * config.num_gpus))
  if 0 < config.test_num_batches < num_steps:
    num_steps = config.test_num_batches

  e = None
  if config.save_selected_docs:
    writer = csv.writer(open("model_chains.csv", 'w'))
    
  for multi_batch in tqdm(test_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps, cluster=config.cluster), total=num_steps):
    ei, doc_lst = evaluator.get_evaluation(sess, multi_batch)
    if config.save_selected_docs:
      selected_docs = []
      for ranked_docs in doc_lst:
        selected_docs.append(ranked_docs[0][0])
        #print(selected_doc)
      writer.writerow(selected_docs)

    e = ei if e is None else e + ei
    if config.vis:
      eval_subdir = os.path.join(config.eval_dir, "{}-{}".format(ei.data_type, str(ei.global_step).zfill(6)))
      if not os.path.exists(eval_subdir):
        os.mkdir(eval_subdir)
      path = os.path.join(eval_subdir, str(ei.idxs[0]).zfill(8))
      graph_handler.dump_eval(ei, path=path)
      
  print(e)
  if config.dump_answer:
    print("dumping answer ...")
    graph_handler.dump_answer(e)
  if config.dump_eval:
    print("dumping eval ...")
    graph_handler.dump_eval(e)


def _get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("config_path")
  return parser.parse_args()


class Config(object):
  def __init__(self, **entries):
    self.__dict__.update(entries)


def _run():
  args = _get_args()
  with open(args.config_path, 'r') as fh:
    config = Config(**json.load(fh))
    main(config)


if __name__ == "__main__":
  _run()
