import os

import tensorflow as tf
from os.path import join

from basic.main import main as m

flags = tf.app.flags

# Names and directories
flags.DEFINE_string("model_name", "basic", "Model name [basic]")
flags.DEFINE_string("dataset", "qangaroo", "[qangaroo]")
flags.DEFINE_string("data_dir", "data/qangaroo", "Data dir [data/qangaroo")
flags.DEFINE_string("run_id", "0", "Run ID [0]")
flags.DEFINE_string("out_base_dir", "out", "out base dir [out]")
flags.DEFINE_string("forward_name", "single", "Forward name [single]")
flags.DEFINE_string("answer_path", "", "Answer path []")
flags.DEFINE_string("eval_path", "", "Eval path []")
flags.DEFINE_string("load_path", "", "Load path []")
flags.DEFINE_string("shared_path", "", "Shared path []")

# Device placement
flags.DEFINE_string("device", "/cpu:0", "default device for summing gradients. [/cpu:0]")
flags.DEFINE_string("device_type", "gpu", "device for computing gradients (parallelization). cpu | gpu [gpu]")
flags.DEFINE_integer("num_gpus", 1, "num of gpus or cpus for computing gradients [1]")

# Essential training and test options
flags.DEFINE_string("mode", "test", "trains | test | forward [test]")
flags.DEFINE_boolean("load", True, "load saved data? [True]")
flags.DEFINE_bool('load_ema', True, "load exponential average of variables when testing?  [True]")
flags.DEFINE_bool("eval", True, "eval? [True]")
flags.DEFINE_bool("wy", False, "Use wy for loss / eval? [False]")
flags.DEFINE_bool("na", False, "Enable no answer strategy and learn bias? [False]")
flags.DEFINE_float("th", 0.5, "Threshold [0.5]")

# Training / test parameters
flags.DEFINE_integer("batch_size", 10, "Batch size [60]")
flags.DEFINE_integer("val_num_batches", 100, "validation num batches [100]")
flags.DEFINE_integer("test_num_batches", 0, "test num batches [0]")
flags.DEFINE_integer("num_epochs", 12, "Total number of epochs for training [12]")
flags.DEFINE_integer("num_steps", 30000, "Number of steps [20000]")
flags.DEFINE_integer("load_step", 0, "load step [0]")
flags.DEFINE_float("init_lr", 0.001, "Initial learning rate [0.001]")
flags.DEFINE_float("input_keep_prob", 0.8, "Input keep prob for the dropout of LSTM weights [0.8]")
flags.DEFINE_float("highway_keep_prob", 1.0, "Input keep prob for the dropout of LSTM weights [0.8]")
flags.DEFINE_float("keep_prob", 0.8, "Keep prob for the dropout of Char-CNN weights [0.8]")
flags.DEFINE_float("wd", 0.0, "L2 weight decay for regularization [0.0]")
flags.DEFINE_integer("hidden_size", 20, "Hidden size [100]") #100
flags.DEFINE_integer("char_out_size", 100, "char-level word embedding size [100]") #100
flags.DEFINE_integer("char_emb_size", 8, "Char emb size [8]")
flags.DEFINE_string("out_channel_dims", "100", "Out channel dims of Char-CNN, separated by commas [100]")
flags.DEFINE_string("filter_heights", "5", "Filter heights of Char-CNN, separated by commas [5]")
flags.DEFINE_bool("finetune", False, "Finetune word embeddings? [False]")
flags.DEFINE_bool("highway", True, "Use highway? [True]")
flags.DEFINE_integer("highway_num_layers", 2, "highway num layers [2]")
flags.DEFINE_bool("share_cnn_weights", True, "Share Char-CNN weights [True]")
flags.DEFINE_bool("share_lstm_weights", True, "Share pre-processing (phrase-level) LSTM weights [True]")
flags.DEFINE_float("var_decay", 0.999, "Exponential moving average decay for variables [0.999]")
flags.DEFINE_integer("emb_dim", 100, ".")

# Optimizations
flags.DEFINE_bool("cluster", False, "Cluster data for faster training [False]")

# Logging and saving options
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_integer("log_period", 100, "Log period [100]")
flags.DEFINE_integer("eval_period", 1000, "Eval period [1000]")
flags.DEFINE_integer("save_period", 1000, "Save Period [1000]")
flags.DEFINE_integer("max_to_keep", 20, "Max recent saves to keep [20]")
flags.DEFINE_bool("dump_eval", True, "dump eval? [True]")
flags.DEFINE_bool("dump_answer", True, "dump answer? [True]")
flags.DEFINE_bool("vis", False, "output visualization numbers? [False]")
flags.DEFINE_bool("dump_pickle", True, "Dump pickle instead of json? [True]")
flags.DEFINE_float("decay", 0.9, "Exponential moving average decay for logging values [0.9]")

# Thresholds for speed and less memory usage
flags.DEFINE_integer("word_count_th", 10, "word count th [100]")
flags.DEFINE_integer("char_count_th", 50, "char count th [500]")
flags.DEFINE_integer("sent_size_th", 8000, "sent size th [64]")  # 400
flags.DEFINE_integer("num_sents_th", 8, "num sents th [8]")
flags.DEFINE_integer("ques_size_th", 30, "ques size th [32]")
flags.DEFINE_integer("word_size_th", 16, "word size th [16]")
flags.DEFINE_integer("para_size_th", 256, "para size th [256]")

# Advanced training options
flags.DEFINE_bool("lower_word", True, "lower word [True]")
flags.DEFINE_bool("swap_memory", True, "swap memory? [True]")
flags.DEFINE_string("data_filter", "max", "max | valid | semi [max]")
flags.DEFINE_bool("use_glove_for_unk", True, "use glove for unk [False]")
flags.DEFINE_bool("known_if_glove", True, "consider as known if present in glove [False]")
flags.DEFINE_string("logit_func", "tri_linear", "logit func [tri_linear]")
flags.DEFINE_string("answer_func", "linear", "answer logit func [linear]")
flags.DEFINE_string("sh_logit_func", "tri_linear", "sh logit func [tri_linear]")

# Ablation options
flags.DEFINE_bool("use_char_emb", True, "use char emb? [True]")
flags.DEFINE_bool("use_word_emb", True, "use word embedding? [True]")
flags.DEFINE_bool("q2c_att", True, "question-to-context attention? [True]")
flags.DEFINE_bool("c2q_att", True, "context-to-question attention? [True]")

# My defined flags
flags.DEFINE_bool("split_supports", False, "Split support docs into multi paragraphs.")
flags.DEFINE_bool("cudnn_rnn", False, "Set to true when training on GPU.")

# Training options
flags.DEFINE_bool("reuse_cell", True, "Reuse MAC-RNN cell.")
flags.DEFINE_bool("get_query_subject", False, ".")
flags.DEFINE_bool("bidaf", False, ".")
flags.DEFINE_integer("select_top_n_doc", 0, ".")
flags.DEFINE_bool("supervise_first_doc", False, ".")
flags.DEFINE_bool("supervise_final_doc", False, ".")
flags.DEFINE_float("attn_loss_coeff", 0.5, ".")
flags.DEFINE_float("first_attn_loss_coeff", 0.5, ".")
flags.DEFINE_bool("shuffle_answer_doc_ids", False, ".")
flags.DEFINE_integer("truncate_at", 500, ".")
flags.DEFINE_bool("restore_ema_model_for_training", False, ".")
flags.DEFINE_bool("use_ranked_docs", False, ".")
flags.DEFINE_integer("read_topk_docs", 0, ".")

# Decoding options
flags.DEFINE_bool("attn_visualization", False, ".")
flags.DEFINE_string("filter_by_annotations", None, "[None | follow | single | multiple]")
flags.DEFINE_bool("save_selected_docs", False, ".")

# Mac cell
flags.DEFINE_string("reasoning_layer", 'mac_rnn', "[bidaf | mac_rnn | gru_rnn | macnet_hudson]")
flags.DEFINE_string("mac_prediction", 'candidates', "[span-single | span-dual | candidates]")
flags.DEFINE_bool("hierarchical_attn", True, "Use hierarchical_attn for reasoning_layer.")
flags.DEFINE_bool("use_control_unit", False, ".")
flags.DEFINE_integer("num_hops", 6, ".")
flags.DEFINE_string("mac_read_strategy", "one_doc_per_it_and_repeat_2nd_step", "[full | one_doc_per_it | mask_previous_max | one_doc_per_it_and_mask_all_read | one_doc_per_it_and_mask_all_read_pairs \
  | one_doc_per_it_and_repeat_2nd_step]")
flags.DEFINE_string("mac_output_unit", 'nested-triplet-mlp', "[similarity | nested-triplet-mlp | triplet-mlp]")
flags.DEFINE_string("mac_reasoning_unit", 'attention-lstm', "[answer_unit | mlp | bi-attn | attention-lstm | concat_first_sent | concat_full_doc]")
flags.DEFINE_string("mac_memory_state_update_rule", None, "[None | bi-attn]")
flags.DEFINE_string("mac_answer_state_update_rule", 'mlp', "[mlp | bi-attn]")
flags.DEFINE_string("oracle", None, "[None | extra]")
flags.DEFINE_bool("attention_cell_dropout", False, 'Dropout in Mac Cell attention cell.')

# Assembler
flags.DEFINE_bool("use_assembler", False, ".")
flags.DEFINE_string("assembler_type", "BiAttn", "BiAttn | ...")
flags.DEFINE_float("assembler_loss_coeff", 1., ".")
flags.DEFINE_bool("restore_non_assembler_model", False, ".")
flags.DEFINE_bool("assembler_repeat_first_doc", False, ".")
flags.DEFINE_bool("assembler_merge_query_st", False, ".")
flags.DEFINE_bool("assembler_biattn_w_first_doc", False, ".")
flags.DEFINE_integer("assembler_bidaf_layer", 1, ".")

#Medhop
flags.DEFINE_bool("medhop", False, '.')


# Other flags that are needed to be defined for tf1.6/py3.6 on AWS
flags.DEFINE_string("out_dir", "", "output directory.")
flags.DEFINE_string("save_dir", "", "output directory.")
flags.DEFINE_string("log_dir", "", "output directory.")
flags.DEFINE_string("eval_dir", "", "output directory.")
flags.DEFINE_string("answer_dir", "", "output directory.")

flags.DEFINE_integer("max_num_sents", 0, "As name.")
flags.DEFINE_integer("max_sent_size", 0, "As name.")
flags.DEFINE_integer("max_para_size", 0, "As name.")
flags.DEFINE_integer("max_ques_size", 0, "As name.")
flags.DEFINE_integer("max_ques_sub_size", 0, "As name.")
flags.DEFINE_integer("max_word_size", 0, "As name.")
flags.DEFINE_integer("char_vocab_size", 0, "As name.")
flags.DEFINE_integer("word_emb_size", 0, "As name.")
flags.DEFINE_integer("word_vocab_size", 0, "As name.")

def main(_):
  config = flags.FLAGS
  config.data_dir = os.path.join('data', config.dataset)
  if config.mode == 'test':
    config.input_keep_prob = 1.0
    config.highway_keep_prob = 1.0

  if config.read_topk_docs > 0:
    config.use_ranked_docs = True

  assert config.mac_prediction == 'candidates' or config.mac_prediction == 'span-single' \
  or config.mac_prediction == 'span-dual'
  config.out_dir = os.path.join(config.out_base_dir, config.model_name, config.dataset, str(config.run_id).zfill(2))

  if config.hierarchical_attn:
    config.get_query_subject = True

  if config.medhop:
    config.data_dir = join(config.data_dir, "medhop")
    config.num_steps =  3000
    config.save_period = 100
    config.log_period = 10
    config.eval_period = 6000
    config.val_num_batches = 0
  
  if config.oracle == 'extra':
    assert config.use_assembler

  if config.split_supports is True:
    config.data_dir = join(config.data_dir, 'split-supports')

    if config.select_top_n_doc > 0 or config.use_ranked_docs:
      if config.filter_by_annotations == 'single':
        if config.emb_dim == 300:
          config.data_dir = join(config.data_dir, 'candi-2layer-tfidf-truncated500-300d840b-followsingle')
        else:
          config.data_dir = join(config.data_dir, 'candi-2layer-tfidf-followsingle')
      elif config.filter_by_annotations == 'multiple':
        if config.emb_dim == 300:
          config.data_dir = join(config.data_dir, 'candi-2layer-tfidf-truncated500-300d840b-followmultiple')
        else:
          config.data_dir = join(config.data_dir, 'candi-2layer-tfidf-followmultiple')
      elif config.filter_by_annotations == 'follow':
        if config.emb_dim == 300:
          config.data_dir = join(config.data_dir, 'candi-2layer-tfidf-truncated500-300d840b-follow')
        else:
          config.data_dir = join(config.data_dir, 'candi-2layer-tfidf-follow')
      else:
        if config.emb_dim == 100:
          config.data_dir = join(config.data_dir, 'candi-2layer-tfidf')
        elif config.emb_dim == 300:
          print('300')
          if config.truncate_at == 500:
            config.data_dir = join(config.data_dir, 'candi-2layer-tfidf-truncated500-300d840b')
          elif config.truncate_at == 300:
            config.data_dir = join(config.data_dir, 'candi-2layer-tfidf-truncated300-300d840b')
          else:
            assert False, ("Large model must uses truncated data.")
        else:
          raise NotImplementedError
    else:
      if config.filter_by_annotations == 'follow':
        config.data_dir = join(config.data_dir, 'w-candi-follow')
      elif config.filter_by_annotations == 'single':
        config.data_dir = join(config.data_dir, 'w-candi-followsingle')
      elif config.filter_by_annotations == 'multiple':
        config.data_dir = join(config.data_dir, 'w-candi-followmultiple')
      else:
        if config.emb_dim == 100:           
          if config.use_doc_selector:
            config.data_dir = join(config.data_dir, 'w-candi')
          else:
            config.data_dir = join(config.data_dir, 'candi-2layer-tfidf')
        elif config.emb_dim == 300:
          print('300')          
          if config.truncate_at == 500:     
            config.data_dir = join(config.data_dir, 'w-candi-truncated500-300d840b')
          elif config.truncate_at == 300:
            config.data_dir = join(config.data_dir, 'w-candi-truncated300-300d840b')
          else:
            assert False, ("Large model must uses truncated data.")
        else:
          raise NotImplementedError
  else:
    config.data_dir = join(config.data_dir, 'concat-supports')
  m(config)

if __name__ == "__main__":
  tf.app.run()
