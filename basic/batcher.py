import numpy as np
import random

from basic.read_data import DataSet

def get_feed_dict(model, batch, is_train, supervised=True):
  assert isinstance(batch, DataSet)
  config = model.config

  N, M, JX, JQ, VW, VC, d, W = \
    config.batch_size, config.max_num_sents, config.max_sent_size, \
    config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size
  if config.split_supports:
    M = 1

  feed_dict = {}

  if config.split_supports:
    x, cx, x_mask = [], [], []
  else:
    x = np.zeros([N, M, JX], dtype='int32')
    cx = np.zeros([N, M, JX, W], dtype='int32')
    x_mask = np.zeros([N, M, JX], dtype='bool')

  q = np.zeros([N, JQ], dtype='int32')
  cq = np.zeros([N, JQ, W], dtype='int32')
  q_mask = np.zeros([N, JQ], dtype='bool')
  x_group = np.zeros([N], dtype='int32')

  q_sub = np.zeros([N, config.max_ques_sub_size], dtype='int32')
  cq_sub = np.zeros([N, config.max_ques_sub_size, W], dtype='int32')
  qsub_mask = np.zeros([N, config.max_ques_sub_size], dtype='bool')

  feed_dict[model.x] = x
  feed_dict[model.x_mask] = x_mask
  feed_dict[model.cx] = cx
  feed_dict[model.q] = q
  feed_dict[model.cq] = cq
  feed_dict[model.q_sub] = q_sub
  feed_dict[model.cq_sub] = cq_sub
  feed_dict[model.q_mask] = q_mask
  feed_dict[model.q_sub_mask] = qsub_mask
  feed_dict[model.is_train] = is_train

  feed_dict[model.x_group] = x_group

  if config.use_glove_for_unk:
    feed_dict[model.new_emb_mat] = batch.shared['new_emb_mat']

  X = batch.data['x']
  CX = batch.data['cx']

  if supervised:
    if config.split_supports is True:
      y = np.zeros([N, M, config.max_para_size], dtype='bool')
      y2 = np.zeros([N, M, config.max_para_size], dtype='bool')
      wy = np.zeros([N, M, config.max_para_size], dtype='bool')
    else:
      y = np.zeros([N, M, JX], dtype='bool')
      y2 = np.zeros([N, M, JX], dtype='bool')
      wy = np.zeros([N, M, JX], dtype='bool')
    na = np.zeros([N], dtype='bool')
    feed_dict[model.y] = y
    feed_dict[model.y2] = y2
    feed_dict[model.wy] = wy
    feed_dict[model.na] = na

    if config.supervise_final_doc or (config.oracle is not None):
      answer_doc_ids = - np.ones([N, 10], dtype='int32')
      answer_word_ids = - np.ones([N, 10], dtype='int32')
      feed_dict[model.answer_doc_ids] = answer_doc_ids
      feed_dict[model.answer_word_ids] = answer_word_ids
      for i, (doc_id, word_id, nai) in enumerate(zip(batch.data['answer_doc_ids'], batch.data['answer_ids_in_doc'], batch.data['na'])):
        if nai:
          if config.supervise_final_doc or config.oracle is not None:
            answer_doc_ids[i, 0] = 0
            answer_word_ids[i, 0] = 0
          continue
        assert len(doc_id) == len(word_id), (len(doc_id), len(word_id))
        if config.shuffle_answer_doc_ids:
          joint_id = list(zip(doc_id, word_id))
          np.random.shuffle(joint_id)  # shuffle doc_ids and word_ids jointly to randomize the supervision label.
          doc_id, word_id = zip(*joint_id) 
    
        for j in range(min(10, len(doc_id))):
          if config.select_top_n_doc > 0:
            if doc_id[j] < config.select_top_n_doc:
              answer_doc_ids[i, j] = doc_id[j]
              answer_word_ids[i, j] = word_id[j]
            else:
              answer_doc_ids[i, 0] = np.random.randint(0, config.select_top_n_doc)
              answer_word_ids[i, 0] = 0
              break
          else:
            answer_doc_ids[i, j] = doc_id[j]
            answer_word_ids[i, j] = word_id[j]

      for i,adis in enumerate(answer_doc_ids):
        if config.supervise_final_doc and adis[0] == -1:
          answer_doc_ids[i, 0] = 0
          answer_word_ids[i, 0] = 0
        
    if config.supervise_first_doc:
      first_doc_ids = np.zeros([N], dtype='int32')
      feed_dict[model.first_doc_ids] = first_doc_ids
      if config.select_top_n_doc == 0:         
        for i, doc_id in enumerate(batch.data['first_doc_ids']):
          first_doc_ids[i] = doc_id

    if config.mac_prediction == 'candidates':
      cand_spanss = batch.data['cand_span']
      cand_span_y = batch.data['cand_span_y']
      cand_wordss = batch.data['cand_word']
      cand_word_y = batch.data['cand_word_y']
      
      max_cand_size = 0
      for ci in cand_spanss:  
        max_cand_size = max(max_cand_size, len(ci))

      candidate_spans = np.zeros([N, M, max_cand_size, 2], dtype='int32')
      candidate_span_y = np.zeros([N, M], dtype='int32')
      num_exceed_cand = np.zeros([N, M, max_cand_size], dtype='int32')
      feed_dict[model.candidate_spans] = candidate_spans
      feed_dict[model.candidate_span_y] = candidate_span_y
      feed_dict[model.num_exceed_cand] = num_exceed_cand

      for i, (cand_spans, nai) in enumerate(zip(cand_spanss, batch.data['na'])):
        num_exceed_candi = 0
        num_exceed_cand_bfr_y = 0
        for j, cand_span in enumerate(cand_spans):
          assert cand_span[0][0] == 0, (cand_span)
          assert cand_span[1][0] == 0, (cand_span)

          # Ignore candidate spans that exceed max_para_size.
          if cand_span[0][1] > config.max_para_size or cand_span[1][1] > config.max_para_size:
            num_exceed_candi += 1
            if not nai:
              assert j != cand_span_y[i][0]            
              if j < cand_span_y[i][0]: 
                num_exceed_cand_bfr_y += 1
            continue
          num_exceed_cand[i, 0, j-num_exceed_candi] = num_exceed_candi
          candidate_spans[i, 0, j-num_exceed_candi, 0] = cand_span[0][1]
          candidate_spans[i, 0, j-num_exceed_candi, 1] = cand_span[1][1]
  
        if nai:
          #na[i] = nai
          continue

        assert cand_span_y[i][0] < len(cand_spans), (cand_span_y[i][0], len(cand_spans))
        candidate_span_y[i, 0] = cand_span_y[i][0] - num_exceed_cand_bfr_y

    for i, (xi, cxi, yi, nai) in enumerate(zip(X, CX, batch.data['y'], batch.data['na'])):
      if nai:
        na[i] = nai
        continue
      start_idx, stop_idx = random.choice(yi)
      j, k = start_idx
      j2, k2 = stop_idx
      
      y[i, j, k] = True
      y2[i, j2, k2-1] = True
      if j == j2:
        wy[i, j, k:k2] = True
      else:
        wy[i, j, k:len(batch.data['x'][i][j])] = True
        wy[i, j2, :k2] = True
      
        
  def _get_word(word):
    d = batch.shared['word2idx']
    for each in (word, word.lower(), word.capitalize(), word.upper()):
      if each in d:
        return d[each]
    if config.use_glove_for_unk:
      d2 = batch.shared['new_word2idx']
      for each in (word, word.lower(), word.capitalize(), word.upper()):
        if each in d2:
          return d2[each] + len(d)
    return 1

  def _get_char(char):
    d = batch.shared['char2idx']
    if char in d:
      return d[char]
    return 1

  if model.period_id is None:
    model.period_id = _get_word('.')
    print("period_id: %d" %model.period_id)

  max_sent_size = 0
  for xi in X:
    for xij in xi:
      max_sent_size = max(max_sent_size, len(xij))

  if config.split_supports is True:
    assert max_sent_size <= config.max_sent_size, (max_sent_size)

  for i, xi in enumerate(X):
    word_count = 0

    for j, xij in enumerate(xi):
      if j == config.max_num_sents:
        raise Exception("Exceed max_num_sents.")
        break
      if config.split_supports and word_count >= config.max_para_size:  
        if config.supervise_first_doc and j <= first_doc_ids[i]:
          first_doc_ids[i] = 0
        if config.supervise_final_doc:
          for count,id in enumerate(answer_doc_ids[i]):
            if j <= id:
              if config.supervise_final_doc and count == 0:  # Must have a valid doc id as the label of the final attention loss.
                answer_doc_ids[i] = [0] + [-1]*9
                answer_word_ids[i] = [0] + [-1]*9
              else:
                answer_doc_ids[i][count:] = - np.ones(10 - count)
                answer_word_ids[i][count:] = - np.ones(10 - count)
              break
        break
      
      if config.split_supports:  # Add every sentence as a separate batch.
        _xij = np.zeros([1, max_sent_size], dtype='int32')
        _xij_mask = np.zeros([1, max_sent_size], dtype='bool')
        x.append(_xij)
        x_mask.append(_xij_mask)

      for k, xijk in enumerate(xij):
        if config.split_supports and word_count >= config.max_para_size:  
        # same as word_count > config.max_sent_size if para_size_th and sent_size_th is set to same.
          break
        if k == config.max_sent_size:
          break
        each = _get_word(xijk)
        assert isinstance(each, int), each
        if config.split_supports:
          x[-1][0, k] = each
          x_mask[-1][0, k] = True
        else:
          x[i, j, k] = each
          x_mask[i, j, k] = True

        word_count += 1
      x_group[i] += 1

    if config.split_supports:
      assert word_count <= config.max_para_size, (word_count)
      

  for i, cxi in enumerate(CX):
    # Create a batch
    word_count = 0

    for j, cxij in enumerate(cxi):
      if j == config.max_num_sents:
        raise Exception("Exceed max_num_sents.")
        break
      if config.split_supports and word_count >= config.max_para_size:
        break

      if config.split_supports:  # Add every sentence as a separate batch.
        _cxij = np.zeros([1, max_sent_size, W], dtype='int32')
        cx.append(_cxij)

      for k, cxijk in enumerate(cxij):          
        if config.split_supports and word_count >= config.max_para_size:
          break
        if k == config.max_sent_size:
          break
        
        for l, cxijkl in enumerate(cxijk):
          if l == config.max_word_size:
            break
          if config.split_supports:
            cx[-1][0, k, l] = _get_char(cxijkl)
          else:
            cx[i, j, k, l] = _get_char(cxijkl)

        word_count += 1 #

    if config.split_supports:
      assert word_count <= config.max_para_size, (word_count)

  for i, qi in enumerate(batch.data['q']):
    for j, qij in enumerate(qi):
      q[i, j] = _get_word(qij)
      q_mask[i, j] = True

  for i, cqi in enumerate(batch.data['cq']):
    for j, cqij in enumerate(cqi):
      for k, cqijk in enumerate(cqij):
        cq[i, j, k] = _get_char(cqijk)
        if k + 1 == config.max_word_size:
          break

  if config.get_query_subject:
    for i, qi in enumerate(batch.data['q2']):
      for j, qij in enumerate(qi):
        q_sub[i, j] = _get_word(qij)
        qsub_mask[i, j] = True

    for i, cqi in enumerate(batch.data['cq2']):
      for j, cqij in enumerate(cqi):
        for k, cqijk in enumerate(cqij):
          cq_sub[i, j, k] = _get_char(cqijk)
          if k + 1 == config.max_word_size:
            break

  all_period_loc_plus_1 = []
  x_len = np.sum(np.asarray(x_mask).astype(int), axis=2)
  for i, xi in enumerate(x):
    period_found = [j + 1 for j, period in enumerate(list(xi[0])) if period == model.period_id]
    if len(period_found) > 10:
      all_period_loc_plus_1.append([period_found[:10]])
    else:
      to_append = period_found + [x_len[i][0]]*(10-len(period_found))
      all_period_loc_plus_1.append([to_append])
  x_sents_len = all_period_loc_plus_1
  feed_dict[model.x_sents_len] = np.stack(x_sents_len)
  if supervised:
    if config.split_supports is False:
      assert np.sum(~(x_mask | ~wy)) == 0  # if x_mask == 0, then wy must be 0

  num_examples = len(batch.data['x'])
  if num_examples < config.batch_size and (config.mac_reasoning_unit == 'attention-lstm'): # Last batch every epoch
    cutoff_1 = np.sum(x_group[:(config.batch_size-num_examples)])
    feed_dict[model.x] = np.stack(x+x[:cutoff_1])
    feed_dict[model.cx] = np.stack(cx+cx[:cutoff_1])
    feed_dict[model.x_mask] = np.stack(x_mask+x_mask[:cutoff_1])
    feed_dict[model.x_sents_len] = np.stack(x_sents_len+x_sents_len[:cutoff_1])

    cutoff_2 = config.batch_size - num_examples
    feed_dict[model.x_group] = np.concatenate([x_group[:num_examples], x_group[:cutoff_2]])
    feed_dict[model.q] = np.concatenate([q[:num_examples], q[:cutoff_2]])
    feed_dict[model.cq] = np.concatenate([cq[:num_examples], cq[:cutoff_2]])
    feed_dict[model.q_sub] = np.concatenate([q_sub[:num_examples], q_sub[:cutoff_2]])
    feed_dict[model.cq_sub] = np.concatenate([cq_sub[:num_examples], cq_sub[:cutoff_2]])
    feed_dict[model.q_mask] = np.concatenate([q_mask[:num_examples], q_mask[:cutoff_2]])
    feed_dict[model.q_sub_mask] = np.concatenate([qsub_mask[:num_examples], qsub_mask[:cutoff_2]])
  else:
    feed_dict[model.x] = np.stack(x)
    feed_dict[model.cx] = np.stack(cx)
    feed_dict[model.x_mask] = np.stack(x_mask)

  assert len(x) == len(x_mask)
  assert len(x) == sum(x_group), (len(x), sum(x_group))
  return feed_dict
