import argparse
import json
import os
import re
from collections import Counter

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import numpy as np
import random
from qangaroo.utils import get_word_span, get_word_idx
from nltk import word_tokenize, pos_tag

def main():
  args = get_args()
  prepro(args)


def get_args():
  parser = argparse.ArgumentParser()
  home = os.path.expanduser("~")
  source_dir = os.path.join(home, "data", "qangaroo/wikihop")
  target_dir = "data/qangaroo"
  glove_dir = os.path.join(home, "data", "glove")
  parser.add_argument('-s', "--source_dir", default=source_dir)
  parser.add_argument('-t', "--target_dir", default=target_dir)
  parser.add_argument("--train_name", default='train-v1.1.json')
  parser.add_argument("--train_ratio", default=0.9, type=int)
  parser.add_argument("--glove_corpus", default="6B")
  parser.add_argument("--glove_dir", default=glove_dir)
  parser.add_argument("--glove_vec_size", default=100, type=int)
  parser.add_argument("--mode", default="full", type=str)
  parser.add_argument("--single_path", default="", type=str)
  parser.add_argument("--tokenizer", default="PTB", type=str)
  parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
  parser.add_argument("--port", default=8000, type=int)
  parser.add_argument("--split", action='store_true')
  parser.add_argument("--suffix", default="")
  parser.add_argument("--split_supports", action='store_true')  # If False, concat n support docs for a query to a single paragraph.
  parser.add_argument("--find_candidates", action='store_true')  # If true, find and save the candidate answer spans.
  parser.add_argument("--rank_by_tfidf", action='store_true')  # If true, rank support docs by tfidf.
  parser.add_argument("--tfidf_layer", default=1, type=int)
  parser.add_argument("--back_tfidf", action='store_true') # If true, find the top-1 tfidf doc between (document)-(top-1 query-docs tfidf doc and doc with answer)
  parser.add_argument('--filter_by_annotations', default=None, type=str)
  parser.add_argument("--truncate_at", default=0, type=int)
  parser.add_argument("--randomize_docs", action='store_true')
  parser.add_argument("--randomize_examples", action='store_true')
  parser.add_argument("--keep_topk_docs_only", default=0, type=int)
  parser.add_argument("--medhop", action='store_true') 
  args =  parser.parse_args()

  if args.medhop:
    print("inside_medhop")
    args.source_dir = os.path.join(home, "data", "qangaroo/medhop")
    args.target_dir = "data/qangaroo/medhop"
  
  if args.split_supports:
    args.target_dir = os.path.join(args.target_dir, 'split-supports')
    if args.find_candidates:
      if args.rank_by_tfidf:
        if args.tfidf_layer == 2:
          if args.filter_by_annotations == 'single':
            if args.mode == 'double':
              args.target_dir = os.path.join(args.target_dir, 'candi-tfidf-answer-tfidf-single')
            else:
              args.target_dir = os.path.join(args.target_dir, 'candi-2layer-tfidf-followsingle')
          elif args.filter_by_annotations == 'multiple':
            if args.mode == 'double':
              args.target_dir = os.path.join(args.target_dir, 'candi-tfidf-answer-tfidf-multiple')
            else:
              args.target_dir = os.path.join(args.target_dir, 'candi-2layer-tfidf-followmultiple')
          elif args.filter_by_annotations == 'follow':
            if args.mode == 'double':
              args.target_dir = os.path.join(args.target_dir, 'candi-tfidf-answer-tfidf-follow')
            else:
              args.target_dir = os.path.join(args.target_dir, 'candi-2layer-tfidf-truncated500-follow')
          else:
            if args.mode == 'double':  # If mode is 'double', first find the doc ids with answer, and then preprocess again to place these docs in top-3.
              args.target_dir = os.path.join(args.target_dir, 'candi-2layer-tfidf-w-answer')
            else:
              if args.glove_vec_size == 300:
                args.target_dir = os.path.join(args.target_dir, 'candi-2layer-tfidf-truncated500-300d840b')
              else:
                args.target_dir = os.path.join(args.target_dir, 'candi-2layer-tfidf')
          #args.target_dir = os.path.join(args.target_dir, 'candi-2layer-tfidf')
        elif args.tfidf_layer == 1:
          if args.filter_by_annotations == 'single':
            if args.mode == 'double':
              args.target_dir = os.path.join(args.target_dir, 'candi-tfidf-answer-tfidf-single')
            else:
              args.target_dir = os.path.join(args.target_dir, 'candi-tfidf-single')
          elif args.filter_by_annotations == 'multiple':
            if args.mode == 'double':
              args.target_dir = os.path.join(args.target_dir, 'candi-tfidf-answer-tfidf-multiple')
            else:
              args.target_dir = os.path.join(args.target_dir, 'candi-tfidf-multiple')
          else:
            if args.mode == 'double':  # If mode is 'double', first find the doc ids with answer, and then preprocess again to place these docs in top-3.
              args.target_dir = os.path.join(args.target_dir, 'candi-tfidf-answer-tfidf')
            else:
              args.target_dir = os.path.join(args.target_dir, 'candi-tfidf')
        else:
          raise NotImplementedError
      else:
        if args.filter_by_annotations == 'follow':
          args.target_dir = os.path.join(args.target_dir, 'w-candi-follow')
        elif args.filter_by_annotations == 'single':
          args.target_dir = os.path.join(args.target_dir, 'w-candi-followsingle')
        elif args.filter_by_annotations == 'multiple':
          args.target_dir = os.path.join(args.target_dir, 'w-candi-followmultiple')
        else:
          if args.glove_vec_size == 300:
            args.target_dir = os.path.join(args.target_dir, 'w-candi-truncated500-300d840b')
          else:
            args.target_dir = os.path.join(args.target_dir, 'w-candi')
  else:
    args.target_dir = os.path.join(args.target_dir, 'concat-supports')
  
  return args

def create_all(args):
  out_path = os.path.join(args.source_dir, "all-v1.1.json")
  if os.path.exists(out_path):
    return
  train_path = os.path.join(args.source_dir, args.train_name)
  train_data = json.load(open(train_path, 'r'))
  dev_path = os.path.join(args.source_dir, args.dev_name)
  dev_data = json.load(open(dev_path, 'r'))
  train_data['data'].extend(dev_data['data'])
  print("dumping all data ...")
  json.dump(train_data, open(out_path, 'w'))


def prepro(args):
  if not os.path.exists(args.target_dir):
    os.makedirs(args.target_dir)

  if args.mode == 'filtered_dev':
    prepro_each(args, 'filtered_dev', out_name='filtered_test')
  elif args.mode == 'split_in_half':
    prepro_each(args, 'dev', start_ratio=0.0, stop_ratio=0.5, out_name='test')
    prepro_each(args, 'dev', start_ratio=0.5, stop_ratio=1.0, out_name='test')
  elif args.mode == 'double':
    if args.filter_by_annotations is None:  
      #prepro_each(args, 'dev', out_name='dev')
      prepro_each(args, 'train', out_name='train', save_json=False)
    prepro_each(args, 'dev', out_name='test', save_json=False)
  elif args.mode == 'full':
    prepro_each(args, 'dev', out_name='test')
    if args.filter_by_annotations is None:  
      #prepro_each(args, 'dev', out_name='dev')
      prepro_each(args, 'train', out_name='train')
  elif args.mode == 'all':
    create_all(args)
    prepro_each(args, 'dev', 0.0, 0.0, out_name='dev')
    prepro_each(args, 'dev', 0.0, 0.0, out_name='test')
    prepro_each(args, 'all', out_name='train')
  elif args.mode == 'single':
    assert len(args.single_path) > 0
    prepro_each(args, "NULL", out_name="single", in_path=args.single_path)
  else:
    prepro_each(args, 'train', out_name='train')
    prepro_each(args, 'dev', out_name='dev')


def save(args, data, shared, data_type):
  data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
  shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
  json.dump(data, open(data_path, 'w'))
  json.dump(shared, open(shared_path, 'w'))


def get_word2vec(args, word_counter):
  glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
  sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
  total = sizes[args.glove_corpus]
  word2vec_dict = {}
  with open(glove_path, 'r', encoding='utf-8') as fh:
    for line in tqdm(fh, total=total):
      array = line.lstrip().rstrip().split(" ")
      word = array[0]
      vector = list(map(float, array[1:]))
      if word in word_counter:
        word2vec_dict[word] = vector
      elif word.capitalize() in word_counter:
        word2vec_dict[word.capitalize()] = vector
      elif word.lower() in word_counter:
        word2vec_dict[word.lower()] = vector
      elif word.upper() in word_counter:
        word2vec_dict[word.upper()] = vector

  print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
  return word2vec_dict


def compute_answer_span(context, answer):
  """
  Find the first occuring answer in the context, and return its span.
  First find independent occurences (' '+answer+' '), if no such span exists, search for answer directly.
  IMPORTANT: After we find an independent occurance, all non-independent answers before it will be ignored because previous context is cut.
  """
  answer = answer.replace(' – ',' ').lower()
  context = context.lower()
  a = re.search(r'({})'.format(' '+answer+' '), context)
  if a is None:
    a2 = re.search(r'({})'.format(answer), context)
    if a2 is None:
      return None, None
    else:
      start = a2.start()
      end = start + len(answer)
      return start, end
  start = a.start()+1
  end = start + len(answer)
  return start, end


def _compute_candidate_span(context, candidate):
  # Find one span for a single candidate
  candidate = candidate.replace(' – ',' ').lower()
  context = context.lower()
  try:
    a = re.search(r'({})'.format(' '+candidate+' '), context)
    if a is None:
      try:
        a2 = re.search(r'({})'.format(candidate), context)
        if a2 is None:
          return None, None
        else:
          start = a2.start()
          end = start + len(candidate)
          return start, end
      except Exception as e:
        return None, None
    start = a.start()+1
    end = start + len(candidate)
    return start, end
  except Exception as e:
    return None, None


def compute_candidate_spans(context, candidates):
  # Find the one span for every candidate
  out_spans = []
  not_found = 0
  candidates_found = []
  real_candidates_found = []
  for i,candidate in enumerate(candidates):
    candidate = candidate.replace(' – ', ' ').lower()
    context = context.lower()
    if candidate == '*':
      not_found += 1
      continue
    try:
      a = re.search(r'({})'.format(' '+candidate+' '), context)
      if a is None:
        try:
          a2 = re.search(r'({})'.format(candidate), context)
          if a2 is None:
            not_found += 1
          else:
            candidates_found.append(candidates[i])
            start = a2.start()
            end = start + len(candidate)
            out_spans.append((start, end))
        except Exception as e:
          print(candidates)
          print(candidate)
          continue
        #not_found += 1
      else:
        candidates_found.append(candidates[i])
        start = a.start() + 1
        end = start + len(candidate)
        out_spans.append((start, end))
        real_candidates_found.append(candidate)
    except Exception as e:
      print(candidates)
      print(candidate)
      continue
    
  return out_spans, not_found, candidates_found, real_candidates_found


def find_doc_with_answer(start_id, doc_lens):
  total_len = 0
  for i,doc_len in enumerate(doc_lens):
    total_len += doc_len
    if start_id < total_len:
      return i, start_id - (total_len - doc_len)
  assert False, ("Answer not found.")
  return 0, 0


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None, save_json=True, pre_answer_doc_id=None):
  """
  data:
  q/cq: query
  y: answer
  rx/rcx: index pairs: (article_no, paragraph_no)
  cy:
  idxs:
  ids:
  answerss:
  na:
  
  shared:
  x/cx: tokenized paragraphs (words and chars)
  p: untokenized paragraphs
  """
  if args.tokenizer == "PTB":
    import nltk
    sent_tokenize = nltk.sent_tokenize
    def word_tokenize(tokens):
      return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
  elif args.tokenizer == 'Stanford':
    from my.corenlp_interface import CoreNLPInterface
    interface = CoreNLPInterface(args.url, args.port)
    sent_tokenize = interface.split_doc
    word_tokenize = interface.split_sent
  else:
    raise Exception()

  if args.medhop:
    from qangaroo.utils import process_tokens_medhop as process_tokens
  else:
    from qangaroo.utils import process_tokens

  if not args.split:
    sent_tokenize = lambda para: [para]

  source_path = in_path or os.path.join(args.source_dir, "{}.json".format(data_type))
  source_data = json.load(open(source_path, 'r'))

  q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
  na = []  # no answer
  cy = []
  x, cx = [], []
  x2 = []
  answers = []
  p, p2 = [], []
  q2, cq2 = [], []
  cand_span, ccand_span, cand_span_y = [], [], []
  cand_word, cand_word_y, cand_word_found, real_cand_word_found = [], [], [], []
  all_cand_spans, A1s, A2s, all_cand_doc_ids, all_cand_ids, all_cand_num_spans_found, real_cand_count = [], [], [], [], [], [], [] # To store all candidate spans, adjacency matrices, candidate's doc ids, candidate's ids
  answer_doc_ids, answer_ids_in_doc = [], []
  topk_2layer_tfidf_docs = []
  first_doc_ids = []
  word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
  start_ai = int(round(len(source_data) * start_ratio))
  stop_ai = int(round(len(source_data) * stop_ratio))
  mis_cand = 0
  found_answer_in_first_n = 0

  tfidf = TfidfVectorizer(strip_accents='unicode')
  bi = 0
  if args.randomize_examples:
    random.shuffle(source_data)
  for ai, article in enumerate(tqdm(source_data[start_ai:stop_ai])):
    candidates = article['candidates']
    query_sub = ' '.join(article['query'].split()[1:])
    query = article['query'].replace('_', ' ')  # get rid of '_' in, e.g., 'record_label' 
    supports = article['supports']
    answer = article['answer']

    if args.truncate_at > 0:
      for si, support in enumerate(supports):
        support_split = support.split(' ')[:args.truncate_at]
        if support_split[-1] != '.':
          support_split += '.'
        supports[si] = ' '.join(support_split)

    if args.randomize_docs:
      random.shuffle(supports)

    if args.filter_by_annotations is not None:
      annotations = article['annotations']
      not_follow = 0
      likely = 0
      follow = 0
      multiple = 0
      single = 0
      follow_and_multiple = 0
      for anno in annotations:
        if anno[0] == 'follows' and anno[1] == 'multiple':
          follow_and_multiple += 1
        if anno[0] == 'follows':
          follow += 1
        if anno[0] == 'not_follow':
          not_follow += 1
        if anno[0] == 'likely':
          likely += 1
        if anno[1] == 'multiple':
          multiple += 1
        if anno[1] == 'single':
          single += 1

      if args.filter_by_annotations == 'follow' and follow < 2:
        continue
      elif args.filter_by_annotations == 'multiple' and (follow < 2 or multiple < 2):
        continue
      elif args.filter_by_annotations == 'single' and (follow < 2 or single < 2):
        continue
    
    xp, cxp = [], []
    xp2 = []
    pp, pp2 = [], []

    x.append(xp)
    cx.append(cxp)
    p.append(pp)
    x2.append(xp2)
    p2.append(pp2)

    para_features = tfidf.fit_transform(supports)
    q_features = tfidf.transform([query_sub])
    dists = pairwise_distances(q_features, para_features, "cosine").ravel()
    sorted_ix = np.lexsort((supports, dists))
    first_doc_ids.append(np.asscalar(sorted_ix[0]))
    assert first_doc_ids[-1] < len(supports), (first_doc_ids[-1], len(supports))
    
    if args.rank_by_tfidf and save_json:
      first_doc_ids[-1] = 0
      para_features = tfidf.fit_transform(supports)
      q_features = tfidf.transform([query_sub])
      dists = pairwise_distances(q_features, para_features, "cosine").ravel()

      if pre_answer_doc_id is not None:
        dists[pre_answer_doc_id[bi]] = 0
      
      sorted_ix = np.lexsort((supports, dists))
      sorted_supports = [supports[idx] for idx in sorted_ix]
      
      if args.tfidf_layer == 1:
        if args.back_tfidf:
          para_features = tfidf.fit_transform(sorted_supports[2:])
          q_features = tfidf.transform([sorted_supports[1] + ' ' + sorted_supports[0]])
          dists = pairwise_distances(q_features, para_features, "cosine").ravel()
          sorted_ix = np.lexsort((sorted_supports[2:], dists))
          supports = [sorted_supports[idx + 2] for idx in sorted_ix]
          
          assert len(sorted_supports) == len(supports) + 2
          supports.insert(0, sorted_supports[1])
          supports.insert(2, sorted_supports[0])
        else:
          supports = sorted_supports
      elif args.tfidf_layer == 2:
        if args.mode == 'double':
          para_features = tfidf.fit_transform(sorted_supports[2:])
          q_features = tfidf.transform([sorted_supports[1]])
          dists = pairwise_distances(q_features, para_features, "cosine").ravel()
          sorted_ix = np.lexsort((sorted_supports[2:], dists))
          supports = [sorted_supports[idx + 2] for idx in sorted_ix]
          
          assert len(sorted_supports) == len(supports) + 2
          supports.insert(0, sorted_supports[1])
          supports.insert(2, sorted_supports[0])
        else:
          para_features = tfidf.fit_transform(sorted_supports[1:])
          q_features = tfidf.transform([sorted_supports[0]])
          dists = pairwise_distances(q_features, para_features, "cosine").ravel()
          sorted_ix = np.lexsort((sorted_supports[1:], dists))
          supports = [sorted_supports[idx + 1] for idx in sorted_ix]
          
          assert len(sorted_supports) == len(supports) + 1
          supports.insert(0, sorted_supports[0])
      else:
        raise NotImplementedError

      if args.keep_topk_docs_only > 0:
        supports = supports[:args.keep_topk_docs_only]
    else:
      sorted_supports = [supports[idx] for idx in sorted_ix]

      para_features = tfidf.fit_transform(supports)
      q_features = tfidf.transform([sorted_supports[0]])
      dists = pairwise_distances(q_features, para_features, "cosine").ravel()
      dists[sorted_ix[0]] = 1e30
      sorted_ix = np.lexsort((supports, dists))
      topk_2layer_tfidf_docs.append([])
      for kk in range(min(7, len(sorted_ix))):
        topk_2layer_tfidf_docs[-1].append(np.asscalar(sorted_ix[kk]))
      
    context = ''
    if args.split_supports is True:
      xi, cxi = [[]], [[]]
      xi_len = []
      for pi, _context in enumerate(supports):
        _context += ' '
        _context = _context.replace("''", '" ')
        _context = _context.replace("``", '" ')
        _context = _context.replace('  ', ' ').replace(' ', ' ')
        context += _context
        _xi = list(map(word_tokenize, sent_tokenize(_context)))
        _xi = [process_tokens(tokens) for tokens in _xi]  # xi = [["blahblah"]]
        _cxi = [[list(xijk) for xijk in xij] for xij in _xi]

        xi[0] += _xi[0]

        xi_len.append(len(_xi[0]))
        
        xp.append(_xi[0])
        cxp.append(_cxi[0])
        pp.append(_context)

      xp2.append(xi[0])
      pp2.append(context)
      assert sum(map(len,xp)) == np.array(xp2).shape[-1], (sum(map(len,xp)), np.array(xp2).shape[-1])
      
    else:
      for pi, _context in enumerate(supports):
        _context += ' '
        _context = _context.replace("''", '" ')
        _context = _context.replace("``", '" ')
        _context = _context.replace('  ', ' ').replace(' ', ' ')
        context += _context

      xi = list(map(word_tokenize, sent_tokenize(context)))
      xi = [process_tokens(tokens) for tokens in xi]  # xi = [["blahblah"]]
      cxi = [[list(xijk) for xijk in xij] for xij in xi]
      xp.append(xi[0])
      cxp.append(cxi[0])
      pp.append(context)
    

    # Only "+= 1" because every sets of support_docs corresponds to only 1 question.
    # In SQuAD, every paragraph can have multiple (len(para['qas'])) questions.
    for xij in xi:  # for sentence in context
      for xijk in xij:  # for word in sentence
        # if xijk == '.':
        #   print(xijk)
        word_counter[xijk] += 1
        lower_word_counter[xijk.lower()] += 1
        for xijkl in xijk:
          char_counter[xijkl] += 1


    # query
    # get words
    qi = word_tokenize(query)
    qi = process_tokens(qi)
    cqi = [list(qij) for qij in qi]

    q2i = word_tokenize(query_sub)
    q2i = process_tokens(q2i)
    cq2i = [list(q2ij) for q2ij in q2i]
    
    # answer
    yi = []
    cyi = []

    candi, ccandi, candi_y = [], [], []
    candi_word_y = []
    candi_word = candidates
    
    cand_span.append(candi)
    ccand_span.append(ccandi)
    cand_span_y.append(candi_y)
    cand_word.append(candi_word)
    cand_word_y.append(candi_word_y)
    answer_text = answer

    tokenized_context = ' '.join(xp2[-1])
    if args.find_candidates:
      assert answer in candidates, (answer, candidates)
      candi_word_y.append(candidates.index(answer))
      candidates_spans, not_found, candidates_found, real_candidates_found = compute_candidate_spans(tokenized_context, candidates)
      cand_word_found.append(candidates_found)
      real_cand_word_found.append(real_candidates_found)
      mis_cand += (not_found > 0)
      for (start, stop) in candidates_spans:
        yi0, yi1 = get_word_span(tokenized_context, xi, start, stop)
        
        assert len(xi[yi0[0]]) > yi0[1]
        assert len(xi[yi1[0]]) >= yi1[1]
        w0 = xi[yi0[0]][yi0[1]]
        w1 = xi[yi1[0]][yi1[1]-1]
        i0 = get_word_idx(tokenized_context, xi, yi0)
        i1 = get_word_idx(tokenized_context, xi, (yi1[0], yi1[1]-1))
        cyi0 = start - i0
        cyi1 = stop - i1 - 1
        candi.append([yi0, yi1])
        ccandi.append([cyi0, cyi1])

      
    if answer == '':
      raise Exception("Answer is empty.")
    else:   
      answer_start, answer_stop = compute_answer_span(tokenized_context, answer) # Find first matching span
      if answer_start is None:
        yi.append([(0, 0), (0, 1)])
        cyi.append([0, 1])
        na.append(True)
        answer_doc_ids.append([0])
        answer_ids_in_doc.append([0])
      
      else:
        if args.find_candidates:  # If we found the answer span, then we must have found the same span in candidates
          assert (answer_start, answer_stop) in \
          (candidates_spans), (answer, candidates, answer_start, answer_stop, candidates_spans)
          ans_idx = candidates_spans.index((answer_start, answer_stop))
          candi_y.append(ans_idx)
        na.append(False)
        yi0, yi1 = get_word_span(tokenized_context, xi, answer_start, answer_stop)
        answer_doc_id, answer_id_in_doc = find_doc_with_answer(yi0[1], xi_len)

        if pre_answer_doc_id is not None:
          assert answer_doc_id < 3, (answer_doc_id)
        answer_doc_ids.append([answer_doc_id])
        answer_ids_in_doc.append([answer_id_in_doc])

        answer_spans = []
        answer_spans.append((answer_start, answer_stop))
        next_answer_start = answer_start
        next_answer_stop = answer_stop
        next_context = tokenized_context[answer_stop:]
        while True:
          next_answer_start, next_answer_stop = compute_answer_span(next_context, answer)
          next_context = next_context[next_answer_stop:]
          if next_answer_start is not None:
            answer_spans.append((next_answer_start + answer_spans[-1][1], next_answer_stop + answer_spans[-1][1]))
          else:
            break
          next_yi0, next_yi1 = get_word_span(tokenized_context, xi, next_answer_start + answer_spans[-2][1], next_answer_stop + answer_spans[-2][1])

          next_answer_doc_id, next_answer_id_in_doc = find_doc_with_answer(next_yi0[1], xi_len)
          
          answer_doc_ids[-1].append(next_answer_doc_id)
          answer_ids_in_doc[-1].append(next_answer_id_in_doc)

        assert len(xi[yi0[0]]) > yi0[1]
        assert len(xi[yi1[0]]) >= yi1[1]
        w0 = xi[yi0[0]][yi0[1]]
        w1 = xi[yi1[0]][yi1[1]-1]
        i0 = get_word_idx(tokenized_context, xi, yi0)
        i1 = get_word_idx(tokenized_context, xi, (yi1[0], yi1[1]-1))
        cyi0 = answer_start - i0
        cyi1 = answer_stop - i1 - 1
        if args.medhop:
          assert answer_text[0] == w0[cyi0], (answer_text[0], w0[cyi0].lower(), answer_text, w0, cyi0)
        else:
          assert answer_text[0] == w0[cyi0].lower(), (answer_text[0], w0[cyi0].lower(), answer_text, w0, cyi0)
        assert answer_text[-1] == w1[cyi1].lower()
        assert cyi0 < 32, (answer_text, w0)
        assert cyi1 < 32, (answer_text, w1)
        
        yi.append([yi0, yi1])
        cyi.append([cyi0, cyi1])

    for qij in qi:
      word_counter[qij] += 1
      lower_word_counter[qij.lower()] += 1
      for qijk in qij:
        char_counter[qijk] += 1

    q.append(qi)
    q2.append(q2i)
    cq.append(cqi)
    cq2.append(cq2i)
    y.append(yi)
    cy.append(cyi)
    ids.append(article['id'])
    answers.append(answer)
    bi += 1

    
  assert len(q) == len(na), (len(qa), len(na))
  assert len(q) == len(y), (len(q), len(y))
  assert len(q) == len(x), (len(q), len(x))
  assert len(q) == len(first_doc_ids), (len(q), len(first_doc_ids))
  assert len(q) == len(answer_doc_ids), (len(q), len(answer_doc_ids))
  # Get embedding map according to word_counter.
  word2vec_dict = get_word2vec(args, word_counter)
  lower_word2vec_dict = get_word2vec(args, lower_word_counter)

  # add context here
  """
  q/cq: query
  y: answer
  rx/rcx: index pairs: (article_no, paragraph_no)
  cy:
  idxs:
  ids:
  answerss:
  na:
  """

  if args.split_supports:
    if args.find_candidates:
      data = {'q': q, 'cq': cq, 'y': y, 'cy': cy, 'ids': ids, 'answers': answers, 'na': na, 'x': x, 'cx': cx, 'p': p, 'x2': x2, 'p2': p2, 'q2': q2, 'cq2': cq2, \
              'cand_span': cand_span, 'ccand_span': ccand_span, 'cand_span_y': cand_span_y, 'cand_word': cand_word, 'cand_word_y': cand_word_y, \
              'cand_word_found': cand_word_found, 'real_cand_word_found': real_cand_word_found, 'answer_doc_ids': answer_doc_ids, 'answer_ids_in_doc': answer_ids_in_doc, 'first_doc_ids': first_doc_ids}
      if args.rank_by_tfidf is False:
        assert len(topk_2layer_tfidf_docs) > 0
        data.update({'topk_2layer_tfidf_docs': topk_2layer_tfidf_docs})
    else:
      data = {'q': q, 'cq': cq, 'y': y, 'cy': cy, 'ids': ids, 'answers': answers, \
              'na': na, 'x': x, 'cx': cx, 'p': p, 'x2': x2, 'p2': p2, 'answer_doc_ids': answer_doc_ids, \
              'answer_ids_in_doc': answer_ids_in_doc, 'first_doc_ids': first_doc_ids}
  else:
    data = {'q': q, 'cq': cq, 'y': y, 'cy': cy, 'ids': ids, 'answers': answers, \
            'na': na, 'x': x, 'cx': cx, 'p': p}
  """
  x/cx: tokenized paragraphs (words and chars)
  p: untokenized paragraphs
  """
  shared = {'word_counter': word_counter, 'char_counter': char_counter, \
    'lower_word_counter': lower_word_counter, 'word2vec': word2vec_dict, \
    'lower_word2vec': lower_word2vec_dict}

  print("saving ...")
  print("no answer: %d" %sum(na))
  print("missing candidates: %d" %mis_cand)
  if save_json:
    save(args, data, shared, out_name)
  else:
    prepro_each(args, data_type, out_name=out_name, pre_answer_doc_id=answer_doc_ids)

if __name__ == "__main__":
  main()
