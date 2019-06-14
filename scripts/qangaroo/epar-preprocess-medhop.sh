python3 -m qangaroo.prepro --split_supports --find_candidates --medhop --glove_corpus="840B" --glove_vec_size=300
cp data/qangaroo/medhop/split-supports/w-candi/data_test.json data/qangaroo/medhop/split-supports/w-candi/data_dev.json
cp data/qangaroo/medhop/split-supports/w-candi/shared_test.json data/qangaroo/medhop/split-supports/w-candi/shared_dev.json