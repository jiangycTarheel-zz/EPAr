# EPAr

## Explore, Propose, and Assemble: An Interpretable Model for Multi-Hop Reading Comprehension 
* Official code for our [ACL 2019 paper](https://arxiv.org/pdf/1906.05210.pdf).
* The initial code was adapted from [BiDAF](https://github.com/allenai/bi-att-flow).

### 1. Dependencies
* We tested our code on TF1.3, TF1.8, TF1.11 and TF1.13.
* See 'requirements.txt'.

### 2. Data:
#### 2.1 Qangaroo:
Download data from http://qangaroo.cs.ucl.ac.uk/index.html and put the uncompressed folder in ~/data directory.

#### 2.2 GloVe Embeddings:
For our main model download the glove.840B.300d.zip word vectors from https://nlp.stanford.edu/projects/glove/ and place it ~/data/glove/. For our smaller model (which we use throughout analysis) download glove.6B.zip from the same link.


### 3. WikiHop:
Here we show how to run our full-size EPAr model with 300-d GloVe embeddings and 100-d LSTM hidden size.

#### 3.1 Data preprocessing:
Run the following script:
```
./scripts/qangaroo/epar-preprocess-with-tfidf.sh 
```
In order to run a small model with 100-d word embeddings and 20-d hidden size, delete these 2 options from the preprocessing scripts: --glove_corpus="840B", --glove_vec_size=300, and delete these 3 options from the training/testing script: --emb_dim=300, --hidden_size=100, cudnn_rnn=True. In addition to that we train our smaller model in 2 stages, first without using the Assembler (refer to oracle-epar-train.sh) and then all the 3 modules jointly using main script (full-epar-train.sh).

#### 3.2 Train:
To train the full 3-module system: 
```
./scripts/qangaroo/full-epar-train.sh 
```
for around 40k iterations. Change the --run_id in our scripts to train different models.

Note: The WikiHop scripts above are designed for multi-gpu setting . Change the num_gpus (and then the batch_size) accordingly. In the provided training scripts, we use 2 gpus and batch size of 5. For training a small model, we recommend 1 gpu and batch size of 10.

#### 3.3 Test:
Run: 
```
./scripts/qangaroo/full-epar-test.sh
```
The model checkpoints are saved in out/basic/qangaroo/[RUN_ID]/save/. 

### 4 MedHop (non-masked):
#### 4.1 Data preprocessing:
Run the following script: 
```
./scripts/qangaroo/epar-preprocess-medhop.sh
```

#### 4.2 Train:
To train full model run this script (should converge in close to 3k iterations):
```
./scripts/qangaroo/full-epar-train-medhop.sh
```
#### 4.3 Test:
To evaluate the trained model on dev set, run the following script:
```
./scripts/qangaroo/full-epar-test-medhop.sh
```
### 5 Pretrained Models
We release our pretrained models for WikiHop and MedHop [here](https://drive.google.com/open?id=1Gz8TVc6adelGD0w8kQvrfY7WBioniBho). On running the testing script on these models, you should get 67.2% accuracy on the WikiHop dev set, and 64.9% accuracy on the MedHop dev set.


# Citation
```
@inproceedings{JiangJoshi2019epar, 
  author={Yichen Jiang, Nitish Joshi, Yen-chun Chen and Mohit Bansal}, 
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics}, 
  title={Explore, Propose, and Assemble: An Interpretable Model for Multi-Hop Reading Comprehension}, 
  year={2019}
}
```




