# TensorFlow-Transformer
Attention Is All You Need

구현중

## Paper
   * Attention Is All You Need: https://arxiv.org/abs/1706.03762
   * Layer Normalization: https://arxiv.org/abs/1607.06450
   * Label Smoothing: https://arxiv.org/abs/1512.00567 
   * Byte-Pair Encoding (BPE): https://arxiv.org/abs/1508.07909  

## Dataset
   * Preprocessed WMT17 en-de: http://data.statmt.org/wmt17/translation-task/preprocessed/  
      * Source: en
      * Target: de
      * train_set: corpus.tc
      * test_set: dev/newstest 
       
   * [Sentences were encoded using byte-pair encoding](https://github.com/SeonbeomKim/Python-Bype_Pair_Encoding)
      * MakeFile: 
         * bpe_applied_data
         * bpe2idx.npy 
         * idx2bpe.npy 
         * cache.npy
         * merge_info.npy

## Code
   * Inference_utils.py
      * greedy
      * beam-search
      * bleu (nltk)
         
   * Transformer.py
      * Transformer implement
     
   * bucket_data_helper.py
      * bucket으로 구성된 데이터를 쉽게 가져오도록 하는 class
      
   * make_dataset.py
      * generate concatenated(source||target) and bucketed data (train, valid dataset)
      * need MakeFile of [Sentences were encoded using byte-pair encoding](https://github.com/SeonbeomKim/Python-Bype_Pair_Encoding) 
      * MakeFile: 
         * bpe_dataset/source_idx_wmt17_en.csv
         * bpe_dataset/target_idx_wmt17_de.csv
         * bpe_dataset/source_idx_newstest2014_en.csv
         * bpe_dataset/source_idx_newstest2015_en.csv
         * bpe_dataset/source_idx_newstest2016_en.csv
         * bpe_dataset/train_set/bucket_data(source, target).csv
         * bpe_dataset/valid_set/bucket_data(source, target).csv
         * bpe_dataset/test_set/bucket_data(source, target).csv
         
   * translation_train.py
     * WMT17 en-de train, validation, test

## Training
   1. [WMT17 Dataset Download](http://data.statmt.org/wmt17/translation-task/preprocessed/)  
   2. [Sentences were encoded using byte-pair encoding](https://github.com/SeonbeomKim/Python-Bype_Pair_Encoding) apply
   3. make_dataset.py
   4. translation_train.py

## Reference
   * https://jalammar.github.io/illustrated-transformer/
   * https://github.com/Kyubyong/transformer
