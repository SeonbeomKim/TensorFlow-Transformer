# TensorFlow-Transformer
Attention Is All You Need


## Translate EN into DE (Train with WMT17)
   * newstest2014 BLEU: 28.51
   * newstest2015 BLEU: 30.22
   * newstest2016 BLEU: 33.88
![final.PNG](./result_img/final.PNG)


## Paper
   * Attention Is All You Need: https://arxiv.org/abs/1706.03762
   * Layer Normalization: https://arxiv.org/abs/1607.06450
   * Label Smoothing: https://arxiv.org/abs/1512.00567 
   * Byte-Pair Encoding (BPE): https://arxiv.org/abs/1508.07909  
   * Beam-Search length penalty: https://arxiv.org/abs/1609.08144

## Env
   * GTX1080TI
   * ubuntu 16.04
   * CUDA 8.0
   * cuDNN 5.1
   * tensorflow 1.4
   * numpy
   * nltk (bleu)
   * tqdm (iteration check bar)
   * python 3
   


## Dataset
   * Preprocessed WMT17 en-de: http://data.statmt.org/wmt17/translation-task/preprocessed/ 
      * train_set: corpus.tc.[en, de]/corpus.tc.[en, de]
      * dev_set: dev.tar/newstest[2014, 2015, 2016].tc.[en, de]
       
   * learn and apply [Sentences were encoded using byte-pair encoding](https://github.com/SeonbeomKim/Python-Bype_Pair_Encoding)
      * -num_merges: 35000
      * -final_voca_threshold: 50    
      * -train_voca_threshold: 1
      * make_file: bpe applied documents and voca
      
## Code
   * transformer.py
      * Transformer graph

   * inference_helper.py
      * greedy
      * beam (length penalty applied)
      * bleu (nltk)
              
   * bucket_data_helper.py
      * bucket으로 구성된 데이터를 쉽게 가져오도록 하는 class
      
   * make_dataset.py
      * generate bucketed bpe2idx dataset for train, valid, test from bpe applied dataset
      * need MakeFile of [Sentences were encoded using byte-pair encoding](https://github.com/SeonbeomKim/Python-Bype_Pair_Encoding) 
      * command: 
         * make bucket train_set wmt17
         ```
          python make_dataset.py 
            -mode train 
            -source_input_path path/bpe_wmt17.en (source bpe applied document data)
            -source_out_path path/source_idx_wmt17_en.csv (source bpe idx data)
            -target_input_path path/bpe_wmt17.de (target bpe applied document data)
            -target_out_path path/source_idx_wmt17_de.csv (target bpe idx data)
            -bucket_out_path ./bpe_dataset/train_set_wmt17 (bucket trainset from source bpe idx data, target bpe idx data)
            -voca_path voca_path/voca_file_name (bpe voca from bpe_learn.py)
         ```
         * make bucket valid_set newstest2014
         ```
          python make_dataset.py 
            -mode infer 
            -source_input_path path/bpe_newstest2014.en (source bpe applied document data)
            -source_out_path path/source_idx_newstest2014_en.csv (source bpe idx data)
            -target_input_path path/dev.tar/newstest2014.tc.de (target original raw data)
            -bucket_out_path ./bpe_dataset/valid_set_newstest2014 (bucket validset from source bpe idx data, target original raw data)
            -voca_path voca_path/voca_file_name (bpe voca from bpe_learn.py)
         ```
         * make bucket test_set newstest2015
         ```
          python make_dataset.py 
            -mode infer 
            -source_input_path path/bpe_newstest2015.en (source bpe applied document data)
            -source_out_path path/source_idx_newstest2015_en.csv (source bpe idx data)
            -target_input_path path/dev.tar/newstest2015.tc.de (target original raw data)
            -bucket_out_path ./bpe_dataset/test_set_newstest2015 (bucket testset from source bpe idx data, target original raw data)
            -voca_path voca_path/voca_file_name (bpe voca from bpe_learn.py)
         ```
         * make bucket test_set newstest2016
         ```
          python make_dataset.py 
            -mode infer 
            -source_input_path path/bpe_newstest2016.en (source bpe applied document data)
            -source_out_path path/source_idx_newstest2016_en.csv (source bpe idx data)
            -target_input_path path/dev.tar/newstest2016.tc.de (target original raw data)
            -bucket_out_path ./bpe_dataset/test_set_newstest2016 (bucket testset from source bpe idx data, target original raw data)
            -voca_path voca_path/voca_file_name (bpe voca from bpe_learn.py)
         ```
   * translation_train.py
     * en -> de translation train, validation, test
     * command
       ```
        python translation_train.py 
          -train_path_2017 ./bpe_dataset/train_set_wmt17 
          -valid_path_2014 ./bpe_dataset/valid_set_newstest2014 
          -test_path_2015 ./bpe_dataset/test_set_newstest2015 
          -test_path_2016 ./bpe_dataset/test_set_newstest2016 
          -voca_path voca_path/voca_file_name
       ```
       
## Training
   1. [WMT17 Dataset Download](http://data.statmt.org/wmt17/translation-task/preprocessed/)  
   2. [apply byte-pair_encoding](https://github.com/SeonbeomKim/Python-Bype_Pair_Encoding)
   3. run make_dataset.py
   4. run translation_train.py

## Reference
   * https://jalammar.github.io/illustrated-transformer/
   * https://github.com/Kyubyong/transformer
