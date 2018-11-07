# TensorFlow-Transformer
Attention Is All You Need

## Paper
   * Attention Is All You Need: https://arxiv.org/abs/1706.03762
   * Layer Normalization: https://arxiv.org/abs/1607.06450
   * Label Smoothing: https://arxiv.org/abs/1512.00567 
   * Byte-Pair Encoding (BPE): https://arxiv.org/abs/1508.07909  

## Dataset
   * Preprocessed WMT17 en-de: http://data.statmt.org/wmt17/translation-task/preprocessed/  
      * Source: en
      * Target: de
      * train_set: corpus.tc (I used only 500,000 line)
      * test_set: dev/newstest
   * [Sentences were encoded using byte-pair encoding](https://github.com/SeonbeomKim/Python-Bype_Pair_Encoding)
      * MakeFile: 
         * bpe_applied_data
         * bpe2idx.npy 
         * idx2bpe.npy 
         * cache.npy
         * merge_info.npy
         * word_frequency_dictionary.npy 
      
      
## Reference
   * https://jalammar.github.io/illustrated-transformer/
   * https://github.com/Kyubyong/transformer
