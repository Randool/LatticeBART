### ~~Lattice上的数值都是负对数吗？~~

`start end token lmwt,acwt,xxx`中`acwt`是否太小了

```
BAC009S0764W0121
0 1 甚至 15.3741,4113.52,3_12_10_10_10_10_... 
1 2 出现 14.4536,3987.55,1265_1265_1265_1265_... 
2 3 交易 7.5916,1230.09,4407_4407_4407_4407_... 
3 6 几乎 10.5667,2188.89,5725_5725_5788_5787_... 
3 4 机 9.06609,1102.71,5725_5725_5788_5787_... 
```

```
1272-128104-0001
0 1 NOR 11.351,2892.95,4_1_1_1_1_1_16_... 
1 3 IS 11.4775,2374.59,14467_14518_14566_... 
1 2 AS 13.7006,2466.26,14467_14518_14566_... 
2 76 MISTER 11.1646,1711.18,63261_63394_... 
3 4 MISTER 9.56968,1758.29,42727_63154_... 
4 5 CALDER'S 16.8403,2194.07,29884_29883_... 
4 6 CALLED 8.30452,637.335,29884_29883_...
```

----
### lattice上的一些参数不清楚

比如`steps/decode_fmllr.sh`的默认参数：
```bash
first_beam=10
beam=13
lattice_beam=6
```

----
### `lattice-prune`和`lattice-determinize-pruned`有啥区别呀？

kaldi关于lattice的工具有：
```
lattice-1best
lattice-add-penalty
lattice-best-path
lattice-depth
lattice-depth-per-frame
lattice-determinize
lattice-determinize-pruned
lattice-determinize-pruned-parallel
lattice-difference
lattice-equivalent
lattice-expand-ngram
lattice-interp
lattice-limit-depth
lattice-minimize
lattice-prune
lattice-push
lattice-reverse
lattice-rmali
lattice-scale
```

其中和剪枝相关的有`lattice-prune`和`lattice-determinize-pruned`

`lattice-prune`参数：

```
Apply beam pruning to lattices
Usage: lattice-prune [options] lattice-rspecifier lattice-wspecifier
 e.g.: lattice-prune --acoustic-scale=0.1 --beam=4.0 ark:1.lats ark:pruned.lats

Options:
  --acoustic-scale            : Scaling factor for acoustic likelihoods (float, default = 1)
  --beam                      : Pruning beam [applied after acoustic scaling] (float, default = 10)
  --inv-acoustic-scale        : An alternative way of setting the acoustic scale: you can set its inverse. (float, default = 1)

Standard options:
  --config                    : Configuration file to read (this option may be repeated) (string, default = "")
  --help                      : Print out usage message (bool, default = false)
  --print-args                : Print the command line arguments (to stderr) (bool, default = true)
  --verbose                   : Verbose level (higher->more logging) (int, default = 0)
```

`lattice-determinize-pruned`参数：

```
Determinize lattices, keeping only the best path (sequence of acoustic states)
for each input-symbol sequence.  This version does pruning as part of the
determinization algorithm, which is more efficient and prevents blowup.
See http://kaldi-asr.org/doc/lattices.html for more information on lattices.

Usage: lattice-determinize-pruned [options] lattice-rspecifier lattice-wspecifier
 e.g.: lattice-determinize-pruned --acoustic-scale=0.1 --beam=6.0 ark:in.lats ark:det.lats

Options:
  --acoustic-scale            : Scaling factor for acoustic likelihoods (float, default = 1)
  --beam                      : Pruning beam [applied after acoustic scaling]. (float, default = 10)
  --delta                     : Tolerance used in determinization (float, default = 0.000976562)
  --max-arcs                  : Maximum number of arcs in output FST (total, not per state (int, default = -1)                                                                                                   
  --max-loop                  : Option used to detect a particular type of determinization failure, typically due to invalid input (e.g., negative-cost loops) (int, default = 0)                                
  --max-mem                   : Maximum approximate memory usage in determinization (real usage might be many times this) (int, default = 50000000)                                                              
  --max-states                : Maximum number of arcs in output FST (total, not per state (int, default = -1)                                                                                                   
  --minimize                  : If true, push and minimize after determinization (bool, default = false)
  --retry-cutoff              : Controls pruning un-determinized lattice and retrying determinization: if effective-beam < retry-cutoff * beam, we prune the raw lattice and retry.  Avoids ever getting empty output for long segments. (float, default = 0.5)
  --write-compact             : If true, write in normal (compact) form. --write-compact=false allows you to retain frame-level acoustic score information, but this requires the input to be in non-compact form
e.g. undeterminized lattice straight from decoding. (bool, default = true)

Standard options:
  --config                    : Configuration file to read (this option may be repeated) (string, default = "")                                                                                                  
  --help                      : Print out usage message (bool, default = false)
  --print-args                : Print the command line arguments (to stderr) (bool, default = true)
  --verbose                   : Verbose level (higher->more logging) (int, default = 0)
```

* ~~这里的`beam`和上面提到的`beam`是什么关系呢？~~
* `inv-acoustic-scale`干什么用的？
* 如果我需要限制剪枝之后的节点个数（<200），那么该怎么做？
* 什么是确定化（determinize）：从任意状态出发跳转的输入标签各不相同
* 什么是最小化（minimize）：


----
### `lat.*.gz`转`lat.*.txt`

```bash
lattice-copy 'ark:gunzip -c lat.1.gz |' ark,t:- | int2sym.pl -f 3 ../graph_tgsmall/words.txt > lat.1.prune.txt
lattice-prune --acoustic-scale=0.08333 --beam=6.0
lattice-determinize-pruned --acoustic-scale=0.08333 --beam=6.0 
```

----
### kaldi的`run.sh`在每个tri中都会在`exp/tri_x`下生成`lat.*.gz`，那么如何对训练集做类似的处理呢？是不是把`test`换成`train_*`就好了？

```bash
for test in test_clean test_other dev_clean dev_other; do
  steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
    exp/tri6b/graph_tgsmall data/$test exp/tri6b/decode_tgsmall_$test
done
```

直接替换会出现维度不匹配的错误
```bash
# gmm-latgen-faster --max-active=2000 --beam=10.0 --lattice-beam=6.0 --acoustic-scale=0.083333 --allow-partial=true --word-symbol-table=exp/tri6b/graph_tgsmall/words.txt exp/train/final.alimdl exp/tri6b/graph_tgsmall/HCLG.fst "ark,s,cs:apply-cmvn  --utt2spk=ark:data/train_2kshort/split20/1/utt2spk scp:data/train_2kshort/split20/1/cmvn.scp scp:data/train_2kshort/split20/1/feats.scp ark:- | add-deltas  ark:- ark:- |" "ark:|gzip -c > exp/train/train_2kshort_tgsmall.si/lat.1.gz" 
# Started at Fri Jul 16 15:49:48 CST 2021
#
gmm-latgen-faster --max-active=2000 --beam=10.0 --lattice-beam=6.0 --acoustic-scale=0.083333 --allow-partial=true --word-symbol-table=exp/tri6b/graph_tgsmall/words.txt exp/train/final.alimdl exp/tri6b/graph_tgsmall/HCLG.fst 'ark,s,cs:apply-cmvn  --utt2spk=ark:data/train_2kshort/split20/1/utt2spk scp:data/train_2kshort/split20/1/cmvn.scp scp:data/train_2kshort/split20/1/feats.scp ark:- | add-deltas  ark:- ark:- |' 'ark:|gzip -c > exp/train/train_2kshort_tgsmall.si/lat.1.gz' 
apply-cmvn --utt2spk=ark:data/train_2kshort/split20/1/utt2spk scp:data/train_2kshort/split20/1/cmvn.scp scp:data/train_2kshort/split20/1/feats.scp ark:- 
add-deltas ark:- ark:- 
ERROR (gmm-latgen-faster[5.5.0~1-5e6f]:LogLikelihoodZeroBased():decodable-am-diag-gmm.cc:50) Dim mismatch: data dim = 39 vs. model dim = 40
```


### 使用哪些数据集？


### 准备哪些实验

由于先训练Enc会导致Enc输出分布发生改变，最终损害LatticeBart的性能（除非将Enc的训练和**Enc-Dec联合训练**结合在一起）

这里无法体现对比学习对Enc帮助的主要原因在于Bart本身的Enc与Dec是高度耦合的，单独训练任意一个部分都会破坏其完整性。

所以不妨先随机初始化Enc和Dec，然后预训练Enc（也可同时训练Dec）

