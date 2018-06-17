## RTFSC

两年前甚至是三年前开始，你会发现越来越多的人转行做算法，业界也很给力，源源不断地发布各种数据挖掘类型的比赛，成为很多人从入门到实践的第一课．如果说学术的作用是推动算法创新，那么此类比赛的作用便是创新落地，以及检验那些在实践中真正work的东西．然而，实际上我发现这样的比赛很少，大多数赛题其实只是官方宣传自己的一种手段，题目类型非常陈旧，以致于参赛选手只需要`import xgboost as xgb`就行了．从我过去一两年的参赛经验来看，`import xgboost as xgb`的确是很有效的，而从头改算法造轮子最终都是劳而无功或者收效甚微．如果你赞同这个观点的话，右上角`star`一下．

唠叨这些跟这个repo有什么关系？　关注我的人里面，有不少是因为看到我以前的一些参赛代码，可能大部分都是在校生，可能现在正在参加某个比赛．我想给一些小小的个人建议，不要日复一日地重复`import xgboost as xgb`或者`import lightgbm as lgb`，做一些门槛更高的东西，比如学术里的前沿算法，比如工程上的高效实现．机器学习涉及到的领域很多很多，你我还需要不断学习，就不要重复地去写`import xxx`了．

这个小项目最初是为了在毕业离校前做一个简单的组内分享，科普一下机器学习算法包的实现流程．现在打算开源，对很多入门的朋友或许有帮助，但因为懒没有写出完整文档，感兴趣的朋友只能将就读代码了，相信Linus，文档是最好的代码.

## 机器学习算法的底层实现与高层调用

以最简单的机器学习算法逻辑回归为例，介绍底层C++实现，以及高层Python调用，掌握ctypes基本用法．

## 源码说明

- `src/`, c++实现逻辑回归，主要源码是`lr.cc`与`utils.cc`．`python_wrapper.cc`实现了一些辅助函数，暴露C风格接口给python
- `python-package`，通过`ctypes`实现python调用C函数，`lr/model.py`封装了相关函数，`example.py`是具体的实例

## 依赖

- Eigen

### 使用方法

- 编译得到动态链接库`liblr.so`

```
g++ -fPIC -shared -fopenmp -o liblr.so python_wrapper.cc lr.cc utils.cc
```

- 复制到相应文件夹下，`cp liblr.so python-package/lr/`

- 运行　`python example.py`

```python

from lr import model
import numpy as np

# custom metric function, mean accuracy
def mean_accuracy(label,pred,size):
    num_pos,hit_pos = 0.0,0.0
    num_neg,hit_neg = 0.0,0.0
    for i in range(size):
        if label[i]==1.0:
            num_pos += 1.0
            if pred[i]>0.5:
                hit_pos += 1.0

        if label[i]==0.0:
            num_neg += 1.0
            if pred[i]<=0.5:
                hit_neg += 1.0
    print "pos-accracy:{0:.5f},neg-accuracy:{1:.5f}".format(hit_pos/num_pos,hit_neg/num_neg)
    return 0.5*hit_pos/num_pos + 0.5*hit_neg/num_neg


features = np.load('features.dat')
labels = np.load('labels.dat')
print features.shape,labels.shape

clf = model(max_iter=1000,alpha=0.01,l2_lambda=0.5,tolerance=0.01)
clf.fit(features,labels,batch_size=1024,early_stopping_round=100,metric=mean_accuracy)
print clf.predict(features[:30])

clf.save("/home/wepon/lr.model")
clf1 = model()
clf1.load("/home/wepon/lr.model")
print clf1.predict(features[:30])

```

