## 机器学习,深入算法底层

- `src/`, c++实现逻辑回归，主要源码是`lr.cc`与`utils.cc`．`python_wrapper.cc`实现了一些辅助函数，暴露C风格接口给python
- `python-package`，通过`ctypes`实现python调用C函数，`lr/model.py`封装了相关函数，`example.py`是具体的实例


### 使用方法

- 编译得到动态链接库`liblr.so`

```
g++ -fPIC -shared -o liblr.so python_wrapper.cc lr.cc utils.cc
```

- 复制到相应文件夹下，`cp liblr.so python-package/lr/`

- 运行　`example.py`

