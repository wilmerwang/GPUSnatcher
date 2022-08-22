# scramble4gpu

scramble4gpu是用来在抢占显卡的脚本，主要是在实验室显卡资源紧张且自己亟需使用显卡的情况下，使用该脚本自动抢占一个或多个显卡。

**建议将scramble4gpu.py更改为train.py；防止被打...**

## 依赖

- numpy
- torch or tensorflow

## 使用

```shell
git clone https://github.com/ilikewind/scramble4gpu.git
cd scramble4gpu
python scramble4gpu.py
```

### 可选参数

![](doc\optional_arg.png)

- -p --proportion 显卡空闲内存 / 全部内存 的阈值，取值在0-1之间。当p取1的时候，表示仅仅列出完全没有被使用的显卡。默认为0.8。
- -n --gpu_nums 需要抢占的GPU数量，建议不要抢太多，容易挨揍。默认是1。
- -t --times 抢占显卡之后，自动释放显卡的时间。默认是1小时。

当想自己设置以上参数的时候：

```shell
# 查找Free显存大于0.9的显卡，抢占4个，1800秒后自动释放
python scramble4gpu.py -p 0.9 -n 4 -t 1800
```
