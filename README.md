# GPUSnatcher

GPUSnatcher是用来在抢占显卡的脚本，主要是在实验室显卡资源紧张且自己亟需使用显卡的情况下，使用该脚本自动抢占一个或多个显卡。

**建议将scramble4gpu.py更改为train.py，并设置仅自己可见，防止被打...**
**请勿恶意抢占!!**

## 依赖

- numpy
- torch or tensorflow

## 使用

- **请先配置Email，否则当抢占GPU之后，不能发送Email**
```
git clone https://github.com/wilmerwang/GPUSnatcher.git
cd GPUSnatcher

# ./email_conf.json
# 建议使用qq邮箱服务,如果用的其他邮箱服务器，请自行设置
{
  "host": "smtp.qq.com",  # qq邮箱server
  "user": "2xxxxxxx6@qq.com",  # 要登陆的qq账号
  "pwd": "xxxxxxxxxxxxxxxx",  # SMTP授权码, qq邮箱--> 设置 --> 账号 --> IMAP/SMTP服务开启 --> 生成授权码
  "sender": "2xxxxxxx6@qq.com",  # 发送者
  "receiver": "2xxxxxxx6@qq.com"  # 接收邮箱,可以是列表比如["a@qq.com", "b@qq.com"]
}
```

- 配置之后运行程序
```shell
python scramble4gpu.py
```

### 可选参数

- -p --proportion 显卡空闲内存 / 全部内存 的阈值，取值在0-1之间。当p取1的时候，表示仅仅列出完全没有被使用的显卡。默认为0.8。
- -n --gpu_nums 需要抢占的GPU数量，建议不要抢太多，容易挨揍。默认是1。
- -t --times 抢占显卡之后，自动释放显卡的时间。默认是30分钟。
- -e --email_conf email的配置参数，默认在./email_conf.json

当想自己设置以上参数的时候：

```shell
# 查看参数详情
python scramble4gpu.py -h

# 查找Free显存大于0.9的显卡，抢占4个，1800秒后自动释放,email配置路径为./email_conf.json
python scramble4gpu.py -p 0.9 -n 4 -t 1800 -e ./email_conf.json
```
