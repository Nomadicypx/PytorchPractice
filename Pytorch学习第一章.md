# Pytorch学习第一章

> 数据操作

| 函数                              | 功能                      |
| --------------------------------- | ------------------------- |
| Tensor(*sizes)                    | 基础构造函数              |
| tensor(data,)                     | 类似np.array的构造函数    |
| ones(*sizes)                      | 全1Tensor                 |
| zeros(*sizes)                     | 全0Tensor                 |
| eye(*sizes)                       | 对角线为1，其他为0        |
| arange(s,e,step)                  | 从s到e，步长为step        |
| linspace(s,e,steps)               | 从s到e，均匀切分成steps份 |
| rand/randn(*sizes)                | 均匀/标准分布             |
| normal(mean,std)/uniform(from,to) | 正态分布/均匀分布         |
| randperm(m)                       | 随机排列                  |

* 这些创建方法都可以在创建的时候指定数据类型dtype和存放device(cpu/gpu)。\

* 用方法`to()`可以将`Tensor`在CPU和GPU（需要硬件支持）之间相互移动。

  ```python
  # 以下代码只有在PyTorch GPU版本上才会执行
  if torch.cuda.is_available():
      device = torch.device("cuda")          # GPU
      y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
      x = x.to(device)                       # 等价于 .to("cuda")
      z = x + y
      print(z)
      print(z.to("cpu", torch.double))       # to()还可以同时更改数据类型
  ```

* tensor的加减乘除
* tensor的复制 x.clone(),使用clone时梯度回传到副本时也会传到源`Tensor`
* 另外一个常用的函数就是`item()`, 它可以将一个标量`Tensor`转换成一个Python number
* 矩阵操作
  * mm/bmm是矩阵乘法/batch的矩阵乘法
  * dot是内积,cross是外积
  * inverse求逆矩阵
  * svd求奇异值分解
* 内存开销：
  * 矩阵的加法千万别写为x = x+y的形式，这样会开辟新的内存空间
  * 写成torch.add(x,y,out=y)或者y+=x或者y[:] = x+y的形式，可以节省内存空间
* 自动求梯度：`Tensor`是这个包的核心类，如果将其属性`.requires_grad`设置为`True`，它将开始追踪(track)在其上的所有操作（这样就可以利用链式法则进行梯度传播了）。完成计算后，可以调用`.backward()`来完成所有梯度计算。此`Tensor`的梯度将累积到`.grad`属性中（如果不想要被继续追踪，可以调用`.detach()`将其从追踪记录中分离出来，这样就可以防止将来的计算被追踪，这样梯度就传不过去了，此外，还可以用`with torch.no_grad()`将不想被追踪的操作代码块包裹起来，这种方法在评估模型的时候很常用，因为在评估模型时，我们并不需要计算可训练参数（`requires_grad=True`）的梯度）
  * 