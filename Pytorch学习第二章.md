# Pytorch学习第二章

> 这部分内容还是进行数据的线性拟合，训练数据的生成这次我全是自己写的，哈哈，熟练了一点点

* 需要注意些什么呢？
  * 我们需要使用nn.Linear模块来进行函数的绘制
  * 我们需要使用nn.MSELoss()函数来定义loss公式
  * 我们需要使用init工具包里的init.normal_和init.constant\_分别对模型的weight和bias进行初始化
  * 我们需要使用optim类下面的优化器来定义优化算法
  * 我们需要使用torch.utils.data.TensorDataset(features,labels)塞入数据，使用torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)来minibatch

* 模型定义代码如下：

```python
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)
    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net) # 使用print可以打印出网络的结构
```

* 模型训练代码如下:

```python
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
```

这里注意一下，minibatch同样是使用增强循环实现的，data_iter就是一个迭代器，此外参数的梯度清零，原来的方法是w.grad.data.zero_()现在变成了 optimizer.zero\_grad()，optimizer.step()相当于以前的sgd部分，用于更新参数