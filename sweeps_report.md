# 总体思路

待调的超参为：

```

模型架构：
dim_z,dim_g,num_channels

缩放因子：
binary_factor_scaling,ternary_factor_scaling,classifier_amplifier

学习率：
learning_rate

正则化：
regularize_z,regularize_g,regularize_h

```

考虑到scaling laws对应了多种不同大小的模型，所以计划先在小模型上调参。具体调参方式为：

参数搜索不枚举绝对数值，而是枚举与$dim_z$的比例或幂律关系。在小模型上找到性能较好的幂律关系后，在较大规模模型上进行验证。

# 调参阶段

### 第一阶段：模型架构的调整

方法：在一个相对合理的学习率、正则化参数下，固定同一模型参数量，调整dim_g/dim_z的比值（如4~7）和num_channels的数值（如6~12）

结论：
* 模型架构中，dim_g/dim_z对loss的影响略微显著，num_channels对loss影响较小（增加后也没有明显的提升），但对运行时间影响非常显著。* 考虑取dim_g/dim_z=5.5（几组较好结果普遍接近此数值），num_channels=6

### 第二阶段：缩放因子的调整

方法：确定模型架构、固定相对合理的学习率、正则化参数，改变binary_factor_scaling、ternary_factor_scaling、classifier_exp的幂律关系。即：ternary_factor_scaling = 1 / dim_z^exp，classifier_amplifier = dim_z^exp等。待调参数即为exp。

结论：
* binary_factor_scaling对loss基本没有太大影响，考虑固定为0.5
* ternary_factor_scaling的exp较好情况为0.385左右
* classifier_amplifier的exp较好情况为0.45左右

### 第三阶段：学习率的调整

方法：
* 确定模型架构与缩放因子后，令学习率为：lr = lr_eta0 / dim_z^lr_exp。（因为loss对学习率的变化较为敏感，考虑不仅设置指数关系也设置常数eta0）
* 考虑在27M、48M、80M等较小模型上都做一些学习率调参实验，拟合出相关的规律

目前还在进行中，观察出的一些现象为：
* 27M模型中，lr=0.003时较好
* 48M模型中，lr=0.004时较好
* 80M模型中，lr=0.006时较好

但采样点过少（仅尝试了3种大小），可能还需要进一步实验观察更普遍的规律

### 第四阶段：正则化调整

方法：
* regularize_z：1/dim_z^\b，b较小，考虑设为0.2~0.8
* regularize_g：1/dim_z^b，b稍大，考虑设为0.3~0.9
* regularize_h：1/dim_z^b，b最大，考虑设为1~1.7
（以上规律从早期的调参实验中大致观测得出）

暂未开始调试。

### 第五阶段：scaling验证

* 得到PT的超参幂律关系后，扩展模型规模验证训练效果
* 与Transformer的Scaling情况对比