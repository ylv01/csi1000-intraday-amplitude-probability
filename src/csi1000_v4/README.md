# csi1000_v4内部包说明

`csi1000_v4`是中证1000当天振幅动态概率模型的项目内部Python包，不是第三方库，无须单独安装。

## 模块

### `feature_pipeline.py`

负责：

- 中证1000、上证50、沪深300日线和1分钟线读取；
- OHLC、重复值、缺失值和时间戳审计；
- 分钟级累计振幅、收益率、波动率等计算；
- 每5分钟预测特征快照构建；
- 历史滞后因子和跨指数因子构建。

### `final_pipeline.py`

负责：

- 盘前、开盘首小时、首小时后三个路由；
- P10至P90九个LightGBM条件分位数模型；
- 三窗口树数选择；
- 完整训练、模型保存和批量预测；
- 分位数单调处理和指定振幅超越概率计算；
- 分路由及总体评价。

### `__init__.py`

对外暴露主要训练和预测接口：

```python
from csi1000_v4 import predict_feature_rows, run_training
```

## 加载方式

`scripts`目录中的入口程序会自动把V4内部的`src`加入Python搜索路径，然后导入本包。例如：

```python
from csi1000_v4.feature_pipeline import audit_inputs
```

外部环境只需安装项目根目录`requirements.txt`列出的LightGBM、pandas、NumPy、scikit-learn和joblib。
