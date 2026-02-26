mnist-mlp-project/
│
├── .github/                   # GitHub 专用配置（如 issue/pr 模板）
├── configs/                   # 配置文件（YAML/JSON）
│   └── config.yaml
├── data/                      # 数据集（通常被 .gitignore 忽略）
│   ├── raw/                    # 原始数据
│   └── processed/              # 预处理后的数据
├── docs/                       # 项目文档（可选）
├── models/                     # 保存训练好的模型
│   └── mnist_mlp.h5
├── notebooks/                  # 探索性分析与报告（Jupyter）
│   └── 01_eda_and_results.ipynb
├── reports/                    # 生成的报告（HTML/PDF）
│   └── figures/                # 图表
├── src/                         # 源代码
│   ├── __init__.py
│   ├── data/                    # 数据加载与预处理
│   │   ├── __init__.py
│   │   ├── download.py
│   │   └── preprocessing.py
│   ├── models/                  # 模型定义
│   │   ├── __init__.py
│   │   └── mlp.py
│   ├── train.py                 # 训练脚本
│   ├── evaluate.py              # 评估脚本
│   ├── predict.py               # 预测脚本（可选）
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       └── metrics.py
├── tests/                        # 单元测试
│   ├── __init__.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore                    # 忽略 data/、models/、__pycache__/ 等
├── LICENSE                       # 许可证（如 MIT）
├── README.md                     # 项目介绍、安装、使用说明
├── requirements.txt              # 依赖包清单
├── setup.py                      # 可选，便于安装本地包
└── Makefile                      # 自动化任务（如 make train）