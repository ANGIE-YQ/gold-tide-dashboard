# Gold Tide · 黄金潮汐交易模型

基于「潮汐法则」框架的沪金AU0短期择时Alpha模型。

**看板地址**: `https://<你的用户名>.github.io/<仓库名>/` (部署后生效)

## 快速开始

```bash
# 安装依赖
pip install numpy pandas matplotlib scikit-learn xgboost

# 刷新看板(拉最新数据+重训练+生成HTML)
python refresh_dashboard.py

# 打开看板
open docs/index.html
```

## 自动更新

GitHub Actions 每个交易日 15:30(北京时间)自动运行,拉取最新数据、重训练模型、更新看板、部署到 GitHub Pages。

手动触发: Actions → "Update Gold Tide Dashboard" → Run workflow

## 项目结构

```
├── refresh_dashboard.py        # 自动刷新脚本
├── gold_tide_optimized.py      # 增强信号引擎(日常使用)
├── run_pipeline.py             # 完整流水线
├── gold_tide_engine.py         # 潮汐分段引擎
├── gold_tide_score.py          # 12项动能评分
├── gold_tide_features.py       # 31维特征层
├── gold_tide_ml.py             # Walk-Forward验证
├── gold_tide_t3.py             # T3风险叠加
├── gold_AU0_daily.csv          # 沪金AU0日线数据
├── gold_features.csv           # 特征矩阵
├── docs/
│   └── index.html              # 看板(自动生成)
└── .github/workflows/
    └── update-dashboard.yml    # GitHub Actions
```

## 部署步骤

1. 创建 GitHub 仓库, 推送代码
2. Settings → Pages → Source: "GitHub Actions"
3. Settings → Actions → General → Workflow permissions: "Read and write permissions"
4. 手动触发一次 Actions → "Update Gold Tide Dashboard" → Run workflow
5. 等待部署完成, 访问 `https://<用户名>.github.io/<仓库名>/`

⚠️ 模型输出仅供参考,不构成投资建议。
