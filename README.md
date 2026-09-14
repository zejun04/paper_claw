# paper_claw

自动抓取 arXiv 上与四足机器人、人形机器人和机械臂操作相关的论文，并生成中文 Markdown 深度摘要。

## 本地运行

```bash
python -m pip install -r requirements.txt
python -m paperclaw fetch --dry-run
python -m paperclaw fetch --days 30
```

正式运行需要设置：

```text
OPENAI_API_KEY=你的 OpenAI API key
OPENAI_MODEL=你的账户可用模型名
```

首次运行默认回补最近 30 天；后续运行默认检查最近 3 天并按 arXiv ID 去重。每次每个分类最多保存 5 篇，文章只归档到模型判断的一个主分类。

## GitHub Actions

`.github/workflows/arxiv-daily.yml` 每天 09:00（Asia/Shanghai）运行，也可以在 Actions 页面手动触发。需要在仓库设置：

- Secret：`OPENAI_API_KEY`
- Repository variable：`OPENAI_MODEL`
- Settings → Actions → General → Workflow permissions 选择允许读写仓库内容。

工作流会把新 Markdown 和 `.paperclaw/state.json` 直接提交到 `main`。PDF 只在运行期间下载到临时目录，不提交到仓库。

## 目录

- `Quard-robot/`：四足、足式和四腿机器人
- `humanoid/`：人形和双足机器人
- `pick/`：机械臂、抓取、操作和拾取放置
