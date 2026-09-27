# 新建 topic

目标是从第一次任务开始建立可迁移、可复用、易接续的工作区。

## 流程

1. 创建 topic 目录并执行 `git init`。
2. 从 `assets/templates/gitignore.sample` 创建 `.gitignore`。
3. 创建 topic README、`shared/`、任务类型目录及其 `_TEMPLATE/`。
4. 按以下边界放置内容：
   - 文档、模板和小于 10 MiB 的任务脚本进入 git；
   - 原始数据进入 `raw/<任务类型>/<单次任务>/`；
   - 可重算产物进入 `out/<任务类型>/<单次任务>/`；
   - checkpoint 和日志进入 `state/`、`logs/`。
5. 首个任务先创建并填写单次任务 README，再开始执行。
6. 任务推进中持续更新 README；稳定的跨类型知识再沉淀到 `shared/`。

## 模板

- `assets/templates/topic-README.md`
- `assets/templates/type-README.md`
- `assets/templates/task-README.md`
- `assets/templates/gitignore.sample`

不要自动配置远端、push 或提交用户未确认的现有文件。
