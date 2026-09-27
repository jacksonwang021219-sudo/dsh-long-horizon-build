# 长程任务模式的 DSH

本仓库是 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)（`dsh`）的一份完整源码快照，在上游 MIT 代码之上加入了**长程任务模式**：上下文策略从「接近窗口上限才压缩」换成**滑动窗口**，让一次会话可以跨数小时、跨多轮继续推进同一件事。

- 上游：`deepseek-ai/deepseek-harness`，快照自 `master` 提交 `47f9438`，许可证 MIT。
- 本仓库是个人维护的**非官方**快照，不是上游官方分支；上游的新提交请直接从上游同步。

## 长程任务模式由三部分组成

| 组成 | 位置 | 作用 |
| --- | --- | --- |
| 上下文窗口插件 | `packages/context/context-window/`（`@deepseek-ai/dsh-context-window`） | 滑动窗口策略本体，同时注册记录协议的系统提示段与两个工具 |
| agent preset | `apps/cli/config/agent-presets/long-horizon/`（显示名「长程任务模式」，id `long-horizon`） | 把上游 `cordis` preset 的上下文策略行换成上面的插件 |
| skill | 同一 preset 目录下的 `skills/long-horizon-task/` | topic 工作区、可信上下文落盘、断点续跑的组织方法 |

这个 preset 随**部署**发布，和上游的 `standard` / `code` / `minimal` / `cordis` 并列在 `apps/cli/config/agent-presets/` 下（preset 的 `system` 根）。

### 上下文策略怎么变

窗口定义在**用户请求**上，而不是「请求 + 回答轨迹」上：

- 最近 `retainUserTurns`（默认 5，含正在准备的那一轮）个用户请求**原样**留在模型可见面上。
- 更早那些轮次的**执行轨迹**（Agent 自己的步骤、工具调用与结果）不留：它们由 Agent 自己写下的**记录**顶替。
- 比窗口还老的轮次整轮离开可见面，折叠成一个由记录重建的 `<folded-history>` 摘要块。
- 单个用户轮内，轨迹超过 `foldTriggerTokens`（默认 400K）后触发轮内折叠，回到最近 `retainTurnTokens`（默认 200K），折叠边界向上取整到完整的 Agent 步，绝不切开一次工具调用与其结果。
- 折叠块落在被它替换掉的「请求 + 回答」原来的位置上，因此以回答的形态呈现：开头固定一行 `针对本问题Agent的任务执行的大致情况如下：`，后面是本轮记录（任务定义、策略与进展、关键结论、交付与验收）。这一行由 harness 写死、每一次折叠都完全一致，作用是把折叠块与普通问答对区分开，让主 Agent 把它认成自己过去给出的回答，而不是一份被塞进来的文档。
- 折叠之后若仍超过模型阈值，才交给 dsh 原有的 `compaction-basic` 压缩。

**没有记录过的内容永不被折叠**，因此折叠只会丢掉会话日志里仍然完整保存的细节：每条记录都写明它覆盖的日志 seq 区间（改动 2 的字段），`session_log` 工具可以按区间读回原始事件。

### 记录协议（改动 1 与改动 2）

原来有一个专门的组装 Agent，每个用户轮开始时新建一次，去工作区把上下文重新检索组装成一块。现在这个 Agent 去掉了，它的职责并进主 Agent loop。**记录由窗口自己写**，`recordMode` 只有 `recorder` 一个合法值（写成别的值在装载时直接报错），折叠规则统一：

- 每次需要记录时，窗口发**一次模型调用**（无工具、无 Agent loop）。主 Agent 的 prompt 只讲「怎么读、怎么纠错」，完全不提「怎么写」；它拿到的工具只有 `read_notes`、`fix_note`、`session_log`，**没有任何写笔记的工具**。
- 记录落在工作区之外：`<notesRoot>/<sessionId>/`（默认 `~/.dsh/notes/`），含 `index.json`、`tasks/task-NNNN.json`、`topics/`、`types/`、`recorder.log`、`recorder-usage.jsonl`。主 Agent 只能通过读工具访问它。
- 记录失败时按 `recorderRetries`（默认 5）重试，并把可执行的失败原因回灌给下一次尝试；连续 `recorderCircuitBreaker`（默认 3）轮记录失败则冻结整会话的折叠，并给出一条可见通知。
- 记录契约要求用会话自己的术语写结论（路径、符号、命令、标识符、数字一律照抄，不改写成描述），**不许出现第三人称**（不写「Agent」「助手」「记录器」「本会话」，也不写动作是谁做的），并且禁止只写指针的字段（「见上一条笔记」这类不携带信息，要么把事实写出来，要么留空）。原因见下一节：笔记会落在被折叠轨迹原来的位置上，被读成本会话自己对那一轮的回答，因此必须脱离上下文即可使用。
- 记录覆盖区间仍然写明日志 seq（改动 2 的字段）：每条折叠块头部带 `log-range`，`session_log` 可按区间读回原始事件。

### 记录器回放哪一份视图（V1 / V2 开关）

`recorderView` 决定这次记录调用把哪一份消息历史发给模型：

- **`aligned`（默认，即 V2）**：主 Agent 自己的 `system` + `tools` + 当前可见窗口。只有尾部不同，因此提供商能命中会话刚刚预热的前缀缓存（实测约 90% 的 prompt 走缓存价），而且记录器看到的是**主 Agent 当前能看到的同一份上下文**，所以它能判断哪些结论已经写过了。
- **`span`（即 V1）**：只回放被记录那一轮自己的消息。请求从第一条消息就与会话分叉，只有 system 与工具块能命中缓存（实测约 65% 的 prompt 按未命中价计费），记录器也看不到更早的摘要已经写了什么——因此 `recorderDedup` 在这个视图下无法执行。

`recorderDedup`（默认 `true`）在契约里写死一条硬规则：**已被 `<existing-notes>` 覆盖或已出现在窗口里的结论一律不得重复**，本轮没有新结论的字段留空。为了这条规则可执行，`index.json` 的每条 task 还带该笔记的 `key_conclusions`（截断到 400 字符），折叠摘要行和 `<existing-notes>` 都会渲染它——否则过期的轮次在可见面上只剩一行标题，规则无据可依。

**成本口径。** 记录器自己的调用不进会话日志，所以每次调用的 token 与缓存命中数会单独写进 `recorder-usage.jsonl`（provider、model、耗时、按缓存类别拆分的 token），可以直接按「缓存命中 / 未命中 / 输出」三档单价计价，不需要估算。

**记录的优先级与保底。** 需要折叠但还没有记录时，窗口先在本步注入 `<window-notice>`（附日志区间）请模型补记，下一步再折叠：

- 跨轮折叠（`turn-trajectory` / `history-digest`）只折叠**已被记录**的轮次；没有记录的轮次保留完整轨迹，绝不静默丢内容。
- 轮内 token 上限是**硬约束**：提醒用尽后窗口仍按它已经承诺过的边界折叠，替换块改为 `<unrecorded-trace>`——明确写出"这一段没有记录、原文在日志 seq a–b，请用 `session_log` 读回、必要时补记"。模型配合时行为不变（正常记录块），不配合时 400K → 200K 的上限依然成立、信息仍可寻回。

### 查询完整日志的能力（改动 1）

主 Agent 新增 `session_log` 工具，按需读回任何已经离开可见面的内容：

- `records: true` 列出历史各轮的记录（轮次、日志区间、摘要）；
- `from` / `to` 读回任意 seq 区间的原始事件，可用 `query` 文本过滤或 `types` 限定事件类型。

除此之外一切不变：同一 persona、同一批 prompt 段落、同一套工具。变的只是历史**什么时候**离开可见面，以及**由什么**顶替它。

## 启动 WebUI 的长程模式

从源码构建并启动，一共四条命令：

```sh
git clone https://github.com/jacksonwang021219-sudo/dsh-long-horizon.git
cd dsh-long-horizon
pnpm install
pnpm run build
DEEPSEEK_API_KEY=sk-... pnpm dsh web
```

最后一条会打印地址，浏览器打开即可（默认 `http://127.0.0.1:3080`；换端口用 `--port 8080`，`--port 0` 让系统随机挑一个空闲端口并把 URL 打出来）。

**不需要任何额外步骤**：preset 作为部署自带的一份发布在 `apps/cli/config/agent-presets/long-horizon/`，`packages/bundle/web-app/cordis.patch.yml` 里的 `agent-presets.default` 指向它，所以**新会话默认就跑长程任务模式**。

### 在界面上确认

- 开新会话前，起始卡片上的预设芯片（悬停提示「即将开始的这个会话所用的 Agent 预设」）显示 **长程任务模式**。
- 会话开始后，头部标签显示 **长程任务模式**（提示「本会话运行的 Agent 预设，开始时即固定」）。
- 设置 → **Agent 预设** 里，「长程任务模式」列在 **内置** 组，带「当前使用」标记。

### 用命令行确认

```sh
curl -s -X POST http://127.0.0.1:3080/api/agentPreset.list \
  -H 'content-type: application/json' \
  -d '{"type":"client-request","rpcId":"1","method":"agentPreset.list","payload":{}}'
```

返回的 `presets` 里应有一项 `"id":"long-horizon"`、`"trust":"system"`、`"isDefault":true`，且没有 `broken` 字段。端口换成你实际用的那个。

### 切走 / 切回

- **单个会话**：开新会话时在预设芯片上选别的（例如「标准模式」）；会话一旦开始就固定用开始时的 preset，想换就新开一个。
- **默认换回普通模式**：设置 → Agent 预设 → 在「标准模式」上点「设为默认」；或在自己的 `<dshHome>/settings.yaml` 写
  ```yaml
  agent-presets:
    default: standard
  ```
  用户层优先于仓库里的默认值。

### 首次运行的其他注意

- 模型凭据走 `DEEPSEEK_API_KEY`，也可以放仓库根 `.env`。只启动界面不需要 key，**发第一条消息**才需要模型。
- Node.js `^22.19 || >=24`，pnpm。依赖加构建产物约 1.8 GB，首次 `install` + `build` 大约 5–10 分钟。
- 端口被占用时用 `--port 8080`；`--host` 换绑定地址。

## 调整

- **窗口大小**：改 preset 里 `context-window` 行的 `retainUserTurns` / `retainTurnTokens` / `foldTriggerTokens` / `digestMaxTokens` / `recordReminders`。preset 目录是运行时按需读取的，改完开新会话即生效；`foldTriggerTokens` 必须大于 `retainTurnTokens`，否则加载即报错。
- **谁写记录**：`recordMode: recorder`（唯一合法值）。配套 `policy: minimal`（主 Agent 没有写入口，prompt 只留读与纠错），并可调 `recorderRetries` / `recorderTimeoutMs` / `recorderWaitMs` / `recorderCircuitBreaker` / `recorderMaxTokens` / `recorderView` / `recorderDedup`。
- **记录器看什么**：`recorderView: aligned | span`（缓存命中率与去重能力的开关），`recorderDedup: true | false`（契约里的去重硬规则）。
- **验证阈值**：把上面几个值调到很小（例如 1 轮 / 3000 / 6000）就能在几轮短对话里触发全部三条折叠规则，适合端到端验证。
- **把这个 preset 装到别的 harness home**（比如你在用另一份 checkout）：
  ```sh
  cp -R apps/cli/config/agent-presets/long-horizon "${DSH_HOME:-$HOME/.dsh}/.agent-presets/"
  ```
  同 id 时**部署自带的根优先**，会遮蔽 home 里那份，所以别两边各改一半。

### 用不了这个 preset 的情况

- `npx @deepseek-ai/dsh web` 之类的**上游发行版跑不了长程模式**：`@deepseek-ai/dsh-context-window` 是本仓库新增、`private: true`、未发布到 npm 的包，换到上游构建上会因插件无法解析而 mount 失败。必须用本仓库源码构建的 `dsh`。
- 另有一个自带的硬约束：`--patch` 覆盖层**无法给 preset 增加根目录**——`apps/cli/src/profile-boot.ts` 的 `composeProfile` 在最后把所有 patch 层里的 `roots` 整体替换为部署自带的那个根。所以 preset 必须放进 `apps/cli/config/agent-presets/` 才能开箱可用。

### 两点已知取舍

- 该 preset 里 `tool-cordis`（运行时自省与临时挂载插件）被**显式禁用**：它注册的 Host Cordis inspect provider id 是进程级的，与上游 `cordis` preset 冲突，两者同一进程同时挂载会直接 mount 失败。需要改 harness 自身时切到「创造模式」。
- 挂着这个 preset 的会话属于 **shell 级信任**：模型写出的 composition 会成为别的会话可挂载的 preset。

## 相对上游的改动

- 新增 `packages/context/context-window/`：`README.md`、`package.json`、`src/{index,window,records,recorder,notes,tools,prompt,state,config,invariant,text}.ts`、`tsconfig.json`、`tsdown.config.ts`、`tests/window.spec.ts`。
- 新增 `.agents/notes/implemented/architecture/2026-09-25-main-agent-turn-records.md`：这次"去掉组装 Agent、改为主 Agent 自记或窗口自记"的决策、备选方案与验证。
- 删除 `packages/context/context-window/src/assembler.ts`。
- 评测工作区（不属于本仓库）：`../long-horizon-eval/`，含两个开源 benchmark 的克隆、驱动脚本与结果；报告见 `../long-horizon-eval/report/long-horizon-eval-report.md`。
- 新增 `apps/cli/config/agent-presets/long-horizon/`：`preset.yml`、`agent.cordis.yml` 与 `skills/long-horizon-task/`（7 个文件）。
- 修改 `packages/bundle/web-app/cordis.patch.yml`：`agent-presets.default` 从 `standard` 改为 `long-horizon`。
- 修改 `tsconfig.host.json`：为上下文窗口插件补一条 project reference。
- 修改 `pnpm-lock.yaml`：新增该 workspace 成员的 importer；其余为本地 pnpm 重新解析带来的 peer 后缀变动。
- 新增本文件 `LONG-HORIZON.md`，并在 `README.md` / `README.zh.md` 顶部各加一段指向它的说明。
- 上游的 `.github/`、`LICENSE`、`THIRD_PARTY_NOTICES.md` 原样保留。

## 许可证与归属

上游代码为 MIT（© 2026 DeepSeek），`LICENSE` 与 `THIRD_PARTY_NOTICES.md` 未改动。本仓库是个人维护的非官方快照，不代表 DeepSeek AI。
