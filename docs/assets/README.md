# Public visual assets / 公开视觉素材

This directory contains the five public images embedded by relative path in
both repository READMEs. It contains no raw browser capture, server profile,
ticket ledger, checkpoint, training log, or TensorBoard event file.

本目录保存中英文 README 通过相对路径嵌入的五张公开图片，不包含原始浏览器
截图、服务器 profile、工单账本、checkpoint、训练日志或 TensorBoard event 文件。

## Provenance and review / 来源与复核

- The two diagrams were generated with the built-in **GPT Image 2** workflow,
  normalized to metadata-free RGB PNGs, and accepted only after two independent
  fresh subagent checks: one semantic-consistency review and one generated-image
  detail-consistency review.
- The three screenshots were rendered and captured through **Chrome**. Each was
  accepted only after an independent fresh subagent semantic-consistency review.
  The overview's connection endpoint is pixel-mosaicked. Raw captures are
  deliberately excluded.
- `dashboard-overview.png` and `dashboard-scratch20-tensorboard.png` are
  sanitized historical captures from 2026-08-13. They document dashboard
  features and one completed/released `scratch20` run; they are not current-state
  evidence.
- `setup-wizard-simulation.png` renders
  `docs/demos/setup-wizard-simulation.html`. All values are fictional, no SSH
  connection is attempted, and no profile is written.

- 两张架构图由内置 **GPT Image 2** 工作流生成并转存为不含元数据的 RGB PNG；每张图
  均须通过两个独立、全新子智能体的复核：一个检查语义一致性，另一个检查生成图细节
  一致性。
- 三张截图均通过 **Chrome** 渲染和捕获；每张图均须通过一个独立、全新子智能体的
  语义一致性复核。看板总览中的连接端点已做像素马赛克，原始截图明确不发布。
- 两张看板图是 2026-08-13 的脱敏历史截图，只用于介绍功能和一个已经完成并释放的
  `scratch20` 任务，不能作为当前服务器状态证据。
- 模拟接入图渲染自 `docs/demos/setup-wizard-simulation.html`；其中全部数值均为虚构，
  不发起 SSH 连接，也不写入 profile。

## Manifest / 清单

| Relative path | Source | Dimensions | SHA-256 |
|---|---|---:|---|
| `docs/assets/diagrams/system-workflow.png` | GPT Image 2; semantic + detail reviewed | 1672 x 941 | `bb13618b2d326bee4abadd656d4876a10ac70da7027e439f731a950727d93282` |
| `docs/assets/diagrams/ticket-system.png` | GPT Image 2; semantic + detail reviewed | 1536 x 1024 | `5d261d7ef89d60d53d0c2440de863bd6e1336d965bc3d4ed49739f42abe98367` |
| `docs/assets/screenshots/dashboard-overview.png` | Chrome; actual sanitized historical dashboard | 1524 x 803 | `1cd7975621c96ce03eee9f92d58ce5c6e4ae579475d21413e3d148873488f218` |
| `docs/assets/screenshots/dashboard-scratch20-tensorboard.png` | Chrome; actual sanitized historical TensorBoard viewer | 1468 x 758 | `5f05ac51b06fafac3ed0c018f00c6c0920e1be28cbb7f36047781ae9bbdfac7d` |
| `docs/assets/screenshots/setup-wizard-simulation.png` | Chrome; fictional offline setup simulation | 1538 x 754 | `d6fb6213796aa9705da58eb361910764599dd14d8b0c4f9283804032ca49b51b` |

If any image changes, rerun its required fresh visual review(s), update this
manifest, and keep the English and Chinese README image-path sets identical.

任意图片发生变化后，都必须重新执行该图片所需的全新视觉复核、更新本清单，并保持
中英文 README 引用的图片路径集合完全一致。
