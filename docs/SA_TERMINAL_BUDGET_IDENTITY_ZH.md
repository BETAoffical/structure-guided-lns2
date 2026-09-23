# 运行时身份补充与审计入口

本轮采集按提交`698e0b4`启动后，独立代码复核发现两处维护问题：

1. 原始登记继承的inputs并未递归包含全部Python依赖，例如预算特征、无次数上限停止规则、随机流工具。
   没有发现这些文件发生变化，但原清单本身不足以拒绝未来的这类变化。
2. 原分析入口把每次审计写入同一个audit-attempt.json，失败后再次成功的结果会被不可覆盖写入保护拒绝。
   本轮尚未触发这个错误，不能把它说成采集失败原因。

为避免在运行中修改已登记采集器，新增独立审计入口，保留原脚本原样：

```powershell
python scripts/audit_sa_terminal_budget.py pin
python scripts/audit_sa_terminal_budget.py verify
```

pin不是重新登记科学协议。它将启动前Git提交中的全部871个Python文件逐文件与当前内容比较，
只容许LF/CRLF表示差异，再登记它们以及新增审计入口和测试，共873份SHA。
补充证据明确记录supplemental_after_launch=true，不冒充事前已有完整依赖清单。
原模型、native、配置和轨迹仍由原inputs和各结果SHA验证，未重跑求解。

后续离线审计使用冻结native路径和既有WSL Python：

```bash
PYTHONPATH=build/linux/sa-wall-clock-v1 /usr/bin/python3 -B scripts/audit_sa_terminal_budget.py analyze
```

本次不使用旧probe脚本的analyze入口。新入口在审计前后检查运行时身份，
复用同一audit_worker重算特征、概率、SA随机数、轨迹与路径质量。
各次审计分别保存，按job_id固定排序，保留失败记录而不覆盖。
报告引用runtime_identity_supplement.json的SHA。

所有采集执行代码保持原SHA，不能将这一登记补充解释成算法修复或性能提升。
今后若新开实验，应在启动前直接登记完整依赖，而不是沿用这一事后补充过程。
