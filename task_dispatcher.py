"""
task_dispatcher.py · 多模型协作任务编排 · 参考实现
=================================================

本文件是 agent-pipeline 仓库里「任务-模型匹配策略」与「五阶段管理」的**可运行示例**，
把文档里的流程规范落成代码骨架。真实场景里，派发侧的调度脚本在 ai-bridge（服务端任务队列）
与 browser-agent（执行侧）中，这里演示的是**编排逻辑本身**如何照文档跑通。

核心设计（见 README 的任务-模型匹配策略）：
  · 批量重复活（格式转换、逐条抄录）  -> 轻量快模型，省成本
  · 复杂推理（方案设计、多步决策）    -> 深度推理模型，降返工
  · 中间地带                          -> 主模型 + 备用降级

五阶段（见 README）：任务书 -> 派发 -> 监督 -> 验收 -> 收尾
  · 边做边存：长任务最怕跑完 80% 崩了全丢，所以每阶段产出都先落盘
  · 异常重试：单任务失败按 max_retries 重试，不连坐整批
  · 经验回写：收尾阶段把踩过的坑写回经验库，下次不再原样踩

仅依赖标准库，可直接 `python task_dispatcher.py` 看一次完整编排演示。
模型调用处用 `_call_model` 占位，接你自己的 LLM 客户端即可。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# 模型档位：把"用什么模型"从硬编码变成按任务类型查表
# ---------------------------------------------------------------------------
LIGHT_MODEL = "fast-lite"   # 轻量快模型：批量重复活
DEEP_MODEL = "deep-reason"  # 深度推理模型：复杂推理
PRIMARY_MODEL = LIGHT_MODEL  # 中间地带默认走主模型，失败降级到 DEEP


def route_model(task_type: str) -> str:
    """任务-模型匹配：按任务类型选模型，不是所有活都丢给最强模型。"""
    return {
        "batch": LIGHT_MODEL,   # 批量重复：快模型足够，量大时成本差一个量级
        "heavy": DEEP_MODEL,    # 复杂推理：快模型返工率高，交给深度模型
        "mid": PRIMARY_MODEL,   # 中间地带：主模型，下方 dispatch 自带降级
    }.get(task_type, PRIMARY_MODEL)


# ---------------------------------------------------------------------------
# 任务书：能直接发出去的任务，至少包含六块（见 README）
# ---------------------------------------------------------------------------
@dataclass
class TaskSheet:
    task_id: str = field(default_factory=lambda: f"T-{uuid.uuid4().hex[:8]}")
    title: str = ""
    task_type: str = "mid"           # batch / heavy / mid
    prompt: str = ""
    accept_criteria: list[str] = field(default_factory=list)  # 验收清单
    max_retries: int = 3

    def as_brief(self) -> dict:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "accept_criteria": self.accept_criteria,
        }


# ---------------------------------------------------------------------------
# 五阶段编排
# ---------------------------------------------------------------------------
class Dispatcher:
    def __init__(self, out_dir: str = "output") -> None:
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.experience: list[str] = []   # 收尾阶段回写的经验库

    # 阶段 1：任务书 —— 把需求写成可验收的书面形式
    def stage_brief(self, sheet: TaskSheet) -> dict:
        if not sheet.accept_criteria:
            raise ValueError("任务书缺验收清单：执行方会开始猜，猜错就是返工")
        self._persist(sheet.task_id, "brief", sheet.as_brief())
        return sheet.as_brief()

    # 阶段 2：派发 —— 按类型配模型，调用（此处用占位）
    def stage_dispatch(self, sheet: TaskSheet, attempt: int = 1) -> dict:
        model = route_model(sheet.task_type)
        try:
            result = self._call_model(model, sheet.prompt)
            self._persist(sheet.task_id, f"dispatch.att{attempt}", {"model": model, "result": result})
            return {"model": model, "result": result}
        except Exception as e:
            # 中间地带任务：主模型失败自动降级到深度模型
            if sheet.task_type == "mid" and model != DEEP_MODEL:
                self.experience.append(f"{sheet.task_id}: 主模型失败，降级 {DEEP_MODEL}")
                return self.stage_dispatch(TaskSheet(**{**sheet.as_brief(), "task_type": "heavy"}), attempt + 1)
            raise

    # 阶段 3：监督 —— 边做边存，中途检查产出
    def stage_supervise(self, sheet: TaskSheet, result: dict) -> None:
        self._persist(sheet.task_id, "supervise", {"checked_at": time.time(), "ok": bool(result.get("result"))})

    # 阶段 4：验收 —— 对照任务书逐条核，不合格打回
    def stage_verify(self, sheet: TaskSheet, result: dict) -> bool:
        passed = all(c in str(result.get("result", "")) for c in sheet.accept_criteria) \
            if sheet.accept_criteria else bool(result.get("result"))
        self._persist(sheet.task_id, "verify", {"passed": passed})
        return passed

    # 阶段 5：收尾 —— 归档 + 自报家门 + 经验回写
    def stage_wrapup(self, sheet: TaskSheet, passed: bool) -> dict:
        summary = {
            "task_id": sheet.task_id,
            "title": sheet.title,
            "passed": passed,
            "experience_written": len(self.experience),
        }
        self._persist(sheet.task_id, "wrapup", summary)
        return summary

    # 编排入口：跑完五阶段，带重试
    def run(self, sheet: TaskSheet) -> dict:
        self.stage_brief(sheet)
        last_err = None
        for attempt in range(1, sheet.max_retries + 1):
            try:
                disp = self.stage_dispatch(sheet, attempt)
                self.stage_supervise(sheet, disp)
                if self.stage_verify(sheet, disp):
                    return self.stage_wrapup(sheet, True)
                last_err = "验收未通过"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                self.experience.append(f"{sheet.task_id} 第{attempt}次失败: {last_err}")
        # 重试耗尽：仍要收尾，记录问题而不是静默丢
        return self.stage_wrapup(sheet, False)

    # ---- 以下为落地相关（占位 / 工具）----
    def _call_model(self, model: str, prompt: str) -> str:
        """占位：接你自己的 LLM 客户端。这里返回模型档位 + 截断 prompt 以示路由生效。"""
        return f"[{model}] 已处理: {prompt[:24]}..."

    def _persist(self, task_id: str, stage: str, payload: dict) -> None:
        path = self.out / f"{task_id}.{stage}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def demo() -> None:
    d = Dispatcher(out_dir="output")
    tasks = [
        TaskSheet(title="批量抄录岗位字段", task_type="batch",
                  prompt="把 50 条岗位 JSON 转成表格", accept_criteria=["表格"]),
        TaskSheet(title="设计迁移方案", task_type="heavy",
                  prompt="设计会话无损迁移的架构", accept_criteria=["架构"]),
        TaskSheet(title="整理周报", task_type="mid",
                  prompt="汇总本周进展成周报", accept_criteria=["周报"]),
    ]
    for t in tasks:
        out = d.run(t)
        print(f"  · {t.title:<14} -> 模型档位={route_model(t.task_type):<10} 通过={out['passed']}")


if __name__ == "__main__":
    print("多模型协作任务编排 · 五阶段演示（任务书->派发->监督->验收->收尾）")
    demo()
    print("演示结束：output/ 下每个任务按阶段落盘，经验回写见 Dispatcher.experience")
