"""读取由 Rust 权威注册表导出的智能动作 Schema。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


ML_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ML_ROOT / "data"
DEFAULT_ACTIONS_PATH = DATA_DIR / "actions.json"

VALID_RISKS = {"read_only", "write", "high", "critical"}
VALID_SENSITIVITIES = {"public", "workspace_content", "sensitive"}


@dataclass(frozen=True)
class ActionDefinition:
    """ActionDefinition 表示一个可训练且可由业务层执行的动作。"""

    id: str
    description: str
    risk: str
    sensitivity: str
    parameters: dict[str, str]
    required_parameters: tuple[str, ...]
    agent_capability: str | None
    developer_tool: str | None


@dataclass(frozen=True)
class ActionSchema:
    """ActionSchema 表示带版本的完整智能动作协议。"""

    schema_version: int
    agent_protocol_version: int
    actions: tuple[ActionDefinition, ...]
    sha256: str

    @property
    def by_id(self) -> dict[str, ActionDefinition]:
        """by_id 返回按动作 ID 索引的定义。"""

        return {action.id: action for action in self.actions}


def export_action_schema(studio_root: Path, output: Path = DEFAULT_ACTIONS_PATH) -> None:
    """export_action_schema 调用 Rust 导出器生成权威 actions.json。"""

    studio_root = studio_root.resolve()
    manifest = studio_root / "Cargo.toml"
    exporter = studio_root / "src" / "bin" / "export-smart-actions.rs"
    if not manifest.is_file() or not exporter.is_file():
        raise ValueError(f"不是有效的 lance-studio 仓库: {studio_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "export-smart-actions",
            "--",
            str(output),
        ],
        cwd=studio_root,
        check=True,
    )


def load_action_schema(
    path: Path = DEFAULT_ACTIONS_PATH,
    *,
    refresh: bool = False,
    studio_root: Path | None = None,
) -> ActionSchema:
    """load_action_schema 导出或读取并严格校验动作 Schema。"""

    if refresh:
        if studio_root is None:
            raise ValueError("刷新动作 Schema 时必须指定 lance-studio 仓库")
        export_action_schema(studio_root, path)
    elif not path.is_file():
        raise FileNotFoundError(f"动作 Schema 快照不存在: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_root = {"schema_version", "agent_protocol_version", "actions"}
    if set(payload) != expected_root:
        raise ValueError(f"actions.json 顶层字段不符合协议: {sorted(payload)}")

    actions: list[ActionDefinition] = []
    seen_ids: set[str] = set()
    for raw in payload["actions"]:
        expected_action = {
            "id",
            "description",
            "risk",
            "sensitivity",
            "parameters",
            "required_parameters",
        }
        allowed_action = expected_action | {"agent_capability", "developer_tool"}
        if not expected_action.issubset(raw) or not set(raw).issubset(allowed_action):
            raise ValueError(f"动作字段不符合协议: {raw.get('id', '<unknown>')}")
        action_id = str(raw["id"])
        if action_id in seen_ids:
            raise ValueError(f"动作 ID 重复: {action_id}")
        seen_ids.add(action_id)
        risk = str(raw["risk"])
        sensitivity = str(raw["sensitivity"])
        if risk not in VALID_RISKS:
            raise ValueError(f"动作 {action_id} 使用了未知风险: {risk}")
        if sensitivity not in VALID_SENSITIVITIES:
            raise ValueError(f"动作 {action_id} 使用了未知敏感级别: {sensitivity}")
        parameters = {str(key): str(value) for key, value in raw["parameters"].items()}
        required = tuple(str(name) for name in raw["required_parameters"])
        undeclared = set(required) - set(parameters)
        if undeclared:
            raise ValueError(f"动作 {action_id} 的必填参数未声明: {sorted(undeclared)}")
        actions.append(
            ActionDefinition(
                id=action_id,
                description=str(raw["description"]),
                risk=risk,
                sensitivity=sensitivity,
                parameters=parameters,
                required_parameters=required,
                agent_capability=raw.get("agent_capability"),
                developer_tool=raw.get("developer_tool"),
            )
        )

    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return ActionSchema(
        schema_version=int(payload["schema_version"]),
        agent_protocol_version=int(payload["agent_protocol_version"]),
        actions=tuple(actions),
        sha256=file_hash,
    )
