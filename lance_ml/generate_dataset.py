"""从权威动作 Schema 生成模板族隔离的脱敏训练数据。"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .action_schema import ActionDefinition, DATA_DIR, load_action_schema
from .dataset import count_by, sha256_file, write_json


DATASET_GENERATOR_VERSION = 2
DEFAULT_SEED = 20260901
SPLITS = ("train", "validation", "test")
TAILS = ("", "。", "，谢谢", "，现在处理", "，在当前空间", "，完成后告诉我")
ALLOWED_SECRET_VALUES = {"<JWT>", "<URL>", "<USERNAME>", "<SECRET>"}
PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")

SPLIT_SHAPES = {
    "train": ("{verb}{payload}", "请{verb}{payload}", "我需要{verb}{payload}"),
    "validation": ("能否替我{verb}{payload}", "{payload}，麻烦{verb}", "现在想{verb}{payload}"),
    "test": ("帮忙{verb}一下：{payload}", "对{payload}执行{verb}", "我的需求是{verb}{payload}"),
}


@dataclass(frozen=True)
class LanguageSpec:
    """LanguageSpec 定义一个动作的表达、槽位和混淆模板。"""

    verbs: tuple[str, ...]
    payloads: tuple[str, ...]
    values: dict[str, tuple[str, ...]]
    missing_payloads: tuple[str, ...]
    confusion_payloads: tuple[str, ...]


LANGUAGE_SPECS = {
    "json.format": LanguageSpec(
        verbs=("格式化 JSON", "校验 JSON", "压缩 JSON"),
        payloads=(" 数据 {input}", "：{input}", " 内容 {input}"),
        values={
            "input": (
                '{"name":"思芽岛","native":true}',
                '{"enabled":false,"count":3}',
                '{"items":[1,2,3]}',
                '[{"id":1},{"id":2}]',
                '{"user":{"name":"Alice"}}',
                '["rust","go","python"]',
                '{"version":2,"debug":null}',
                '{"nested":{"ready":true}}',
            )
        },
        missing_payloads=("但还没有数据", "一段 JSON", "，内容稍后提供"),
        confusion_payloads=(
            "格式化 JSON {input}，不是做 Base64 编码",
            "校验 {input} 的 JSON 语法，不要搜索文件",
            "压缩 JSON {input}，不是用 GZip 压缩",
        ),
    ),
    "timestamp.convert": LanguageSpec(
        verbs=("转换", "换算", "解析时间"),
        payloads=(" {value} 到 {timezone}", "时间值 {value}", " {value} 的日期时间"),
        values={
            "value": (
                "1717200000",
                "1717200000000",
                "2026-09-01 09:30:00",
                "2026/09/01 18:45",
                "2026年09月01日",
                "20260901093000",
                "2025-12-31T23:59:59+08:00",
                "946684800",
                "1893456000000",
                "2030.01.01 08:00",
            ),
            "timezone": ("Asia/Shanghai", "北京时间", "UTC", "本地时区"),
        },
        missing_payloads=("一个时间值", "日期和时间戳", "但我还没给具体时间"),
        confusion_payloads=(
            "只换算 {value}，不要创建便签",
            "把 {value} 解析为日期，不是搜索文件",
            "查看 {value} 对应时间，不要打开日历",
        ),
    ),
    "cron.parse": LanguageSpec(
        verbs=("解析 Cron", "查看定时计划", "预览 Cron 执行时间"),
        payloads=(" 表达式 {expression}", "：{expression}", " {expression} 的下一次执行"),
        values={
            "expression": (
                "*/5 * * * *",
                "0 */5 * * * *",
                "0 9 * * 1-5",
                "30 8 * * *",
                "0 0 1 * *",
                "15 10 * * MON-FRI",
                "0 */2 * * * *",
                "0 0 12 1/1 * ?",
            )
        },
        missing_payloads=("但还没给表达式", "一个定时规则", "，Cron 内容稍后提供"),
        confusion_payloads=(
            "解析 Cron {expression}，不是创建日历笔记",
            "预览 {expression} 的执行时间，不要换算时间戳",
            "检查定时表达式 {expression}，不要打开终端",
        ),
    ),
    "base64.convert": LanguageSpec(
        verbs=("处理 Base64", "转换 Base64", "进行 Base64 编解码"),
        payloads=("，{mode}内容 {input}", "：把 {input} {mode}", " 输入 {input}，模式 {mode}"),
        values={
            "input": (
                "Hello Lance",
                "5oCd6I6J5bKb",
                "SGVsbG8gV29ybGQ=",
                "eyJuYW1lIjoibGFuY2UifQ==",
                "Rust 和 Go",
                "dGVzdC1kYXRh",
                "开发工具",
                "MTIzNDU2",
            ),
            "mode": ("编码", "解码"),
        },
        missing_payloads=("但还没有输入内容", "一段 Base64", "，数据稍后提供"),
        confusion_payloads=(
            "把 {input} 做 Base64 {mode}，不是处理图片",
            "对文本 {input} 进行 Base64 {mode}，不要解析 JWT",
            "使用 Base64 文本工具{mode} {input}，不是 URL 编码",
        ),
    ),
    "password.open": LanguageSpec(
        verbs=("打开密码管理", "进入密码保险箱", "查看已保存的凭据"),
        payloads=("", "工具", "面板"),
        values={},
        missing_payloads=(),
        confusion_payloads=(
            "打开密码管理，不要生成随机密码",
            "进入密码保险箱，不是新建凭据",
            "查看已保存的账号，不要解析 JWT",
        ),
    ),
    "number_base.convert": LanguageSpec(
        verbs=("转换进制", "换算数字进制", "进行进制转换"),
        payloads=("，把 {value} 转为 {target_radix} 进制", "：{value} 到 {target_radix} 进制", " 数值 {value}，目标 {target_radix} 进制"),
        values={
            "value": ("0xff", "0b1010", "0o755", "36#z", "255", "-42", "1024", "7fffffff"),
            "target_radix": ("2", "8", "10", "16", "36"),
        },
        missing_payloads=("但还没给数字", "一个数值", "，目标进制稍后提供"),
        confusion_payloads=(
            "把 {value} 转为 {target_radix} 进制，不是做 Base64 编码",
            "换算数字 {value} 的进制，不要转换时间戳",
            "用进制工具处理 {value}，不是计算哈希",
        ),
    ),
    "base64_image.convert": LanguageSpec(
        verbs=("转换 Base64 图片", "处理图片 Base64", "进行图片与 Data URL 互转"),
        payloads=("，{mode} {input}", "：输入 {input}，执行{mode}", " 文件或数据 {input}，模式 {mode}"),
        values={
            "input": (
                "<IMAGE_PATH>",
                "assets/logo.png",
                "images/photo.webp",
                "screenshots/home.png",
                "<BASE64_IMAGE>",
                "data:image/png;base64,<BASE64_IMAGE>",
            ),
            "mode": ("图片转 Base64", "Base64 转图片"),
        },
        missing_payloads=("但还没选择图片", "一个图片或 Data URL", "，输入稍后提供"),
        confusion_payloads=(
            "把图片 {input} 做{mode}，不是处理普通文本",
            "使用 Base64 图片工具处理 {input}，不要转换图片格式",
            "对 {input} 执行{mode}，不是生成二维码",
        ),
    ),
    "gzip.convert": LanguageSpec(
        verbs=("处理 GZip", "转换 GZip", "进行 GZip 压缩解压"),
        payloads=("，{mode} {input}", "：把 {input} {mode}", " 输入 {input}，模式 {mode}"),
        values={
            "input": (
                "Hello Lance",
                "需要压缩的日志文本",
                "H4sIAAAAAAAA/8tIzcnJBwCGphA2BQAAAA==",
                "<GZIP_BASE64>",
                "Rust workspace metadata",
                "一段重复重复重复的内容",
            ),
            "mode": ("压缩", "解压"),
        },
        missing_payloads=("但还没有数据", "一段 GZip 内容", "，输入稍后给"),
        confusion_payloads=(
            "用 GZip {mode} {input}，不是压缩 JSON 格式",
            "处理 GZip 数据 {input}，不要做 Base64 文本转换",
            "对 {input} 执行 GZip {mode}，不是转换图片",
        ),
    ),
    "jwt.decode": LanguageSpec(
        verbs=("离线解析", "解码", "查看 JWT"),
        payloads=(" {token}", "令牌 {token}", " {token} 的 Header 和 Claims"),
        values={"token": ("<JWT>",)},
        missing_payloads=("一个 JWT", "JWT 内容", "但令牌稍后再给"),
        confusion_payloads=(
            "只离线解析 {token}，不要保存到密码库",
            "查看 {token} 的 Claims，不验证签名",
            "解码 {token}，不是打开同名文件",
        ),
    ),
    "qr_code.process": LanguageSpec(
        verbs=("处理二维码", "使用二维码工具", "进行二维码生成读取"),
        payloads=("，{mode} {input}", "：输入 {input}，执行{mode}", " 内容 {input}，模式 {mode}"),
        values={
            "input": (
                "https://lance.local/docs",
                "思芽岛开发工具",
                "WIFI:S:Studio;T:WPA;P:<SECRET>;;",
                "assets/qrcode.png",
                "screenshots/payment-qr.png",
                "会议签到链接",
            ),
            "mode": ("生成二维码", "读取二维码"),
        },
        missing_payloads=("但还没给内容或图片", "一个二维码", "，输入稍后提供"),
        confusion_payloads=(
            "把 {input} {mode}，不是做 Base64 图片转换",
            "使用二维码工具{mode} {input}，不要转换图片格式",
            "对 {input} 执行{mode}，不是 URL 编码",
        ),
    ),
    "url.convert": LanguageSpec(
        verbs=("处理 URL 编码", "转换 URL 组件", "进行百分号编解码"),
        payloads=("，{mode} {input}", "：把 {input} {mode}", " 输入 {input}，模式 {mode}"),
        values={
            "input": (
                "hello world",
                "name=思芽岛",
                "a%2Fb%3Fc%3D1",
                "https%3A%2F%2Fexample.com%2Fdocs",
                "查询 参数",
                "path/to file",
                "email+tag@example.com",
                "中文路径/说明",
            ),
            "mode": ("编码", "解码"),
        },
        missing_payloads=("但还没有 URL 内容", "一个 URL 组件", "，输入稍后提供"),
        confusion_payloads=(
            "对 URL 组件 {input} {mode}，不是 Base64 转换",
            "把 {input} 做百分号{mode}，不要打开网页",
            "使用 URL 工具{mode} {input}，不是转换进制",
        ),
    ),
    "hash.generate": LanguageSpec(
        verbs=("计算哈希", "生成摘要", "计算校验和"),
        payloads=("，算法 {algorithm}，内容 {input}", "：用 {algorithm} 处理 {input}", " 输入 {input}，选择 {algorithm}"),
        values={
            "input": (
                "hello world",
                "思芽岛",
                "release-2026-09",
                "Cargo.toml",
                "checksum-input",
                "123456",
                "Rust and Go",
                "需要校验的文本",
            ),
            "algorithm": ("MD5", "SHA-1", "SHA-256", "SHA-512"),
        },
        missing_payloads=("但还没给输入内容", "一段待校验文本", "，数据稍后提供"),
        confusion_payloads=(
            "计算 {input} 的 {algorithm}，不是生成密码",
            "为 {input} 生成 {algorithm} 摘要，不要做 Base64 编码",
            "用哈希工具校验 {input}，不是解析 JWT",
        ),
    ),
    "password.generate": LanguageSpec(
        verbs=("生成随机密码", "创建安全密码", "使用密码生成器"),
        payloads=("", "，长度 {length}", "，生成 {length} 位密码"),
        values={"length": ("12", "16", "20", "24", "32", "48")},
        missing_payloads=(),
        confusion_payloads=(
            "生成一个新随机密码，不要打开密码保险箱",
            "创建 {length} 位安全密码，不是保存凭据",
            "用密码生成器生成 {length} 位密码，不要计算哈希",
        ),
    ),
    "uuid.generate": LanguageSpec(
        verbs=("生成 UUID", "创建 GUID", "使用 UUID 生成器"),
        payloads=("", "，版本 {version}", "：生成 {version}"),
        values={"version": ("v1", "v4", "GUID", "v7")},
        missing_payloads=(),
        confusion_payloads=(
            "生成 {version} UUID，不要创建随机密码",
            "创建一个 {version} 标识符，不是计算哈希",
            "用 UUID 工具生成 {version}，不要打开文件",
        ),
    ),
    "image.convert": LanguageSpec(
        verbs=("转换图片格式", "进行无损图片转换", "使用图片转换器"),
        payloads=("，把 {path} 转为 {format}", "：输入 {path}，输出 {format}", " 文件 {path}，目标格式 {format}"),
        values={
            "path": (
                "assets/logo.jpg",
                "images/photo.png",
                "screenshots/home.webp",
                "design/icon.bmp",
                "exports/cover.tiff",
                "assets/banner.jpeg",
            ),
            "format": ("PNG", "WebP", "BMP"),
        },
        missing_payloads=("但还没选择图片", "一个图片文件", "，目标格式稍后提供"),
        confusion_payloads=(
            "把图片 {path} 转为 {format}，不是做 Base64 图片编码",
            "转换 {path} 的文件格式，不要读取二维码",
            "用图片格式工具处理 {path}，不是压缩图片质量",
        ),
    ),
    "regex.test": LanguageSpec(
        verbs=("测试正则", "验证正则表达式", "查找正则匹配"),
        payloads=("，表达式 {pattern}，文本 {test_text}", "：用 {pattern} 匹配 {test_text}", " 模式 {pattern}，测试内容 {test_text}"),
        values={
            "pattern": (
                "\\d+",
                "^[a-z]+$",
                "(?P<name>\\w+)",
                "https?://[^\\s]+",
                "[A-Z][a-z]+",
                "\\bTODO\\b",
            ),
            "test_text": (
                "build 20260901 completed",
                "hello",
                "Name Alice",
                "visit https://example.com/docs",
                "Hello Lance Studio",
                "TODO fix routing",
            ),
        },
        missing_payloads=("但还没有表达式和文本", "一个正则规则", "，测试内容稍后提供"),
        confusion_payloads=(
            "用正则 {pattern} 匹配 {test_text}，不是搜索文件",
            "测试表达式 {pattern}，不要执行代码",
            "列出 {test_text} 中 {pattern} 的匹配，不要格式化 JSON",
        ),
    ),
    "sticky_note.create": LanguageSpec(
        verbs=("创建便签", "新建便签", "记到便签"),
        payloads=("，内容是 {content}", "：{content}", "并写下 {content}"),
        values={
            "content": (
                "下午三点开会",
                "整理本周计划",
                "检查 JWT 文档",
                "完成 Git 代码审查",
                "给客户回复邮件",
                "明天更新日历",
                "记录终端快捷键",
                "买咖啡和水果",
                "周五提交周报",
                "阅读项目设计稿",
                "确认发布清单",
                "学习 Rust 生命周期",
            )
        },
        missing_payloads=("但内容还没想好", "，稍后再补正文", "一个空白便签"),
        confusion_payloads=(
            "把 {content} 写进便签，不要打开文件",
            "记录 {content}，不是创建日历事件",
            "用普通便签保存 {content}，不要启动终端",
        ),
    ),
    "password.create": LanguageSpec(
        verbs=("保存凭据", "存入密码保险箱", "创建密码条目"),
        payloads=(
            "：网站 {url}，用户名 {username}，密码 {secret}",
            "，地址 {url}，账号 {username}，Secret {secret}",
            "，URL {url}，user {username}，password {secret}",
        ),
        values={
            "url": ("<URL>",),
            "username": ("<USERNAME>",),
            "secret": ("<SECRET>",),
        },
        missing_payloads=("但还没有密码", "，账号信息稍后填写", "一个空凭据条目"),
        confusion_payloads=(
            "不要解析，把网站 {url}、账号 {username} 和密码 {secret} 存进保险箱",
            "保存 {url} 的 {username} 与 {secret}，不要写普通便签",
            "把 {secret} 作为 {url} 的密码保存，不是生成新密码",
        ),
    ),
    "file.open": LanguageSpec(
        verbs=("打开", "定位并打开", "在编辑器查看"),
        payloads=("文件 {path}", " {path}", "工作区里的 {path}"),
        values={
            "path": (
                "README.md",
                "src/main.rs",
                "src/tools.rs",
                "docs/design.md",
                "Cargo.toml",
                "notes/today.md",
                "config/app.toml",
                "src/git_status.rs",
                "terminal.md",
                "日程/九月.md",
                "assets/logo.png",
                "src/studio/mod.rs",
            )
        },
        missing_payloads=("一个文件", "工作区文档", "但还没说文件名"),
        confusion_payloads=(
            "打开名为 {path} 的文件，不要启动终端",
            "在编辑器查看 {path}，不是执行 Git 命令",
            "只打开 {path}，不要搜索文件内容",
        ),
    ),
    "file.search": LanguageSpec(
        verbs=("搜索文件", "查找", "在工作区找"),
        payloads=(" {query}", "名称包含 {query} 的文件", "与 {query} 有关的路径"),
        values={
            "query": (
                "README",
                "配置",
                "terminal.open",
                "Git 状态",
                "calendar",
                "password",
                "smart input",
                "白板",
                "TODO",
                "main.rs",
                "settings",
                "JWT",
            )
        },
        missing_payloads=("一个文件", "工作区内容", "但关键词还没给"),
        confusion_payloads=(
            "在文件列表里搜索 {query}，不是执行命令",
            "查找包含 {query} 的路径，不要打开终端",
            "只搜索 {query} 文件，不创建便签",
        ),
    ),
    "calendar.note.create": LanguageSpec(
        verbs=("创建日历笔记", "写入日历", "添加日期笔记"),
        payloads=("，日期 {date}，内容 {markdown}", "：{date} 记录 {markdown}", "到 {date}，正文 {markdown}"),
        values={
            "date": (
                "2026-09-01",
                "2026-09-02",
                "2026-09-15",
                "2026-10-01",
                "2026-12-31",
                "2027-01-01",
            ),
            "markdown": (
                "发布新版本",
                "团队周会",
                "整理 Git 分支",
                "复盘本周工作",
                "准备产品演示",
                "更新项目计划",
            ),
        },
        missing_payloads=("但还没定日期和内容", "，日期稍后补", "一个未完成的日历记录"),
        confusion_payloads=(
            "把 {markdown} 写到 {date} 的日历，不要建普通便签",
            "在 {date} 记录 {markdown}，不是换算时间戳",
            "创建 {date} 的日历笔记 {markdown}，不要打开文件",
        ),
    ),
    "whiteboard.element.create": LanguageSpec(
        verbs=("在白板创建", "画到白板", "新增白板元素"),
        payloads=(" {element}", "一个 {element}", "：{element}"),
        values={
            "element": (
                "蓝色矩形",
                "写着 JWT 的文本框",
                "连接两个节点的箭头",
                "项目流程图节点",
                "黄色圆形",
                "Git 发布步骤",
                "终端结构示意图",
                "日历计划卡片",
                "文件搜索流程",
                "红色菱形",
            )
        },
        missing_payloads=("一个元素", "新图形但还没说类型", "空白对象"),
        confusion_payloads=(
            "在白板创建 {element}，不是写文档",
            "画一个 {element}，不要创建普通便签",
            "只新增 {element} 到白板，不打开文件",
        ),
    ),
    "git.status": LanguageSpec(
        verbs=("查看 Git 状态", "读取工作区变更", "显示版本库状态"),
        payloads=("", "和未提交文件", "但不要修改任何内容"),
        values={},
        missing_payloads=(),
        confusion_payloads=(
            "只查看 Git 状态，不要打开终端",
            "显示未提交变更，不要执行提交",
            "读取版本库状态，不是打开 git-status.md",
        ),
    ),
    "terminal.open": LanguageSpec(
        verbs=("打开终端", "新建终端会话", "启动一个 PTY"),
        payloads=("", "面板", "并聚焦输入区"),
        values={},
        missing_payloads=(),
        confusion_payloads=(
            "打开终端，不是打开 terminal.md 文件",
            "新建 PTY 会话，不要查看 Git 状态",
            "启动终端面板，不是搜索 terminal.open",
        ),
    ),
}

ABSTAIN_TOPICS = (
    "今天的天气",
    "晚饭吃什么",
    "Rust 的所有权",
    "团队协作",
    "产品方向",
    "电影推荐",
    "学习计划",
    "代码风格",
    "设计理念",
    "项目历史",
    "工具路由",
    "人工智能",
    "时间管理",
    "工作习惯",
    "文档结构",
    "命名方式",
    "需求背景",
    "用户体验",
    "性能问题",
    "安全原则",
)
ABSTAIN_QUALIFIERS = (
    "基本概念",
    "优缺点",
    "可能方案",
    "背景信息",
    "为什么",
    "常见做法",
    "风险",
    "未来方向",
)
ABSTAIN_PATTERNS = {
    "train": (
        "请解释 {topic} 的 {qualifier}，不要执行工具",
        "关于 {topic}，我只是问 {qualifier}",
        "先讨论 {topic} 的 {qualifier}，什么都别改",
        "我还没决定怎么处理 {topic}，只想了解 {qualifier}",
        "取消与 {topic} 有关的操作，说明 {qualifier}",
    ),
    "validation": (
        "暂时不要采取动作，讲讲 {topic} 的 {qualifier}",
        "这只是问题：{topic} 有哪些 {qualifier}",
        "先别打开任何东西，我想知道 {topic} 的 {qualifier}",
        "不需要工具结果，只分析 {topic} 的 {qualifier}",
        "停止操作并解释 {topic} 的 {qualifier}",
    ),
    "test": (
        "不要创建或打开内容，只说明 {topic} 的 {qualifier}",
        "我在闲聊 {topic}，想听听 {qualifier}",
        "没有明确操作，请介绍 {topic} 的 {qualifier}",
        "先保持现状，再谈 {topic} 的 {qualifier}",
        "算了，改为解释 {topic} 的 {qualifier}",
    ),
}


def parse_args() -> argparse.Namespace:
    """parse_args 解析数据生成参数。"""

    parser = argparse.ArgumentParser(description="生成智能工具路由训练数据")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--refresh-actions", action="store_true")
    parser.add_argument(
        "--studio-root",
        type=Path,
        help="刷新动作 Schema 时使用的 lance-studio 仓库路径",
    )
    return parser.parse_args()


def render_pattern(pattern: str, values: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    """render_pattern 渲染模板并返回字符级槽位范围。"""

    pieces: list[str] = []
    slots: list[dict[str, Any]] = []
    cursor = 0
    length = 0
    for match in PLACEHOLDER.finditer(pattern):
        literal = pattern[cursor : match.start()]
        pieces.append(literal)
        length += len(literal)
        name = match.group(1)
        value = values[name]
        start = length
        pieces.append(value)
        length += len(value)
        slots.append({"name": name, "start": start, "end": length, "value": value})
        cursor = match.end()
    pieces.append(pattern[cursor:])
    return "".join(pieces), slots


def expand_family(
    *,
    action: ActionDefinition,
    split: str,
    family: str,
    pattern: str,
    values: dict[str, tuple[str, ...]],
    missing_fields: tuple[str, ...] = (),
    limit: int = 6,
    seed: int,
) -> list[dict[str, Any]]:
    """expand_family 在单个模板族内部生成稳定的词面变化。"""

    names = tuple(dict.fromkeys(PLACEHOLDER.findall(pattern)))
    undeclared = set(names) - set(values)
    if undeclared:
        raise ValueError(f"模板 {family} 缺少值域: {sorted(undeclared)}")
    combinations = itertools.product(*(values[name] for name in names)) if names else [()]
    candidates: list[tuple[str, list[dict[str, Any]]]] = []
    for combination in combinations:
        selected = dict(zip(names, combination))
        for tail in TAILS:
            text, slots = render_pattern(pattern + tail, selected)
            candidates.append((text, slots))

    random.Random(f"{seed}:{family}").shuffle(candidates)
    records = []
    for text, slots in candidates[:limit]:
        for slot in slots:
            parameter_kind = action.parameters.get(slot["name"])
            if parameter_kind == "secret_handle" and slot["value"] not in ALLOWED_SECRET_VALUES:
                raise ValueError(f"敏感模板 {family} 包含非占位符槽位")
        digest = hashlib.sha256(f"{action.id}\0{family}\0{text}".encode()).hexdigest()[:20]
        context_hash = int(digest[:8], 16)
        records.append(
            {
                "id": digest,
                "family": family,
                "text": text,
                "context": {
                    "active_mode": ("workspace", "editor", "quick_access")[context_hash % 3],
                    "has_selection": bool((context_hash // 3) % 2),
                },
                "tool": action.id,
                "slots": slots,
                "missing_fields": list(missing_fields),
                "risk": action.risk,
                "sensitivity": action.sensitivity,
                "split": split,
            }
        )
    return records


def generate_action_records(action: ActionDefinition, seed: int) -> list[dict[str, Any]]:
    """generate_action_records 生成单个动作的正常、缺参和混淆样本。"""

    spec = LANGUAGE_SPECS[action.id]
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        for shape_index, shape in enumerate(SPLIT_SHAPES[split]):
            for payload_index, payload in enumerate(spec.payloads):
                verb = spec.verbs[(shape_index + payload_index) % len(spec.verbs)]
                pattern = shape.format(verb=verb, payload=payload)
                family = f"{action.id}:normal:{split}:{shape_index}:{payload_index}"
                records.extend(
                    expand_family(
                        action=action,
                        split=split,
                        family=family,
                        pattern=pattern,
                        values=spec.values,
                        seed=seed,
                    )
                )

        for index, payload in enumerate(spec.confusion_payloads):
            prefix = ("明确要求：", "容易混淆，但请", "注意意图，")[SPLITS.index(split)]
            family = f"{action.id}:confusion:{split}:{index}"
            records.extend(
                expand_family(
                    action=action,
                    split=split,
                    family=family,
                    pattern=prefix + payload,
                    values=spec.values,
                    seed=seed,
                )
            )

        for index, payload in enumerate(spec.missing_payloads):
            shape = SPLIT_SHAPES[split][index % len(SPLIT_SHAPES[split])]
            verb = spec.verbs[index % len(spec.verbs)]
            family = f"{action.id}:missing:{split}:{index}"
            records.extend(
                expand_family(
                    action=action,
                    split=split,
                    family=family,
                    pattern=shape.format(verb=verb, payload=payload),
                    values=spec.values,
                    missing_fields=action.required_parameters,
                    seed=seed,
                )
            )
    return records


def generate_abstain_records(split: str, seed: int) -> list[dict[str, Any]]:
    """generate_abstain_records 生成提问、否定、取消和无工具意图样本。"""

    records: list[dict[str, Any]] = []
    values = {"topic": ABSTAIN_TOPICS, "qualifier": ABSTAIN_QUALIFIERS}
    abstain = ActionDefinition(
        id="abstain",
        description="证据不足时拒绝选择工具",
        risk="read_only",
        sensitivity="public",
        parameters={},
        required_parameters=(),
        agent_capability=None,
        developer_tool=None,
    )
    for index, pattern in enumerate(ABSTAIN_PATTERNS[split]):
        family = f"abstain:{split}:{index}"
        family_records = expand_family(
            action=abstain,
            split=split,
            family=family,
            pattern=pattern,
            values=values,
            limit=180,
            seed=seed,
        )
        for record in family_records:
            record["slots"] = []
        records.extend(family_records)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """write_jsonl 稳定写入一个数据集切分。"""

    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    """main 导出动作协议并生成三个互斥模板族数据集。"""

    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    actions_path = output_dir / "actions.json"
    schema = load_action_schema(
        actions_path,
        refresh=args.refresh_actions,
        studio_root=args.studio_root,
    )
    schema_ids = set(schema.by_id)
    language_ids = set(LANGUAGE_SPECS)
    if schema_ids != language_ids:
        raise ValueError(
            "语言模板必须完整覆盖权威动作 Schema；"
            f"缺少={sorted(schema_ids - language_ids)}，多余={sorted(language_ids - schema_ids)}"
        )

    records: list[dict[str, Any]] = []
    for action in schema.actions:
        records.extend(generate_action_records(action, args.seed))
    for split in SPLITS:
        records.extend(generate_abstain_records(split, args.seed))

    seen_texts: dict[str, tuple[str, str]] = {}
    split_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for record in records:
        previous = seen_texts.get(record["text"])
        if previous is not None and previous[0] != record["tool"]:
            raise ValueError(
                f"相同文本不能对应不同动作: {record['text']} ({previous[0]}, {record['tool']})"
            )
        if previous is not None and previous[1] != record["split"]:
            raise ValueError(f"相同文本跨数据集泄漏: {record['text']}")
        if previous is not None:
            continue
        seen_texts[record["text"]] = (record["tool"], record["split"])
        split_records[record["split"]].append(record)

    family_splits: dict[str, str] = {}
    for split, split_items in split_records.items():
        for record in split_items:
            prior = family_splits.setdefault(record["family"], split)
            if prior != split:
                raise ValueError(f"模板族跨数据集泄漏: {record['family']}")
        random.Random(args.seed + SPLITS.index(split)).shuffle(split_items)
        write_jsonl(output_dir / f"{split}.jsonl", split_items)

    total = sum(len(items) for items in split_records.values())
    abstain = sum(
        1 for items in split_records.values() for record in items if record["tool"] == "abstain"
    )
    abstain_ratio = abstain / total
    minimum_samples = len(schema.actions) * 300
    maximum_samples = len(schema.actions) * 500
    if not minimum_samples <= total <= maximum_samples:
        raise ValueError(
            f"扩展数据量必须为 {minimum_samples:,}～{maximum_samples:,}，当前为 {total}"
        )
    if abstain_ratio < 0.30:
        raise ValueError(f"abstain 比例不得低于 30%，当前为 {abstain_ratio:.2%}")

    file_hashes = {
        f"{split}.jsonl": sha256_file(output_dir / f"{split}.jsonl") for split in SPLITS
    }
    version_digest = hashlib.sha256(
        "".join(file_hashes[name] for name in sorted(file_hashes)).encode()
    ).hexdigest()
    manifest = {
        "dataset_generator_version": DATASET_GENERATOR_VERSION,
        "dataset_version": f"v{DATASET_GENERATOR_VERSION}-{version_digest[:12]}",
        "schema_version": schema.schema_version,
        "schema_sha256": schema.sha256,
        "action_count": len(schema.actions),
        "seed": args.seed,
        "samples": total,
        "abstain_ratio": abstain_ratio,
        "files": file_hashes,
        "splits": {
            split: {
                "samples": len(items),
                "tools": count_by(items, "tool"),
                "families": len({record["family"] for record in items}),
            }
            for split, items in split_records.items()
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(
        f"dataset={manifest['dataset_version']} schema={schema.schema_version} "
        f"seed={args.seed} samples={total} abstain={abstain_ratio:.2%}"
    )


if __name__ == "__main__":
    main()
