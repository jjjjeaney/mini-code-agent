"""工具层的安全校验。

路径：把模型给的路径锁在指定目录里，分两步各管各的：

    resolved = resolve_within(root, raw)   # 只解析
    ok = is_within(root, resolved)         # 只判断

内容：判断一段字节是不是文本。这部分全是纯函数，只吃 bytes 不吃路径，
所以不用起临时文件就能测；开文件、采样是读工具的活。

限额：单次读多大的文件、留多少字符给模型上下文。

过滤：列目录时哪些文件/目录根本不给模型看。
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path

# 单文件大小上限：超过就拒绝读取整文件。教学版先用 256 KiB，后面再做 offset/limit。
MAX_READ_BYTES = 256 * 1024

# 单次读进上下文的字符数上限。超了就截断，并在正文里说明被截了。
# 注意是字符不是字节——8000 个中文字符在 UTF-8 下要占 24000 字节。
MAX_TEXT_CHARS = 12000

EXCLUDED_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".nox", ".venv", "venv", "env", "node_modules", ".next",
    ".nuxt", "dist", "build", ".turbo", ".idea", ".vscode", "target",
    "bin", "obj", ".cache", "tmp", "temp", "htmlcov",
}

EXCLUDED_FILES = {
    ".coverage", ".env", ".env.local", ".env.production", ".env.development",
    ".DS_Store", "Thumbs.db",
}

# 敏感文件靠整名列举必然漏——.env.staging、.env.bak、deploy.key 是数不完的。
# 用通配兜住整个家族。代价是 .env.example 这类模板也会被滤掉，不过那是安全的方向。
SENSITIVE_FILE_PATTERNS = (
    ".env*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "credentials*.json",
    "service-account*.json",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
)

# 敏感目录：里面的东西一律不给读。.git 单列出来，是因为它的 config 里经常嵌着
# 带 token 的 remote URL——那是很常见的一种泄露。
SENSITIVE_DIRS = {".git", ".ssh", ".aws", ".gnupg"}

class OutOfSandboxError(Exception):
    """模型要访问的路径落在了允许的目录之外。

    is_within 只回 True/False，不抛异常；这个类型留给调用方在判否时抛。
    报错原文会经 dispatch 原样交给模型，所以说清"哪越界了、只能去哪"是调用方的事。
    """


class NotTextFileError(Exception):
    """这段内容看着不是能当文本读的东西。

    ensure_text_file 只回 True/False，不抛异常；这个类型留给调用方在判否时抛。
    """


class UndecodableTextError(Exception):
    """内容是文本，但不是我们认得的那种编码，解不开。

    和 NotTextFileError 分开是有意的：那个是"根本不是文字"，这个是"是文字，只是读法
    不对"。模型看到之后该做的事不一样——前者该换个文件，后者该换个编码。
    """


class SensitiveFileError(Exception):
    """这个文件碰不得——里面可能装着密钥、token 或凭据。

    由调用方在 is_sensitive 判真时抛。拦在读之前，内容根本不会被读进来。
    """


def resolve_within(root: Path, raw: str | Path) -> Path:
    """把模型给的路径解析成绝对路径。只解析，不判断越界。

    绝对路径喂给 / 会整个替换掉左边，相对路径才接在右边，两种写法都收。
    `..` 和符号链接都在这一步被抹平，返回的是它实际指向的位置。
    """
    return (Path(root) / raw).resolve()


def is_within(root: Path, candidate: Path) -> bool:
    """判断 candidate 是否落在 root 里面。

    两个参数都必须是解析过的绝对路径：root 用 Path.cwd().resolve() 那种，
    candidate 用 resolve_within() 的返回值。

    传没解析过的路径进来会判错，而且错的方向是危险的：candidate 里带着 `..` 时，
    is_relative_to 只看字面分量，`root/a/../../evil` 会被判成"在 root 里"，
    实际却指到外面去了。
    """
    return candidate.is_relative_to(root)


# 带 BOM 的几种文本。长 BOM 必须排在短的前面——`\xff\xfe` 是 `\xff\xfe\x00\x00`
# 的前缀，顺序反了 UTF-32 会被认成 UTF-16。
_BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xfe\xff", "utf-16-be"),
    (b"\xff\xfe", "utf-16-le"),
)


def detect_encoding(sample: bytes) -> str | None:
    """样本开头带 BOM 的话，返回该用的编码；认不出返回 None。"""
    for bom, encoding in _BOMS:
        if sample.startswith(bom):
            return encoding
    return None


def ensure_text_file(sample: bytes) -> bool:
    """这段字节能当文本读吗。是就回 True，看着像二进制就回 False。

    只看 NUL 字节：正常文本里不会出现它，而图片、压缩包、可执行文件里到处都是。
    git 判断二进制用的也是这套办法。

    唯一的例外是带 BOM 的 UTF-16 / UTF-32——它们每个 ASCII 字符后面都跟一个 NUL，
    光看字节和二进制没区别。所以先认 BOM，认出来直接放行。

    故意不在这里试 UTF-8 解码：解不开不等于二进制。GBK 中文就是文本，只是不按
    UTF-8 存，把它判成二进制是误诊——正确的说法是"编码不对"。能不能解码，是读
    文件那一步该单独报的事。

    这里既不开文件也不抛异常，提示语由调用方看着办。
    """
    if detect_encoding(sample) is not None:
        return True
    return b"\x00" not in sample


def ensure_within_size(path: Path, max_bytes: int = MAX_READ_BYTES) -> None:
    # 整文件读取的硬上限。Day 3 不做 offset/limit，超限直接拒绝，
    # 课后挑战里再补按行读。
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"file too large: {size} bytes > {max_bytes}; "
            f"read a smaller file or use grep instead"
        )


def truncate_text(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    """把文本裁到前 limit 个字符。真裁了的话，末尾附一句说明。

    按字符数而不是字节数裁：同样 8000 个字，中文在 UTF-8 下要 24000 字节，
    按字节裁会让中文文件比英文文件早得多地被砍掉。

    静默截断是最糟的——模型会以为文件就这么长，然后基于错的认知去改代码。
    所以"被截了、原文多长"要写进正文，跟着内容一起交给模型。
    """
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n[已截断：全文共 {len(text)} 个字符，这里只给了前 {limit} 个]"


def should_skip(path: Path) -> bool:
    """列目录时，这个文件或目录要不要滤掉。

    滤的是两类东西，动机不一样：

    - **敏感文件**：.env、私钥、凭据这类，看见了就等于把它发给了模型
    - **噪音**：依赖、缓存、构建产物，看了白占上下文

    按名字匹配，不管在第几层——`config/.env` 和根目录的 `.env` 一样敏感。
    文件名统一转小写再比：Windows 上本来就不区分大小写，Linux 上顶多多滤掉几个
    大写变体，在安全的方向上偏一点不要紧。

    目录只按整名匹配，文件既按整名也套通配。

    注意这只是让 `list_files` 不列出来。模型照样可以指名去读 `.env`——
    要不要禁止读，是另一层的决定。
    """
    name = path.name.lower()
    if path.is_dir():
        return name in EXCLUDED_DIRS
    return name in EXCLUDED_FILES or any(
        fnmatchcase(name, pattern) for pattern in SENSITIVE_FILE_PATTERNS
    )


def is_sensitive(path: Path) -> bool:
    """这个路径碰不碰得——里面可能装着密钥、token 或凭据。

    和 should_skip 的区别就是它存在的理由：那个管"列目录时藏不藏"，这个管"连读都不许"。
    噪音文件（.coverage 之类）该藏但不该拦，所以两个判断不能合并成一个。

    看路径的**每一段**，不只是文件名：`.git/config` 里可能嵌着带 token 的 remote URL，
    而它的文件名 `config` 本身看不出任何问题——只看最后一段会漏掉。
    """
    if any(part.lower() in SENSITIVE_DIRS for part in path.parts[:-1]):
        return True
    name = path.name.lower()
    return name in SENSITIVE_DIRS or any(
        fnmatchcase(name, pattern) for pattern in SENSITIVE_FILE_PATTERNS
    )


