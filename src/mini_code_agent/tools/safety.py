"""工具层的安全校验。

路径：把模型给的路径锁在指定目录里，分两步各管各的：

    resolved = resolve_within(root, raw)   # 只解析
    ok = is_within(root, resolved)         # 只判断

内容：判断一段字节是不是文本。这部分全是纯函数，只吃 bytes 不吃路径，
所以不用起临时文件就能测；开文件、采样是读工具的活。

限额：单次读多大的文件、留多少字符给模型上下文。

过滤：列目录时哪些文件/目录根本不给模型看。

命令：一条危险命令黑名单。挡的是"顺手就来一下"的那类极度危险动作，**不是边界**。
"""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from pathlib import Path

# 单文件大小上限：超过就拒绝读取整文件。教学版先用 256 KiB，后面再做 offset/limit。
MAX_READ_BYTES = 256 * 1024

# 单次读进上下文的字符数上限（**通用**那一档）。超了就截断，并在正文里说明被截了。
# 注意是字符不是字节——20000 个中文字符在 UTF-8 下要占 24000 字节。
#
# **它不只是工具结果的上限**：系统提示词（AGENTS.md 正文、git 改动清单）和权限确认里的
# diff 也拿它当默认值。所以调工具那两档时不要动这个数——那会连带把项目规矩文件和用户
# 批准前看到的改动一起截短。那两处各有各的尺度，和工具结果不是一回事。
MAX_TEXT_CHARS = 20000

# 读文件那一档。比别处宽：文件是模型要去改的东西，截在中间它就得再跑一次工具；而且
# read_file 自带 start_line / end_line，宽不代表一定会读满，想少读它自己会说。
MAX_READ_FILE_CHARS = 50_000

# 其它工具结果那一档。比别处紧：它们大多是"先看个大概"（列目录、搜内容、取网页），
# 真要更多，换个更窄的问法比塞一整条进上下文划算。
MAX_TOOL_CHARS = 5_000

# 落盘之后正文里留多少（见 tools/spill.py）。它唯一的用处是让模型判断"要不要去取"，
# 不是让它在预览里干活——所以比上面两档都短得多。
PREVIEW_CHARS = 1_500

EXCLUDED_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".nox", ".venv", "venv", "env", "node_modules", ".next",
    ".nuxt", "dist", "build", ".turbo", ".idea", ".vscode", "target",
    "bin", "obj", ".cache", "tmp", "temp", "htmlcov",
    # 工具自己的痕迹（改文件前的备份），不该出现在模型的视野里
    ".agent",
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
# 连读都不许的地方。`.agent` 是自己人写的（改前备份、会话 transcript），但里面装的是**模型
# 看过的一切**——包括命令输出里涂黑那层漏掉的密钥。所以这儿不看"谁写的"，只看"可能有什么"。
SENSITIVE_DIRS = {".agent", ".git", ".ssh", ".aws", ".gnupg"}

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
    """这个路径碰不得——可能是装着密钥、token 或凭据的文件，也可能是 .git 这种目录。

    由调用方在 is_sensitive 判真时抛。读和写都用到：读是怕内容被带出去；写是怕覆盖掉
    凭据，更怕在那种位置新建出别的东西——比如 .git/hooks/ 下面放个脚本。
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


def resolve_inside(root: Path, raw: str | Path) -> Path:
    """解析路径并确认它落在 root 里面。存不存在、是不是目录由调用方各自判。

    这是安全闸门，几个工具都要走——**所以只有这一份**。抄成多份的话，以后补一条校验只会补在
    其中一份上，而安全校验有多份就等于只会被修好一份。

    写文件的工具要允许"还不存在"，所以围栏和"必须存在"是分开的两件事。
    """
    target = resolve_within(root, raw)
    if not is_within(root, target):
        raise OutOfSandboxError(
            f"路径 {str(raw)!r} 指向 {target}，超出了允许的范围 {root}。"
            f"只能访问 {root} 里面的东西。"
        )
    return target


def find_symlink(raw: str | Path, root: Path) -> Path | None:
    """沿着**没解析过**的路径一段段看，返回第一段符号链接。

    为什么不能拿解析后的 target 去判 `is_symlink()`：链接在 resolve_within 那一步就被展开了，
    等拿到 target，"这是链接"这个信息已经没了。只能顺着原始路径重新走一遍。

    和围栏一样，这是安全校验，**只有这一份**。
    """
    unresolved = Path(raw)
    if unresolved.is_absolute():
        try:
            unresolved = unresolved.relative_to(root)
        except ValueError:
            return None  # 压根不在 root 下面，围栏那一步会另行拒绝
    current = root
    for part in unresolved.parts:
        current = current / part
        if current.is_symlink():
            return current
    return None


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
    """**一次读进来**的硬上限：超过了就拒，让调用方换个做法。

    这是"这一次读多少"的闸，不是"这个文件不能碰"：读文件那条路自己还有行号区间
    （`read_file` 的 start_line / end_line），所以超限不是没救，是得换个读法。
    grep 那边用它来**跳过**太大的文件（那边接住异常，不当错误往上抛）。

    报错是给人看、也给模型看的，所以写中文——这个项目里对模型说的话没有英文。
    """
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"{path.name} 太大了：{size} 字节，一次最多读 {max_bytes} 字节。"
            f"读文件的话可以用 start_line / end_line 只读一段，找内容用 grep。"
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


# 命令输出里出现这些就整行涂黑。全是小写，比较时也转小写。
#
# 这是**启发式，不是保证**：不含这些词的密钥照样会漏出去（一串裸的十六进制、
# 一个自定义变量名都拦不住）。它挡的是"顺手 cat 一下 .env"这类最常见的泄露，
# 真正该指望的是"别让容器够得着那些文件"，不是这一层。别把它当安全边界。
SENSITIVE_OUTPUT_PATTERNS = (
    "api",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "private key",
    "authorization",
    "bearer",
    "-----begin",
    # 各家密钥的前缀，比关键词精确一些
    "sk-",
    "ghp_",
    "github_pat_",
    "xoxb-",
    "akia",
)

REDACTED = "[已涂黑]"

# 危险命令黑名单：命中就拒，不再问用户（调用点在 shell.py，在权限闸门**之后**）。
#
# **这是启发式，不是边界。** 命令是文本，绕开它有一百种写法（引号、变量、base64、拆成两条）。
# 真正的边界是容器：只读根、无网络、非 root、只有项目目录可写、跑完就删，还有 `.git` 的只读挂载。
#
# 收什么、不收什么，标准只有一条：**后果不可逆，而且是整块地毁**。
#
#   1. 把整个项目毁掉——那是容器里唯一可写的地方，而且大部分没有备份
#   2. 不可逆地丢掉没提交的工作——这几条不经过文件工具，`.agent/.history` 兜不住
#   3. 设备级破坏 / 拿到宿主机或 root——现在够不着，留着防沙箱参数被改
#   4. 把容器拖死——`--pids-limit` 兜着，同上
#
# **一般的删文件、删目录不收。** `rm -rf build/`、`rm -rf node_modules`、`rm foo.py` 都是正常
# 干活，拦它们只会把黑名单变成噪音。这不是猜的：按"参数里带 `/` 就拦"写，`rm -rf build/` 会被
# 误伤，而真正要命的 `rm -rf /workspace` 反而漏掉——所以下面每条都精确到"打到根上"。
DENY_PATTERNS: tuple[tuple[str, str], ...] = (
    # 1. 把整个项目毁掉
    (
        "删项目根目录（rm -r 打到 /、/workspace、.、..、* 或 $PWD）",
        r"\brm\s+(?:-\S+\s+)*-\S*[rR]\S*(?:\s+-\S+)*\s+"
        r"(?:/(?:workspace)?/?|\.\.?/?|\./\*|\*|\$\{?PWD\}?)(?=\s|$)",
    ),
    (
        "删系统目录（rm -r 打到 /etc、/usr 这类）",
        r"\brm\s+(?:-\S+\s+)*-\S*[rR]\S*(?:\s+-\S+)*\s+"
        r"/(?:etc|usr|var|root|boot|lib|bin|sbin|opt|srv|home)(?:/\S*)?(?=\s|$)",
    ),
    (
        "删掉 git 仓库目录（.git）",
        r"\brm\s+(?:-\S+\s+)*-\S*[rR]\S*(?:\s+-\S+)*\s+\.git(?:/\S*)?(?=\s|$)",
    ),
    ("把项目根整个 find -delete 掉", r"\bfind\s+(?:\.|/workspace)/?\s*(?:-\w+\s+)?-delete\b"),
    # 2. 不可逆地丢掉没提交的工作（这几条没有备份兜底）
    ("清掉未跟踪文件（git clean -f）", r"\bgit\s+clean\b[^|;&]*-[A-Za-z]*f[A-Za-z]*(?=\s|$)"),
    ("丢弃工作区改动（git reset --hard）", r"\bgit\s+reset\s+--hard\b"),
    (
        "丢弃工作区改动（git checkout/restore 打到 .）",
        r"\bgit\s+(?:checkout|restore)\s+(?:--\s+)?\.(?:/\S*)?(?=\s|$)",
    ),
    ("强推（git push --force）", r"\bgit\s+push\b[^|;&]*(?:--force\b|\s-f(?=\s|$))"),
    ("重写历史（git filter-branch / filter-repo）", r"\bgit\s+filter-(?:branch|repo)\b"),
    # 3. 设备级破坏（容器里够不着，防沙箱参数被改）
    ("格式化或抹掉设备（mkfs / wipefs）", r"\b(?:mkfs(?:\.\w+)?|wipefs)\b"),
    ("直写块设备（dd of=/dev/…）", r"\bdd\b[^|;&]*\bof\s*=\s*/dev/"),
    # 4. 拿到宿主机或 root（同上）
    ("拿特权起容器或摸 docker.sock", r"\bdocker\b[^|;&]*--privileged\b|docker\.sock"),
    (
        "把宿主机根挂进容器",
        r"\bdocker\b[^|;&]*(?:-v|--volume)\s+/|--mount\b[^|;&]*(?:src|source)\s*=\s*/",
    ),
    ("提权（sudo / su / pkexec）", r"\b(?:sudo|su|pkexec)\b"),
    # 5. 把容器拖死（pids-limit 兜着，留着防参数被改）
    ("下载就执行（curl / wget | sh）", r"\b(?:curl|wget)\b[^|;&]*\|\s*(?:ba)?sh\b"),
    ("fork bomb", r":\(\)\s*\{.*\|.*&.*\}"),
)


def deny_reason(command: str) -> str | None:
    """这条命令在黑名单上吗？命中就返回那条规则的名字，没有就 None。

    比之前先把命令**整成一行**：行尾续行的反斜杠去掉、换行折成空格。不然 `rm -rf \\` 换行 `/`
    这种写法能溜过去——正则里的 `.` 不跨行。
    """
    flattened = " ".join(re.sub(r"\\\s*\n", " ", command).split())
    for name, pattern in DENY_PATTERNS:
        if re.search(pattern, flattened):
            return name
    return None


def redact_sensitive(text: str) -> tuple[str, int]:
    """把像是装着密钥的行整行涂黑，返回 (处理后的文本, 涂黑了几行)。

    命中就整行换掉，不做"只涂掉值那一段"的花活：整行的说法容易讲清楚、也容易验证
    ——"这一行出现了这些词，就不给模型看"。代价是误伤（README 里提一句 API 也会被
    涂掉），但涂掉一行信息的损失，比漏一个密钥小得多。
    """
    kept: list[str] = []
    hidden = 0
    for line in text.splitlines():
        lowered = line.lower()
        if any(pattern in lowered for pattern in SENSITIVE_OUTPUT_PATTERNS):
            kept.append(REDACTED)
            hidden += 1
        else:
            kept.append(line)
    return "\n".join(kept), hidden


