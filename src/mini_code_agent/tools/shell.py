"""命令执行工具：把命令放进一次性 docker 容器里跑。

为什么要容器：模型能跑任意 shell 命令，那就等于把整个宿主机交给它。容器把"它能碰
什么"压到最小——只读的根文件系统、没有网络、只有挂载进去的项目目录可写、跑完就删。

**沙箱才是这里的安全边界。** 下面的命令风险分类和输出涂黑都只是降低概率的启发式，
别拿它们当防线——它们挡的是"手滑"和"顺手读一下"，挡不住真想绕的。
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from mini_code_agent.tools.base import (
    ContainerStartFailedError,
    DangerousCommandError,
    DockerUnavailableError,
    ShellDisabledError,
    ToolContext,
)
from mini_code_agent.tools.safety import deny_reason, redact_sensitive

# 沙箱镜像：仓库根目录那个 Dockerfile 构建出来的（python:3.12-slim 加 git）。带 git 是为了
# 模型能读本地历史（status / diff / log）——写历史那条路本来就被挡着，见下面 .git 的只读挂载。
# 它**不会**被自动拉取（见 --pull=never），所以得先在仓库根目录构建一次。
DEFAULT_IMAGE = "mca_sandbox"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 300

# 每条流最多留多少字节。超了继续读但不再留——不继续读的话子进程写满管道会卡死。
MAX_OUTPUT_BYTES = 8_000

SHELL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "要执行的 shell 命令，交给容器里的 /bin/sh -c 跑",
        },
        "timeout_seconds": {
            "type": "integer",
            "description": (
                f"最多跑多少秒，默认 {DEFAULT_TIMEOUT_SECONDS}，"
                f"上限 {MAX_TIMEOUT_SECONDS}。到点容器会被终止"
            ),
        },
    },
    "required": ["command"],
}


def bash(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """在一次性容器里跑一条命令，返回退出码和输出。

    容器能碰到的只有 ctx.cwd 挂进去的 /workspace，其它一概没有：根文件系统只读、
    没有网络、没有 capabilities、以非 root 身份跑、进程数和内存都有上限。

    时间到了或者命令卡住了，**杀掉 docker 客户端是不够的**——容器会继续跑。所以超时
    路径会拿 --cidfile 里的 id 去 docker rm -f。（实测过：杀掉客户端之后容器还在。）

    输出会先涂黑（含密钥关键词的整行换掉），再按上限截断。

    命令先过一道危险命令黑名单（`safety.py` 的 `DENY_PATTERNS`）：命中就直接拒，不执行。
    那是**启发式、不是边界**——边界是下面这个容器。
    """
    if not ctx.allow_shell:
        raise ShellDisabledError(
            "这个 agent 没开命令执行能力。要走这条路，得在建 ToolContext 时把 "
            "allow_shell 打开（CLI 上是 --shell）。"
        )

    command = kwargs["command"]
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command 得是非空的字符串。")

    # 危险命令黑名单。放在这儿（工具的闸门里）意味着它**在权限闸门之后**：用户先被问过、
    # 点了头，这儿照样能拒——"批准"不等于"这件事就安全"，和路径围栏、读-before-写一个道理。
    # 好处是用户能看见模型试过什么（那条命令会在 trace 里），坏处是他可能白点一次头。
    denied = deny_reason(command)
    if denied is not None:
        raise DangerousCommandError(
            f"这条命令被危险命令黑名单拦下了（{denied}），没有执行过。"
            f"换个不危险的做法；别改写成别的样子再试。"
        )

    timeout = _timeout(kwargs.get("timeout_seconds"))

    with tempfile.TemporaryDirectory(prefix="mca-shell-") as workspace:
        cidfile = Path(workspace) / "container.cid"
        argv = _docker_argv(ctx.cwd, command, cidfile)

        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,  # 命令要是去读 stdin，会一直挂着等
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise DockerUnavailableError(
                "没找到 docker 命令。命令执行能力依赖 docker，先装好并让它跑起来。"
            ) from exc

        stdout, stderr, timed_out = _run(process, timeout, cidfile)

    if not timed_out and process.returncode == 125:
        # docker run 自己的退出码 125 表示"容器都没起来"，不是命令的失败——
        # daemon 没起、镜像不存在、参数写错，都走这里。
        said = stderr.decode("utf-8", "replace").strip()
        hint = ""
        if "Unable to find image" in said or "No such image" in said:
            # 沙箱镜像是要自己构建的，而上面又写了 --pull=never，所以这一步特别容易漏；
            # 没有这句提示的话，"容器没起来"看不出是镜像不在。
            hint = (
                f" 沙箱镜像 {DEFAULT_IMAGE} 不在本地——"
                f"先在仓库根目录跑 docker build -t {DEFAULT_IMAGE} ."
            )
        raise ContainerStartFailedError(f"容器没起来，命令没有执行过。docker 说：{said}{hint}")

    return _format(process.returncode, stdout, stderr, timed_out, timeout)


def _timeout(value: Any) -> int:
    """把模型给的超时秒数夹到合理区间。没给或者给得离谱都用默认值。"""
    if value is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"timeout_seconds 得是整数，收到的是 {value!r}") from None
    return max(1, min(seconds, MAX_TIMEOUT_SECONDS))


def _docker_argv(workspace: Path, command: str, cidfile: Path) -> list[str]:
    """拼 docker run 的参数。

    用户命令是作为**一个 argv 元素**传给容器里的 /bin/sh -c 的：宿主机这层没有 shell
    参与，不存在注入；容器里再解释一遍是刻意的（模型就是要跑 shell 命令），边界由
    容器承担。
    """
    # 项目目录挂进去、可写（agent 就是要改代码）。as_posix() 是为了 Windows：
    # docker CLI 认 D:/... 这种，不认反斜杠。
    mounts = ["--mount", f"type=bind,src={workspace.as_posix()},dst=/workspace"]

    git_dir = workspace / ".git"
    if git_dir.exists():
        # .git 单独盖一层**只读**。项目目录整个是可写的，但仓库目录不该被写——里面能放的
        # 东西会在**宿主机**上生效，比如一个 pre-commit 脚本，会在你自己下次提交时被执行。
        # 只读之后读历史照旧（status / diff / log），写不进去。
        # 项目没有 .git 时不能挂：docker 会因为"源路径不存在"直接失败，容器都起不来。
        mounts += ["--mount", f"type=bind,src={git_dir.as_posix()},dst=/workspace/.git,ro"]

    return [
        "docker",
        "run",
        "--rm",
        "--name", f"mca-{uuid.uuid4().hex[:10]}",
        "--cidfile", str(cidfile),
        # 不让 docker 去拉镜像：拉取是 daemon 在宿主机上做的，--network none 管不着它。
        # 代价是镜像必须已经在本地，否则会以 125 失败——那条路有单独的报错。
        "--pull=never",
        # 没有 init 的话，容器里 fork 出来的子进程没人收尸，会留一堆僵尸。
        "--init",
        "--network", "none",
        "--read-only",
        # 挂载在上面拼好了：项目目录可写，.git 那层只读。
        *mounts,
        "-w", "/workspace",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "1000:1000",
        "--pids-limit", "128",
        "--memory", "2g",
        "--cpus", "2",
        "--env", "HOME=/tmp",
        "--env", "TMPDIR=/tmp",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "GIT_CONFIG_GLOBAL=/dev/null",
        "--env", "GIT_CONFIG_NOSYSTEM=1",
        "--env", "GIT_TERMINAL_PROMPT=0",
        # 上面那条只读的 .git 会让 git status 抢不到 index.lock（它默认想顺手刷一下 index
        # 缓存），这个变量就是 git 给这种场景准备的开关：可有可无的锁就别去拿了。
        "--env", "GIT_OPTIONAL_LOCKS=0",
        # git 会拒绝操作"属主不是自己"的目录，挂进来的 /workspace 正是这种；而上面又把
        # global config 指到了 /dev/null，所以只能靠这组变量把 safe.directory 加回来。
        "--env", "GIT_CONFIG_COUNT=1",
        "--env", "GIT_CONFIG_KEY_0=safe.directory",
        "--env", "GIT_CONFIG_VALUE_0=/workspace",
        DEFAULT_IMAGE,
        "/bin/sh",
        "-c",
        command,
    ]


def _run(
    process: subprocess.Popen[bytes], timeout: int, cidfile: Path
) -> tuple[bytes, bytes, bool]:
    """读输出、等它结束；超时就把客户端和容器一起收掉。"""
    sinks = [{"kept": b"", "total": 0}, {"kept": b"", "total": 0}]
    readers = [
        threading.Thread(target=_drain, args=(process.stdout, sinks[0])),
        threading.Thread(target=_drain, args=(process.stderr, sinks[1])),
    ]
    # stdout 和 stderr 必须**同时**有人读：只读一条的话，另一条写满管道（Linux 上
    # 大概 64KB）子进程就卡在 write 上，两条都不动了。
    for reader in readers:
        reader.start()

    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        process.wait()
        _force_remove(cidfile)

    for reader in readers:
        reader.join(timeout=10)

    return sinks[0]["kept"], sinks[1]["kept"], timed_out


def _drain(stream: Any, sink: dict[str, Any]) -> None:
    """把一条管道读到 EOF，但只留前 MAX_OUTPUT_BYTES 字节。

    超了之后**还得继续读**（读到了直接扔），不能停——停了子进程会卡在写那一头。
    """
    kept = bytearray()
    total = 0
    while True:
        chunk = stream.read1(8192)
        if not chunk:
            break
        total += len(chunk)
        if len(kept) < MAX_OUTPUT_BYTES:
            kept.extend(chunk[: MAX_OUTPUT_BYTES - len(kept)])
    stream.close()
    sink["kept"] = bytes(kept)
    sink["total"] = total


def _force_remove(cidfile: Path) -> None:
    """照着 cidfile 把容器强删掉。

    这条是给超时路径用的：杀掉 docker 客户端并不会让容器停下来（实测确认过），
    --rm 也只在容器自己正常退出时才生效。所以必须显式删。
    """
    try:
        container_id = cidfile.read_text(encoding="utf-8").strip()
    except OSError:
        return
    if not container_id:
        return
    subprocess.run(
        ["docker", "rm", "-f", container_id], capture_output=True, timeout=60, check=False
    )


def _format(
    exit_code: int, stdout: bytes, stderr: bytes, timed_out: bool, timeout: int
) -> str:
    """把结果拼成给模型看的文本。"""
    out_text, out_hidden = redact_sensitive(stdout.decode("utf-8", "replace"))
    err_text, err_hidden = redact_sensitive(stderr.decode("utf-8", "replace"))

    sections: list[str] = []
    if timed_out:
        sections.append(f"[已超时：{timeout} 秒没跑完，容器已经终止]")
    else:
        sections.append(f"[退出码 {exit_code}]")

    if out_text.strip():
        sections.append(f"--- stdout ---\n{out_text}")
    if err_text.strip():
        sections.append(f"--- stderr ---\n{err_text}")
    if not out_text.strip() and not err_text.strip():
        sections.append("(命令没有任何输出)")

    if out_hidden + err_hidden:
        sections.append(f"[已涂黑 {out_hidden + err_hidden} 行：里面出现了疑似密钥的关键词]")
    if len(stdout) >= MAX_OUTPUT_BYTES or len(stderr) >= MAX_OUTPUT_BYTES:
        sections.append(
            f"[输出被截断：每条流只保留前 {MAX_OUTPUT_BYTES} 字节。"
            f"想看全就用管道收窄范围，比如 | tail -50]"
        )

    return "\n\n".join(sections)
