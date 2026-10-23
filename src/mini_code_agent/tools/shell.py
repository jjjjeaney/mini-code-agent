"""命令执行工具：按 `ctx.sandbox` 跑在一次性 docker 容器里，或直接跑在宿主机上。

docker 是默认模式，也是这里的安全边界：只读根文件系统、没有网络、只有项目目录可写、
跑完就删。`host` 模式没有这层隔离——命令以当前用户身份跑，能访问整个用户环境、网络和
凭据。下面的命令风险分类和输出涂黑都只是降低概率的启发式，两种模式都别拿它们当防线。
"""

from __future__ import annotations

import locale
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mini_code_agent.sandbox import DEFAULT_SANDBOX, parse_sandbox
from mini_code_agent.tools.base import (
    ContainerStartFailedError,
    DangerousCommandError,
    DockerUnavailableError,
    ShellDisabledError,
    ShellUnavailableError,
    ToolContext,
)
from mini_code_agent.tools.safety import deny_reason, redact_sensitive

# 沙箱镜像：仓库根目录那个 Dockerfile 构建出来的（python:3.12-slim 加 git）。带 git 是为了
# 模型能读本地历史（status / diff / log）——写历史那条路本来就被挡着，见下面 .git 的只读挂载。
# 它**不会**被自动拉取（见 --pull=never），所以得先在仓库根目录构建一次。
DEFAULT_IMAGE = "mca_sandbox"
DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 300

# 每条流最多留多少字节。**这不是给模型的限额**，是"我们能留住多少原文"——超了继续读但
# 不再留（不继续读的话子进程写满管道会卡死）。
#
# 留得比过去多得多（以前只有 8000）：留下来的这份会被 registry 整篇落盘，上下文里只放
# 一段预览（见 tools/spill.py）。2 MiB 这个数是有意的——常见的命令输出（测试日志、构建
# 日志）都在几百 KB 以内，留得住就是整条留得住；再往上是几 MB 的机器输出，多半该用管道
# 收窄而不是整篇读，而且这份文本还要过一遍涂黑，留太多只是白花时间。
MAX_CAPTURED_BYTES = 2 * 1024 * 1024

SHELL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": (
                "这条命令要做什么，一句话说清。**写给用户看的**：放行之前要他点头，"
                "而这一行就是他判断的依据——说清意图，别复述命令原文"
            ),
        },
        "command": {
            "type": "string",
            "description": "要执行的 shell 命令；具体由本次运行的沙箱模式决定用哪个 shell 解释",
        },
        "timeout_seconds": {
            "type": "integer",
            "description": (
                f"最多跑多少秒，默认 {DEFAULT_TIMEOUT_SECONDS}，"
                f"上限 {MAX_TIMEOUT_SECONDS}。到点命令进程会被终止"
            ),
        },
    },
    "required": ["description", "command"],
}


def bash(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """按 `ctx.sandbox` 跑一条命令，返回退出码和输出。

    docker 模式下，容器能碰到的只有 ctx.cwd 挂进去的 /workspace，其它一概没有：根文件
    系统只读、没有网络、没有 capabilities、以非 root 身份跑、进程数和内存都有上限。
    超时路径会拿 --cidfile 里的 id 去 docker rm -f。

    host 模式下命令直接以当前用户身份跑在 ctx.cwd 里：没有隔离，能访问整个用户环境、
    网络和凭据。超时路径尽力杀掉进程树（POSIX killpg / Windows taskkill /T）。

    输出会先涂黑（含密钥关键词的整行换掉）。涂黑是**在落盘之前**做的——落下去的那份和
    给模型看的那份必须是同一份，不然落盘就成了绕开涂黑的一条路。

    太长的话由 registry 整篇落盘、上下文里只留预览（这个工具在那边标了 spill）。所以
    这里只留得住 MAX_CAPTURED_BYTES 以内——连这个都超的，才真的丢东西。

    `description` 是模型自己写的"这条命令要做什么"，必填。它**不参与任何判定**（黑名单、
    权限、沙箱都只看 `command`），去处只有一个：用户批准之前看到的那段预览（见
    `prompting.preview_command`）。

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

    # `description` 不是给我们看的元数据，是**给用户看的**：这条命令要跑得他点头，而他手上
    # 只有这一行说明和下面那条命令。空着就等于把那一步判断推回给"读命令原文自己猜"，
    # 而省掉那个正是它存在的理由。它**不参与任何判定**——黑名单、权限、沙箱照旧只看
    # `command`；这里卡一道只是不让模型把它交了差。
    description = kwargs.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            "description 得是非空的字符串：一句话说清这条命令要做什么，它是给用户看的。"
        )

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
    sandbox = parse_sandbox(ctx.sandbox)
    if sandbox == "docker":
        return _bash_in_docker(ctx.cwd, command, timeout)
    return _bash_on_host(ctx.cwd, command, timeout)


def _bash_in_docker(cwd: Path, command: str, timeout: int) -> str:
    """docker 分支：原来的容器执行路径，原样搬过来。"""
    with tempfile.TemporaryDirectory(prefix="mca-shell-") as workspace:
        cidfile = Path(workspace) / "container.cid"
        argv = _docker_argv(cwd, command, cidfile)

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

        def stop_container() -> None:
            # 杀掉 docker 客户端不会让容器停下来，--rm 也只在容器自己正常退出时才生效。
            # 所以先杀客户端、等它退出，再照着 cidfile 把容器强删。
            process.kill()
            process.wait()
            _force_remove(cidfile)

        stdout, stderr, totals, timed_out = _run(process, timeout, stop_container)

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

    return _format(
        process.returncode, stdout, stderr, totals, timed_out, timeout, sandbox="docker"
    )


def _bash_on_host(cwd: Path, command: str, timeout: int) -> str:
    """host 分支：直接在当前用户环境里跑，没有容器隔离。"""
    if not cwd.is_dir():
        raise ValueError(f"工作目录不存在：{cwd}")
    argv, encoding = _host_argv(command)
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=None,
            # POSIX 下让 shell 单独成为一个进程组，超时时才能整组杀掉。
            start_new_session=(os.name != "nt"),
        )
    except FileNotFoundError as exc:
        raise ShellUnavailableError(
            "host 模式没找到可用的 shell。优先用 bash；没有 bash 时 Windows 用 COMSPEC、"
            "POSIX 用 /bin/sh。确认 PATH 里有 bash，或者改用 --sandbox docker。"
        ) from exc

    stdout, stderr, totals, timed_out = _run(
        process, timeout, lambda: _kill_tree(process)
    )
    return _format(
        process.returncode,
        stdout,
        stderr,
        totals,
        timed_out,
        timeout,
        sandbox="host",
        encoding=encoding,
    )


def _host_argv(command: str) -> tuple[list[str], str]:
    """host 模式下用哪个 shell，以及输出按什么编码解码。

    优先 bash（Git Bash / WSL 都算），它和 docker 里那个 `/bin/sh -c` 的语义最接近。
    找不到再退回系统 shell：Windows 的 cmd 默认代码页不是 UTF-8，所以解码也跟着变。
    """
    bash = shutil.which("bash")
    if bash:
        return [bash, "-c", command], "utf-8"
    if os.name == "nt":
        comspec = os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe")
        return [comspec, "/d", "/s", "/c", command], locale.getpreferredencoding(False)
    return [shutil.which("sh") or "/bin/sh", "-c", command], "utf-8"


def _kill_tree(process: subprocess.Popen[bytes]) -> None:
    """尽力杀掉 host 命令的整棵进程树。

    POSIX 用 killpg（配合 start_new_session），Windows 用 taskkill /T。主动脱离进程组
    或父子关系的进程仍可能活下来——这是尽力而为，不是保证。
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except OSError:
            pass
        if process.poll() is None:
            process.kill()
        return

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except OSError:
        pass
    if process.poll() is None:
        process.kill()


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

    agent_dir = workspace / ".agent"
    if agent_dir.exists():
        # `.agent/` 同理——但理由比 `.git` 还硬：里面装的**全都是"会在宿主机上生效"的东西**
        # ——会话流水、改前备份、长期记忆、skill。其中 skill 最要紧：它进 system prompt，
        # 是模型**会照着做的步骤**，所以一条命令往 `.agent/skill/` 里写一个，就等于给下一轮
        # （同一会话的下一轮，索引每轮重扫）塞一段指令进去。
        # 挂成只读之后容器里照样看得见（备份和流水都还能读），只是写不进去。
        # 目录不存在时不挂：docker 会因为"源路径不存在"直接失败，容器都起不来。
        mounts += ["--mount", f"type=bind,src={agent_dir.as_posix()},dst=/workspace/.agent,ro"]

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
    process: subprocess.Popen[bytes], timeout: int, on_timeout: Callable[[], None]
) -> tuple[bytes, bytes, tuple[int, int], bool]:
    """读输出、等它结束；超时就让调用方收掉进程或容器。

    第三个返回值是两条流**各自吐了多少字节**（不是留住多少）。留住的那份超长时会被落盘，
    一般用不上它；只有连落盘那份都装不下的时候，_format 才需要拿它说清丢了多少。
    """
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
        on_timeout()
        process.wait()

    for reader in readers:
        reader.join(timeout=10)

    return sinks[0]["kept"], sinks[1]["kept"], (sinks[0]["total"], sinks[1]["total"]), timed_out


def _drain(stream: Any, sink: dict[str, Any]) -> None:
    """把一条管道读到 EOF，但只留前 MAX_CAPTURED_BYTES 字节。

    超了之后**还得继续读**（读到了直接扔），不能停——停了子进程会卡在写那一头。
    """
    kept = bytearray()
    total = 0
    while True:
        chunk = stream.read1(8192)
        if not chunk:
            break
        total += len(chunk)
        if len(kept) < MAX_CAPTURED_BYTES:
            kept.extend(chunk[: MAX_CAPTURED_BYTES - len(kept)])
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
    exit_code: int,
    stdout: bytes,
    stderr: bytes,
    totals: tuple[int, int],
    timed_out: bool,
    timeout: int,
    *,
    sandbox: str = DEFAULT_SANDBOX,
    encoding: str = "utf-8",
) -> str:
    """把结果拼成给模型看的文本。

    totals 是两条流各吐了多少字节。只在超过"留得住的量"时才用得上——那是**真的丢了**：
    落盘救不回来，因为落下去的也只有留住的那部分。

    host 模式在 Windows 上没有 bash 时走 cmd.exe，输出是本地代码页，所以解码编码由调用方
    传进来；docker 和 bash 都是 UTF-8。
    """
    out_text, out_hidden = redact_sensitive(stdout.decode(encoding, "replace"))
    err_text, err_hidden = redact_sensitive(stderr.decode(encoding, "replace"))

    sections: list[str] = []
    if timed_out:
        where = "容器已经终止" if sandbox == "docker" else "命令进程已经终止"
        sections.append(f"[已超时：{timeout} 秒没跑完，{where}]")
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
    if max(totals) > MAX_CAPTURED_BYTES:
        sections.append(
            f"[输出太多：一共 {sum(totals):,} 字节，每条流只留住了前 "
            f"{MAX_CAPTURED_BYTES:,} 字节，剩下的没留住。"
            f"要看的部分在后面的话，用管道收窄范围，比如 | tail -50]"
        )

    return "\n\n".join(sections)
