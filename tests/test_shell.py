"""bash 工具的测试。

不需要 docker 的那部分（几道闸门、参数拼装、超时夹取）正常跑；真正起容器的整组在
没装 docker 的机器上会跳过。
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from mini_code_agent.tools.base import ShellDisabledError, ToolContext
from mini_code_agent.tools.shell import (
    DEFAULT_IMAGE,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
    _docker_argv,
    _timeout,
    bash,
)


def _docker_ok(args: list[str]) -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(args, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def sandbox_missing() -> str | None:
    """跑真容器的用例要两样：docker 活着，以及沙箱镜像已经构建过。返回缺的那一样。"""
    if not _docker_ok(["docker", "info"]):
        return "这台机器上没有可用的 docker"
    if not _docker_ok(["docker", "image", "inspect", DEFAULT_IMAGE]):
        return f"本地还没有沙箱镜像（先跑 docker build -t {DEFAULT_IMAGE} .）"
    return None


_missing = sandbox_missing()
needs_docker = pytest.mark.skipif(_missing is not None, reason=_missing or "")


@pytest.fixture
def shell_ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, allow_shell=True)


class TestBashGuards:
    """这些闸门都在起容器之前就把调用拦下了，所以不需要 docker。"""

    def test_shell_is_off_by_default(self, tmp_path):
        ctx = ToolContext(cwd=tmp_path)

        with pytest.raises(ShellDisabledError, match="没开命令执行能力"):
            bash({"command": "echo hi"}, ctx)

    def test_an_empty_command_is_rejected(self, shell_ctx):
        with pytest.raises(ValueError, match="非空"):
            bash({"command": "   "}, shell_ctx)

    def test_a_non_string_command_is_rejected(self, shell_ctx):
        with pytest.raises(ValueError, match="非空"):
            bash({"command": 123}, shell_ctx)


def _mounts(argv: list[str]) -> list[str]:
    """argv 里所有 --mount 的取值，按出现顺序。"""
    return [argv[index + 1] for index, item in enumerate(argv) if item == "--mount"]


class TestBashArguments:
    """拼出来的 docker 参数长什么样——也不需要 docker。"""

    def test_only_the_project_directory_is_mounted(self, tmp_path):
        """项目里没有 .git 时就这一处挂载。别的都靠 --read-only 和 --tmpfs 挡在外面。"""
        argv = _docker_argv(tmp_path, "echo hi", tmp_path / "c.cid")

        assert _mounts(argv) == [f"type=bind,src={tmp_path.as_posix()},dst=/workspace"]

    def test_the_git_directory_is_mounted_on_top_read_only(self, tmp_path):
        """.git 单独盖一层只读：项目目录整个可写，但仓库目录里能放的东西会在**宿主机**上
        生效——比如一个 pre-commit 脚本，会在你自己下次提交时被执行。"""
        (tmp_path / ".git").mkdir()

        argv = _docker_argv(tmp_path, "git status", tmp_path / "c.cid")

        assert _mounts(argv) == [
            f"type=bind,src={tmp_path.as_posix()},dst=/workspace",
            f"type=bind,src={(tmp_path / '.git').as_posix()},dst=/workspace/.git,ro",
        ]

    def test_the_user_command_is_one_single_argument(self, tmp_path):
        """整条命令作为一个 argv 元素交给容器里的 /bin/sh -c：宿主机这层没有 shell
        参与，所以没有注入可言；容器里再解释一遍是刻意的。"""
        argv = _docker_argv(tmp_path, "echo 'a b' && ls", tmp_path / "c.cid")

        assert argv[-3:] == ["/bin/sh", "-c", "echo 'a b' && ls"]

    def test_the_hardening_flags_are_all_there(self, tmp_path):
        joined = " ".join(_docker_argv(tmp_path, "true", tmp_path / "c.cid"))

        for flag in (
            "--rm",
            "--init",
            "--pull=never",
            "--read-only",
            "--network none",
            "--cap-drop ALL",
            "--security-opt no-new-privileges",
            "--user 1000:1000",
            "--pids-limit",
            "--memory",
            "--cpus",
        ):
            assert flag in joined, f"少了 {flag}"

    def test_git_is_told_the_workspace_is_safe(self, tmp_path):
        """挂进来的目录属主不是容器里的 uid，不显式加 safe.directory 的话 git 会拒绝干活
        ——而 GIT_CONFIG_GLOBAL 又被指到了 /dev/null，等于把退路也堵了。"""
        argv = _docker_argv(tmp_path, "true", tmp_path / "c.cid")

        assert "GIT_CONFIG_KEY_0=safe.directory" in argv
        assert "GIT_CONFIG_VALUE_0=/workspace" in argv

    def test_git_is_told_not_to_take_optional_locks(self, tmp_path):
        """只读的 .git 会让 git status 抢不到 index.lock（它默认想顺手刷一下 index 缓存），
        这个变量就是 git 给这种场景准备的开关：可有可无的锁别去拿。"""
        argv = _docker_argv(tmp_path, "true", tmp_path / "c.cid")

        assert "GIT_OPTIONAL_LOCKS=0" in argv


class TestTimeoutClamping:
    def test_the_default(self):
        assert _timeout(None) == DEFAULT_TIMEOUT_SECONDS

    def test_it_is_capped(self):
        assert _timeout(99999) == MAX_TIMEOUT_SECONDS

    def test_it_has_a_floor(self):
        assert _timeout(0) == 1

    def test_garbage_is_rejected(self):
        with pytest.raises(ValueError, match="得是整数"):
            _timeout("abc")


@needs_docker
class TestBashInDocker:
    def test_it_runs_a_command(self, shell_ctx):
        out = bash({"command": "echo hello"}, shell_ctx)

        assert "[退出码 0]" in out
        assert "hello" in out

    def test_a_nonzero_exit_code_is_a_result_not_an_exception(self, shell_ctx):
        """命令失败是**结果**——交回模型让它自己改。工具出错才是异常。"""
        out = bash({"command": "exit 3"}, shell_ctx)

        assert "[退出码 3]" in out

    def test_the_project_directory_is_mounted(self, shell_ctx, tmp_path):
        (tmp_path / "marker.txt").write_bytes(b"from the host\n")

        out = bash({"command": "cat /workspace/marker.txt"}, shell_ctx)

        assert "from the host" in out

    def test_it_can_write_the_workspace_but_not_the_root(self, shell_ctx, tmp_path):
        out = bash(
            {"command": "touch /workspace/new.txt && echo wrote; touch /etc/nope || echo blocked"},
            shell_ctx,
        )

        assert "wrote" in out
        assert "blocked" in out
        assert (tmp_path / "new.txt").exists()

    def test_output_is_redacted(self, shell_ctx):
        out = bash({"command": "echo 'DEEPSEEK_API_KEY=sk-live-abcdef'"}, shell_ctx)

        assert "sk-live-abcdef" not in out
        assert "已涂黑" in out

    def test_a_timeout_actually_removes_the_container(self, shell_ctx, monkeypatch):
        """杀掉 docker 客户端并不会让容器停下来（实测过），得照着 cidfile 显式 rm -f。
        把 uuid 打桩成固定值，就为了能按名字去找那个容器。"""
        monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=0))
        name = "mca-0000000000"

        out = bash({"command": "sleep 30", "timeout_seconds": 2}, shell_ctx)

        assert "已超时" in out
        still_running = subprocess.run(
            ["docker", "ps", "--filter", f"name={name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        assert still_running == "", f"超时之后容器还在跑：{still_running}"

    def test_long_output_is_capped(self, shell_ctx):
        out = bash({"command": "python -c \"print('x' * 20000)\""}, shell_ctx)

        assert "输出被截断" in out
        assert len(out) < 20000
