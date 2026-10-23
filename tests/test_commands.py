"""斜杠命令：认出来就地办掉、改得动档位、认不出来也不发给模型。全部本地跑，不碰网络。"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from mini_code_agent.commands import (
    ChatState,
    Command,
    CommandContext,
    Commands,
    default_commands,
    render_session_list,
)
from mini_code_agent.commands.base import split_command
from mini_code_agent.commands.builtin import make_help
from mini_code_agent.console import console
from mini_code_agent.context_window import window_for
from mini_code_agent.core.agent import CANCELLED_ERROR, AgentResult
from mini_code_agent.core.tasks import TaskManager
from mini_code_agent.interaction import PermissionBroker
from mini_code_agent.permissions import Grants, PermissionRequest
from mini_code_agent.results import DEFAULT_RESULTS
from mini_code_agent.sandbox import DEFAULT_SANDBOX
from mini_code_agent.sessions import SESSIONS_DIR, Session, start
from mini_code_agent.summarizing import COMPACT_AT


class StubProvider:
    """命令测试只需要 `model`；请求方法不走，所以不实现 complete / stream。"""

    def __init__(self, model: str | None = None) -> None:
        self.model = model


def make_ctx(
    mode: str = "default",
    *,
    cwd: Path | None = None,
    allow_shell: bool = False,
    session: Session | None = None,
    grants: Grants | None = None,
    messages: list[dict] | None = None,
    compact: Callable[[], str | None] | None = None,
    model: str | None = None,
    sandbox: str = DEFAULT_SANDBOX,
    results: str = DEFAULT_RESULTS,
    context_tokens: int | None = None,
    context_covered: int = 0,
    tasks: TaskManager | None = None,
    broker: PermissionBroker | None = None,
) -> CommandContext:
    """一份手边的东西。全给上默认值，测哪一块就只改哪一块。

    `compact` 默认是个"什么都不做、什么都不说"的桩——真那个要连 provider，在 cli 里。
    `model` 默认是 None，窗口按"未知模型"那一档（见 context_window），所以是 40 万而不是
    0——印出"窗口 0" 那种数谁看都不对。
    """
    state = ChatState(
        permission_mode=mode,
        messages=list(messages or []),
        provider=StubProvider(model),
        window=window_for(model),
        sandbox=sandbox,
        results=results,
    )
    if context_tokens is not None:
        state.remember_context(context_tokens, context_covered)
    return CommandContext(
        console=console,
        state=state,
        cwd=cwd or Path("proj"),
        allow_shell=allow_shell,
        # opened=True：这就是"盘上已经有一次会话"的样子。没落盘的那种由 /status 那条单独的
        # 用例去测（它说的是另一句话）。
        session=session
        or Session(id="s1", path=Path("proj/.agent/sessions/s1.jsonl"), opened=True),
        grants=grants if grants is not None else Grants(),
        compact=compact if compact is not None else (lambda: None),
        tasks=tasks,
        broker=broker,
    )


def approved(*paths: str, tool: str = "write_file") -> Grants:
    """一份"本会话批过这几个文件"的 Grants，走的是真的 remember（所以键是归一化过的）。"""
    grants = Grants()
    for path in paths:
        grants.remember(PermissionRequest(tool_name=tool, tool_args={"path": path}))
    return grants


def a_command(name: str, **kwargs) -> Command:
    """一条不做任何事的命令，只用来占名字。"""
    return Command(name=name, run=lambda ctx, arg: "handled", **kwargs)


class TestChatState:
    """`ChatState` 上那两个"换历史"的东西——压缩和折叠都要走它，所以它自己得是对的。"""

    def test_replacing_swaps_the_history_and_drops_the_anchor(self):
        state = ChatState(
            permission_mode="default",
            messages=[{"role": "user", "content": "a"}],
            provider=StubProvider(),
            window=window_for(None),
        )
        state.remember_context(120_000, 1)
        fresh = [{"role": "user", "content": "b"}]

        assert state.replaced(fresh) is True

        assert state.messages is fresh
        assert state.context_tokens is None  # 锚点跟着作废了
        assert state.context_covered == 0

    def test_handing_the_same_one_back_is_not_a_replacement(self):
        """压缩和折叠都是"没改动就原样返回传进来那份"，所以**对象身份**就是"换没换"的信号。

        没换就不动能锚点——不然每轮都会白白退回粗估。
        """
        state = ChatState(
            permission_mode="default",
            messages=[{"role": "user", "content": "a"}],
            provider=StubProvider(),
            window=window_for(None),
        )
        state.remember_context(120_000, 1)

        assert state.replaced(state.messages) is False

        assert state.context_tokens == 120_000


class TestDispatch:
    def test_a_plain_line_is_left_to_the_caller(self, capsys):
        """不是命令就返回 not_a_command，让调用方照常发给模型。"""
        assert default_commands().dispatch("帮我看看这个文件", make_ctx()) == "not_a_command"
        assert capsys.readouterr().out == ""

    def test_a_line_starting_with_a_slash_is_never_forwarded(self, capsys):
        """以 `/` 开头就是这个命名空间：打错了只该得到一句"没有这个命令"。"""
        assert default_commands().dispatch("/hlep", make_ctx()) == "handled"
        out = capsys.readouterr().out
        assert "/hlep" in out
        assert "没有发给模型" in out  # 用户得知道那句话没递出去

    def test_a_bare_slash_is_an_unknown_command(self, capsys):
        assert default_commands().dispatch("/", make_ctx()) == "handled"
        assert "这个命令" in capsys.readouterr().out

    def test_the_name_is_case_insensitive(self, capsys):
        ctx = make_ctx()
        assert default_commands().dispatch("/PERMISSION_MODE plan", ctx) == "handled"
        assert ctx.state.permission_mode == "plan"


class TestHelp:
    def test_it_lists_every_command(self, capsys):
        default_commands().dispatch("/help", make_ctx())

        out = capsys.readouterr().out
        wanted = (
            "/help",
            "/permission_mode",
            "/sandbox",
            "/model",
            "/status",
            "/sessions",
            "/tasks",
            "/compact",
            "/grants",
            "/exit",
        )
        for name in wanted:
            assert name in out
        assert "/quit" in out  # 别名也列出来，不然没人知道它存在

    def test_it_lists_commands_registered_after_itself(self, capsys):
        """`/help` 抓的是那张表本身、不是快照——它是第一条注册的，别的一条都还没进来。"""
        commands = Commands()
        commands.register(make_help(commands))
        commands.register(a_command("later", summary="后加的"))

        commands.dispatch("/help", make_ctx())

        assert "later" in capsys.readouterr().out


class TestPermissionMode:
    def test_no_argument_reports_the_current_one(self, capsys):
        default_commands().dispatch("/permission_mode", make_ctx("plan"))

        out = capsys.readouterr().out
        assert "plan" in out
        assert "只读" in out  # 光有个名字不算说清楚，档位是"会问/不会问"的意思

    def test_it_switches_the_mode(self, capsys):
        ctx = make_ctx()
        assert default_commands().dispatch("/permission_mode plan", ctx) == "handled"

        assert ctx.state.permission_mode == "plan"
        assert "plan" in capsys.readouterr().out

    def test_the_short_form_works(self, capsys):
        """`accept` 是 `accept edits` 的短写法，跟命令行上一样。"""
        ctx = make_ctx()
        default_commands().dispatch("/permission_mode accept", ctx)

        assert ctx.state.permission_mode == "accept edits"

    def test_a_mode_that_makes_no_sense_changes_nothing(self, capsys):
        """看不懂的词**不兜底**：兜底的后果是用户以为切过去了，实际还在原来那档上跑。"""
        ctx = make_ctx("plan")
        assert default_commands().dispatch("/permission_mode 随便写点什么", ctx) == "handled"

        assert ctx.state.permission_mode == "plan"
        assert "不是权限模式" in capsys.readouterr().out

    def test_bypass_can_be_switched_on_from_within_with_a_warning(self, capsys):
        """用户自己点名切 bypass 是允许的，但会当场打一行红色警告。"""
        ctx = make_ctx()
        assert default_commands().dispatch("/permission_mode bypass", ctx) == "handled"

        assert ctx.state.permission_mode == "bypass permissions"
        out = capsys.readouterr().out
        assert "⚠" in out
        assert "什么都不问" in out

    def test_switching_to_the_mode_it_is_already_on_says_so(self, capsys):
        ctx = make_ctx("plan")
        default_commands().dispatch("/permission_mode plan", ctx)

        assert "已经是" in capsys.readouterr().out


class TestFork:
    def test_no_argument_reports_the_current_one(self, capsys):
        default_commands().dispatch("/fork", make_ctx())

        out = capsys.readouterr().out
        assert "关" in out
        assert "/fork on" in out

    def test_it_turns_fork_on_and_says_the_cache_will_be_rebuilt(self, capsys):
        """切它会换提示词里那段说明，也就是重建整个前缀——代价要当场说清楚。"""
        ctx = make_ctx()

        assert default_commands().dispatch("/fork on", ctx) == "handled"

        assert ctx.state.allow_fork is True
        out = capsys.readouterr().out
        assert "开" in out
        assert "缓存" in out

    def test_it_turns_fork_off(self, capsys):
        ctx = make_ctx()
        ctx.state.allow_fork = True

        default_commands().dispatch("/fork off", ctx)

        assert ctx.state.allow_fork is False
        assert "关" in capsys.readouterr().out

    def test_a_word_that_makes_no_sense_changes_nothing(self, capsys):
        """看不懂就**不动**：静默取反比不切更糟，用户会以为切过去了。"""
        ctx = make_ctx()

        assert default_commands().dispatch("/fork 随便", ctx) == "handled"

        assert ctx.state.allow_fork is False
        assert "看不懂" in capsys.readouterr().out

    def test_switching_to_the_state_it_is_already_in_says_so(self, capsys):
        ctx = make_ctx()

        default_commands().dispatch("/fork off", ctx)

        assert "已经是" in capsys.readouterr().out

    def test_it_is_listed_in_help(self, capsys):
        default_commands().dispatch("/help", make_ctx())

        assert "/fork" in capsys.readouterr().out


class TestModel:
    def test_no_argument_reports_the_current_one(self, capsys):
        default_commands().dispatch("/model", make_ctx(model="deepseek-v4-flash"))

        out = capsys.readouterr().out
        assert "模型：deepseek-v4-flash" in out
        assert f"窗口 {window_for('deepseek-v4-flash'):,}" in out
        assert "/model <模型id>" in out

    def test_switching_changes_provider_and_window(self, capsys):
        ctx = make_ctx(model="deepseek-v4-flash")

        assert default_commands().dispatch("/model qwen-max", ctx) == "handled"

        assert ctx.state.provider.model == "qwen-max"
        assert ctx.state.window == window_for("qwen-max")
        out = capsys.readouterr().out
        assert "模型换成 qwen-max" in out
        assert "下一轮" in out
        assert "缓存" in out
        assert f"窗口 {window_for('qwen-max'):,}" in out

    def test_switching_to_the_same_model_says_so(self, capsys):
        ctx = make_ctx(model="deepseek-v4-flash")

        default_commands().dispatch("/model deepseek-v4-flash", ctx)

        assert ctx.state.provider.model == "deepseek-v4-flash"
        assert "已经是" in capsys.readouterr().out

    def test_an_unknown_model_is_accepted_with_the_default_window(self, capsys):
        ctx = make_ctx(model="deepseek-v4-flash")

        assert default_commands().dispatch("/model made-up", ctx) == "handled"

        assert ctx.state.provider.model == "made-up"
        assert ctx.state.window == window_for("made-up")
        out = capsys.readouterr().out
        assert "窗口表里没有" in out
        assert f"{window_for('made-up'):,}" in out

    def test_a_blank_argument_just_reports(self, capsys):
        ctx = make_ctx(model="deepseek-v4-flash")

        default_commands().dispatch("/model   ", ctx)

        assert ctx.state.provider.model == "deepseek-v4-flash"
        assert "模型：deepseek-v4-flash" in capsys.readouterr().out

    def test_it_warns_about_running_background_tasks(self, capsys):
        class BusyTasks:
            def running(self) -> int:
                return 2

        ctx = make_ctx(model="deepseek-v4-flash", tasks=BusyTasks())

        default_commands().dispatch("/model qwen-max", ctx)

        assert "后台任务" in capsys.readouterr().out

    def test_it_is_listed_in_help(self, capsys):
        default_commands().dispatch("/help", make_ctx())

        assert "/model" in capsys.readouterr().out


class TestSandbox:
    def test_no_argument_reports_the_current_one(self, capsys):
        default_commands().dispatch("/sandbox", make_ctx())

        out = capsys.readouterr().out
        assert "当前的沙箱：docker" in out
        assert "/sandbox docker|host" in out

    def test_it_switches_to_host_with_a_warning(self, capsys):
        ctx = make_ctx()

        assert default_commands().dispatch("/sandbox host", ctx) == "handled"

        assert ctx.state.sandbox == "host"
        out = capsys.readouterr().out
        assert "⚠" in out
        assert "宿主机" in out
        assert "缓存" in out

    def test_it_switches_back_to_docker(self, capsys):
        ctx = make_ctx(sandbox="host")

        assert default_commands().dispatch("/sandbox docker", ctx) == "handled"

        assert ctx.state.sandbox == "docker"
        assert "docker" in capsys.readouterr().out

    def test_the_same_mode_says_so(self, capsys):
        ctx = make_ctx()

        default_commands().dispatch("/sandbox docker", ctx)

        assert ctx.state.sandbox == "docker"
        assert "已经是" in capsys.readouterr().out

    def test_a_bad_mode_changes_nothing(self, capsys):
        ctx = make_ctx(sandbox="host")

        default_commands().dispatch("/sandbox podman", ctx)

        assert ctx.state.sandbox == "host"
        assert "不是沙箱模式" in capsys.readouterr().out

    def test_it_says_the_setting_has_no_effect_without_shell(self, capsys):
        default_commands().dispatch("/sandbox", make_ctx(allow_shell=False))

        assert "暂时不生效" in capsys.readouterr().out

    def test_it_warns_about_running_background_tasks(self, capsys):
        class BusyTasks:
            def running(self) -> int:
                return 1

        ctx = make_ctx(tasks=BusyTasks())

        default_commands().dispatch("/sandbox host", ctx)

        assert "后台任务" in capsys.readouterr().out


class TestStatus:
    def test_it_reports_the_sandbox(self, capsys):
        default_commands().dispatch("/status", make_ctx(sandbox="host"))

        out = capsys.readouterr().out
        assert "沙箱：host" in out
        assert "宿主机" in out

    def test_it_reports_the_tool_result_policy(self, capsys):
        """结果被藏起来时用户得能确认这个设置——否则"结果没打印"看着像工具没跑。"""
        default_commands().dispatch("/status", make_ctx(results="hidden"))

        out = capsys.readouterr().out
        assert "工具结果：hidden" in out
        assert "出错的结果仍会完整打印" in out

    def test_it_reports_the_current_model(self, capsys):
        default_commands().dispatch("/status", make_ctx(model="qwen-max"))

        out = capsys.readouterr().out
        assert "模型：qwen-max" in out
        assert f"窗口 {window_for('qwen-max'):,}" in out

    def test_it_reports_where_it_runs_and_under_what(self, capsys, tmp_path):
        default_commands().dispatch("/status", make_ctx("plan", cwd=tmp_path))

        out = capsys.readouterr().out
        assert f"工作目录：{tmp_path}" in out
        assert "权限档位：plan" in out
        assert "只读" in out  # 档位附了它意味着什么

    @pytest.mark.parametrize("allow_shell, expected", [(True, "开"), (False, "关")])
    def test_it_says_whether_commands_can_run(self, capsys, allow_shell, expected):
        default_commands().dispatch("/status", make_ctx(allow_shell=allow_shell))

        assert f"命令执行：{expected}" in capsys.readouterr().out

    @pytest.mark.parametrize("allow_fork, expected", [(True, "开"), (False, "关")])
    def test_it_says_whether_fork_is_on(self, capsys, allow_fork, expected):
        ctx = make_ctx()
        ctx.state.allow_fork = allow_fork

        default_commands().dispatch("/status", ctx)

        assert f"fork：{expected}" in capsys.readouterr().out

    def test_it_reports_the_session_and_how_much_history_there_is(self, capsys):
        session = Session(
            id="abcd", path=Path("proj/.agent/sessions/abcd.jsonl"), opened=True, written=7
        )
        ctx = make_ctx(session=session, messages=[{"role": "user", "content": "x" * 100}])

        default_commands().dispatch("/status", ctx)

        out = capsys.readouterr().out
        assert "会话文件：.agent" in out  # 相对 cwd 印，不占半屏
        assert "abcd.jsonl" in out
        assert "落盘 7 条消息" in out
        assert "历史大小：1 条" in out
        # 没有真用过接口（没锚点），所以只能说"估算"，再报出窗口和阈值
        assert "token（估算）" in out
        assert f"窗口 {window_for(None):,}" in out
        assert f"到 {int(window_for(None) * COMPACT_AT):,} 才压" in out

    def test_it_says_when_the_number_is_real_not_estimated(self, capsys):
        """锚点是上一发真请求报的——那就不是估的，这个字得改，不然这个数没法信。"""
        ctx = make_ctx(
            messages=[{"role": "user", "content": "x"}],
            context_tokens=120_000,
            context_covered=1,
        )

        default_commands().dispatch("/status", ctx)

        assert "token（真数）" in capsys.readouterr().out

    def test_it_says_the_session_has_not_landed_yet(self, capsys):
        """会话是懒建的：还没说过话，盘上就没有那个文件——不该印一个不存在的路径配"落盘 0 条"。"""
        not_yet = Session(id="s1", path=Path("proj/.agent/sessions/s1.jsonl"))

        default_commands().dispatch("/status", make_ctx(session=not_yet))

        out = capsys.readouterr().out
        assert "还没落盘" in out
        assert "s1.jsonl" not in out

    def test_it_says_when_the_history_has_been_compacted(self, capsys):
        """压过没有，光看对话是看不出来的——但它决定了开头那些事模型还记不记得住。"""
        session = Session(id="s1", path=Path("s1.jsonl"), has_summary=True)

        default_commands().dispatch("/status", make_ctx(session=session))

        assert "已经压成摘要了" in capsys.readouterr().out

    def test_it_says_nothing_about_summaries_when_there_were_none(self, capsys):
        default_commands().dispatch("/status", make_ctx())

        assert "摘要" not in capsys.readouterr().out

    def test_it_reports_background_tasks(self, capsys, tmp_path):
        manager, _ = a_finished_task(tmp_path)

        default_commands().dispatch("/status", make_ctx(cwd=tmp_path, tasks=manager))

        assert "后台任务：0 个在跑，1 个已结束" in capsys.readouterr().out

    def test_it_stays_quiet_when_there_are_none(self, capsys):
        default_commands().dispatch("/status", make_ctx(tasks=TaskManager()))

        assert "后台任务：" not in capsys.readouterr().out

    def test_it_reports_pending_approvals(self, capsys):
        broker = PermissionBroker()
        broker.defer(
            PermissionRequest(tool_name="write_file", tool_args={"path": "a.py"}),
            "[general-purpose] ",
        )

        default_commands().dispatch("/status", make_ctx(broker=broker))

        assert "后台授权：1 个在等" in capsys.readouterr().out


def a_finished_task(tmp_path, final: str = "找到三处"):
    manager = TaskManager()
    task_id = manager.start(
        lambda emit, cancel: AgentResult(final=final, trace=[], messages=[]),
        label="[explore] ",
        agent_type="explore",
        description="找压缩逻辑",
        cwd=tmp_path,
        spill_dir=None,
    )
    assert manager.wait(task_id, timeout=2)
    return manager, task_id


class TestTasks:
    """/tasks：看得见、停得下。后台任务是花了钱、在用户看不见的地方跑着的。"""

    def test_it_says_so_when_there_is_no_table(self, capsys):
        default_commands().dispatch("/tasks", make_ctx())

        assert "没有后台任务表" in capsys.readouterr().out

    def test_it_says_so_when_nothing_has_run(self, capsys):
        default_commands().dispatch("/tasks", make_ctx(tasks=TaskManager()))

        assert "还没有后台任务" in capsys.readouterr().out

    def test_it_lists_a_finished_task(self, capsys, tmp_path):
        manager, task_id = a_finished_task(tmp_path)

        default_commands().dispatch("/tasks", make_ctx(cwd=tmp_path, tasks=manager))

        out = capsys.readouterr().out
        assert task_id in out
        assert "完成" in out
        assert "找压缩逻辑" in out

    def test_show_prints_the_conclusion(self, capsys, tmp_path):
        manager, task_id = a_finished_task(tmp_path, final="找到三处")

        default_commands().dispatch(
            f"/tasks show {task_id}", make_ctx(cwd=tmp_path, tasks=manager)
        )

        assert "找到三处" in capsys.readouterr().out

    def test_cancel_asks_a_running_task_to_stop(self, capsys, tmp_path):
        began = threading.Event()

        def work(emit, cancel):
            began.set()
            cancel.wait(2)
            return AgentResult(
                final=None, trace=[], messages=[], error=CANCELLED_ERROR
            )

        manager = TaskManager()
        task_id = manager.start(
            work,
            label="[explore] ",
            agent_type="explore",
            description="找压缩逻辑",
            cwd=tmp_path,
            spill_dir=None,
        )
        assert began.wait(2)

        default_commands().dispatch(
            f"/tasks cancel {task_id}", make_ctx(cwd=tmp_path, tasks=manager)
        )

        assert "会在下一次发请求之前停下" in capsys.readouterr().out
        assert manager.wait(task_id, timeout=2)

    def test_cancelling_a_finished_task_says_so(self, capsys, tmp_path):
        manager, task_id = a_finished_task(tmp_path)

        default_commands().dispatch(
            f"/tasks cancel {task_id}", make_ctx(cwd=tmp_path, tasks=manager)
        )

        assert "已经结束了" in capsys.readouterr().out

    def test_an_unknown_id_is_refused(self, capsys, tmp_path):
        manager, _ = a_finished_task(tmp_path)

        default_commands().dispatch(
            "/tasks show bg-nope", make_ctx(cwd=tmp_path, tasks=manager)
        )

        assert "没有 'bg-nope' 这个后台任务" in capsys.readouterr().out

    def test_an_unknown_action_is_refused(self, capsys, tmp_path):
        manager, task_id = a_finished_task(tmp_path)

        default_commands().dispatch(
            f"/tasks kill {task_id}", make_ctx(cwd=tmp_path, tasks=manager)
        )

        assert "看不懂" in capsys.readouterr().out

    def test_it_reports_pending_approvals(self, capsys, tmp_path):
        manager, _ = a_finished_task(tmp_path)
        broker = PermissionBroker()
        broker.defer(
            PermissionRequest(tool_name="write_file", tool_args={"path": "a.py"}),
            "[general-purpose] ",
        )

        default_commands().dispatch(
            "/tasks", make_ctx(cwd=tmp_path, tasks=manager, broker=broker)
        )

        assert "授权请求在等" in capsys.readouterr().out


class TestCompact:
    def test_it_runs_the_compaction(self):
        calls: list[bool] = []
        ctx = make_ctx(compact=lambda: calls.append(True) or None)

        default_commands().dispatch("/compact", ctx)

        assert calls == [True]

    def test_it_stays_quiet_when_the_compaction_has_nothing_to_add(self, capsys):
        """压成功那几行由 _compact 自己打（它知道切在第几条），命令不重复说一遍。"""
        default_commands().dispatch("/compact", make_ctx(compact=lambda: None))

        assert capsys.readouterr().out == ""

    def test_it_relays_what_the_compaction_had_to_say(self, capsys):
        default_commands().dispatch("/compact", make_ctx(compact=lambda: "没什么可压的"))

        assert "没什么可压的" in capsys.readouterr().out

    def test_an_argument_is_refused(self, capsys):
        calls: list[bool] = []
        ctx = make_ctx(compact=lambda: calls.append(True) or None)

        default_commands().dispatch("/compact 随便写点什么", ctx)

        assert calls == []  # 参数不对就别压——压一次是一次要花钱的模型调用
        assert "不接受参数" in capsys.readouterr().out


class TestGrants:
    def test_it_lists_what_has_been_approved(self, capsys):
        default_commands().dispatch("/grants", make_ctx(grants=approved("src/a.py", "src/b.py")))

        out = capsys.readouterr().out
        assert "write_file" in out
        # 键里的路径是归一化过的（Windows 上是反斜杠），照着那个形式断言
        assert os.path.normpath("src/a.py") in out
        assert os.path.normpath("src/b.py") in out

    def test_it_says_so_when_nothing_has_been_approved(self, capsys):
        default_commands().dispatch("/grants", make_ctx())

        assert "还没有批准过" in capsys.readouterr().out

    def test_clear_empties_it(self, capsys):
        """清的是循环里那个 Grants 本身，不是它的一份副本——所以下一轮真的会重新问。"""
        grants = approved("src/a.py")

        default_commands().dispatch("/grants clear", make_ctx(grants=grants))

        assert grants.entries == set()
        assert "清掉了 1 条" in capsys.readouterr().out

    def test_clear_on_an_empty_one_says_so(self, capsys):
        default_commands().dispatch("/grants clear", make_ctx())

        assert "本来就是空的" in capsys.readouterr().out

    def test_an_argument_other_than_clear_is_refused(self, capsys):
        default_commands().dispatch("/grants nope", make_ctx())

        assert "只认 clear" in capsys.readouterr().out


class TestExit:
    @pytest.mark.parametrize("line", ["/exit", "/quit"])
    def test_it_stops_the_loop(self, line):
        assert default_commands().dispatch(line, make_ctx()) == "stop"


class TestParsing:
    def test_the_argument_keeps_its_spaces(self):
        """`accept edits` 中间那个空格是名字的一部分，拆细了还得拼回去。"""
        assert split_command("/permission_mode accept edits") == (
            "permission_mode",
            "accept edits",
        )

    @pytest.mark.parametrize(
        "line",
        ["/help", "/help   ", "/  help", "/\thelp\t"],
    )
    def test_the_name_stops_at_the_first_blank(self, line):
        assert split_command(line) == ("help", "")


class TestRegistration:
    def test_a_duplicate_name_is_rejected(self):
        commands = Commands([a_command("x")])
        with pytest.raises(ValueError, match="已经注册过"):
            commands.register(a_command("x"))

    def test_an_alias_colliding_with_another_name_is_rejected(self):
        """别名和正名同权，所以撞了也一样要拦——不然查出来的是哪条全看注册顺序。"""
        commands = Commands([a_command("a", aliases=("b",))])
        with pytest.raises(ValueError, match="已经注册过"):
            commands.register(a_command("b"))

    def test_an_alias_is_not_listed_as_its_own_command(self):
        commands = Commands([a_command("a", aliases=("b",))])

        assert [command.name for command in commands.list()] == ["a"]

    def test_a_name_with_a_blank_in_it_is_rejected(self):
        """名字里带空白就永远查不到——分派时按空白切名字。"""
        with pytest.raises(ValueError, match="只能是字母"):
            Command(name="permission mode", run=lambda ctx, arg: "handled")

    def test_every_default_command_can_be_reached_by_its_aliases(self):
        commands = default_commands()
        for command in commands.list():
            for alias in command.aliases:
                assert commands.by_name[alias] is command


def a_spoken_session(tmp_path, said: str = "给我看看那个文件"):
    session = start(tmp_path, permission_mode="default", allow_shell=False)
    session.append([{"role": "user", "content": said}])
    return session


class TestSessionList:
    """会话列表的渲染。启动参数 --list 和以后的 /sessions 共用这一份。"""

    def test_it_lists_what_is_there(self, capsys, tmp_path):
        session = a_spoken_session(tmp_path)

        render_session_list(console, tmp_path)

        out = capsys.readouterr().out
        assert session.id in out
        assert "1 条" in out
        assert "给我看看那个文件" in out
        assert "--resume" in out  # 列出来就是为了让你知道该接哪个

    def test_it_says_so_when_there_has_never_been_one(self, capsys, tmp_path):
        render_session_list(console, tmp_path)

        assert "还没有过会话" in capsys.readouterr().out

    def test_the_ones_that_never_said_anything_are_counted_not_listed(self, capsys, tmp_path):
        """空壳是每次启动的副产品，逐行列出来会把列表淹掉——报个数就够了。"""
        spoken = a_spoken_session(tmp_path)
        (tmp_path / SESSIONS_DIR / "29991231-235959.jsonl").write_text(
            '{"type": "meta", "cwd": "."}', encoding="utf-8"
        )

        render_session_list(console, tmp_path)

        out = capsys.readouterr().out
        assert spoken.id in out
        assert "29991231-235959" not in out
        assert "另有 1 次没说过话的" in out

    def test_a_compacted_session_is_marked(self, capsys, tmp_path):
        session = a_spoken_session(tmp_path)
        session.record_summary(1, "压过了")

        render_session_list(console, tmp_path)

        assert "压过" in capsys.readouterr().out

    def test_a_session_that_was_never_compacted_is_not_marked(self, capsys, tmp_path):
        a_spoken_session(tmp_path)

        render_session_list(console, tmp_path)

        assert "压过" not in capsys.readouterr().out


class TestSessionsCommand:
    """/sessions：跟启动参数的 --list 是同一份渲染，多一个"我在这儿"的星号。"""

    def test_it_marks_the_session_you_are_in(self, capsys, tmp_path):
        mine = a_spoken_session(tmp_path, "我这次说的")
        theirs = a_spoken_session(tmp_path, "另一次的")

        default_commands().dispatch("/sessions", make_ctx(cwd=tmp_path, session=mine))

        out = capsys.readouterr().out
        assert f"* {mine.id}" in out
        assert f"* {theirs.id}" not in out  # 星号只打在这一次上

    def test_it_says_so_when_this_session_has_not_landed(self, capsys, tmp_path):
        """还没说过话的这次盘上没有、列不出来——那就说明一句，别让人在单子上找自己。"""
        not_yet = Session(id="s1", path=tmp_path / "s1.jsonl")  # opened 默认 False

        default_commands().dispatch("/sessions", make_ctx(cwd=tmp_path, session=not_yet))

        assert "还没落盘" in capsys.readouterr().out

    def test_an_argument_is_refused(self, capsys, tmp_path):
        default_commands().dispatch("/sessions 随便写点什么", make_ctx(cwd=tmp_path))

        assert "不接受参数" in capsys.readouterr().out
