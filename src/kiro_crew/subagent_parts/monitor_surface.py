"""The manager's orphan and stall watch: the reaper loop, process probes, orphan
notices, tool-call attribution for the stall detector, slow-command notes, live cost
sampling, the dependency coordinator, the task-queue pump, and the delegations into
:class:`~kiro_crew.subagent_manager.OrphanStallMonitor`.

Each function takes the manager as ``self`` and is bound as the ``SubagentManager``
attribute of the same name in :mod:`kiro_crew.subagent_parts.manager`.

Manager fields read: ``_admission``, ``_monitor``.

Manager fields written: none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os
    import time

    from kiro_crew.subagent import (
        TERMINAL_TOOL_STATUSES,
        Any,
        SubagentDelivery,
        SubagentInfo,
        SubagentManager,
        ToolCallState,
        _redact,
        boottime_now,
        failure_name,
        logger,
        platform_compat,
        record_slow_command,
    )


def start_reaper(self: SubagentManager) -> None:
    return self._monitor.start_reaper_impl()


async def _reconcile_orphans(self: SubagentManager) -> None:
    return await self._monitor._reconcile_orphans_impl()


async def _end_leaked_processes(self: SubagentManager) -> None:
    return await self._monitor._end_leaked_processes_impl()


async def _end_orphan_process(
    self: SubagentManager, agent_id: str, state: dict, *, reason: str
) -> str | None:
    return await self._monitor._end_orphan_process_impl(agent_id, state, reason=reason)


def _is_pid_alive(pid: int) -> bool:
    """Check if a PID is still running."""
    # os.kill(pid, 0) would terminate the process on Windows — probe instead.
    return platform_compat.pid_exists(pid)


def _is_orphan_process(pid: int, spawned_at: float) -> bool:
    """Check if PID belongs to the original subagent (not a recycled PID).

    Compares /proc/{pid} creation time against the recorded spawn time.
    Returns False if the process was created after the agent was spawned
    (indicating PID reuse).
    """
    try:
        proc_stat = os.stat(f"/proc/{pid}")
        return proc_stat.st_ctime <= spawned_at + 2.0
    except (FileNotFoundError, OSError):
        return False


async def _kill_orphan_pid(pid: int) -> str | None:
    """Best-effort SIGKILL of an orphaned process; returns what stopped it.

    ``None`` once the process is gone (killed here or already exited), else the
    failure the kill raised (``PermissionError: …``), so the caller's audit row
    can say ``failed`` rather than ``killed``. Never raises: the caller must still
    finish its reconciliation.

    Uses :func:`platform_compat.kill_pid_async` because the Windows ``taskkill``
    waits up to five seconds, and this runs on the event loop.
    """
    try:
        await platform_compat.kill_pid_async(pid, platform_compat.SIGKILL)
    except ProcessLookupError:
        return None
    except OSError as exc:
        return failure_name(exc)
    return None


async def _notify_orphan(
    self: SubagentManager, agent_id: str, state: dict, has_result: bool
) -> str | None:
    return await self._monitor._notify_orphan_impl(agent_id, state, has_result)


async def _try_inject_orphan_notification(
    self: SubagentManager,
    parent_session: str,
    msg: str,
    meta: dict | None = None,
    delivery: SubagentDelivery | None = None,
) -> bool | str:
    return await self._monitor._try_inject_orphan_notification_impl(
        parent_session, msg, meta, delivery
    )


async def _send_orphan_slack_dm(self: SubagentManager, msg: str) -> None:
    return await self._monitor._send_orphan_slack_dm_impl(msg)


def _live_shared_count(self: SubagentManager, pid: int | None, agents: "list[SubagentInfo]") -> int:
    return self._monitor._live_shared_count_impl(pid, agents)


def _sample_live_costs(self: SubagentManager) -> None:
    return self._monitor._sample_live_costs_impl()


def _refresh_learned_settled(self: SubagentManager) -> None:
    return self._monitor._refresh_learned_settled_impl()


def _record_cost(self: SubagentManager, info: SubagentInfo) -> None:
    return self._monitor._record_cost_impl(info)


async def _reaper_loop(self: SubagentManager) -> None:
    return await self._monitor._reaper_loop_impl()


def _is_startup_stalled(
    self: SubagentManager, info: SubagentInfo, now: float, mono_now: float | None = None
) -> bool:
    return self._monitor._is_startup_stalled_impl(info, now, mono_now)


def _note_tool_dispatch(info: SubagentInfo, event: Any) -> None:
    """Record the in-flight tool for liveness attribution.

    Mirrors ``AcpSessionHandle``'s ``_inflight_tool`` snapshot: title, the
    already-redacted input, the dispatch instant, and the TRUSTED
    ``is_shell`` / ``tool_name`` fields from ``_meta.kiro`` (never the
    LLM-authored title). The subagent event loop already receives the same
    ``AcpEvent``, and keeping only ``title`` would leave stall detection
    with nothing to attribute evidence with.

    The hand-off runs through the shared :class:`InFlightToolTracker`
    (``dispatch``), the same rule the main agent uses.

    The oracle is retired, not cleared: a movement walk still running against
    the PREVIOUS command holds the old instance, and clearing in place would let
    its late write land on the new tool's baseline and read as movement.
    """
    state = ToolCallState(
        title=event.title or "",
        command=event.tool_input or "",
        dispatch_ts=time.monotonic(),
        dispatch_boot_ts=boottime_now(),
        # No consumer parking on this path: a subagent's events are consumed
        # by the run loop itself, with no approval / IM send / hook holding a
        # frame, so this stamp cannot lag the runtime's spawn the way the
        # dashboard dispatch loop's can.
        dispatch_parked_secs=0.0,
        is_shell=bool(getattr(event, "is_shell", False)),
        tool_name=getattr(event, "tool_name", "") or "",
        # Only a provenance-verified identity names the server, so a frame
        # without one cannot select the trusted wait contract.
        mcp_server_name=(
            getattr(event, "mcp_server_name", "") or ""
            if getattr(event, "mcp_identity_trusted", False) is True
            else ""
        ),
    )
    # A native child's call (``sub_session_id`` set) is judged but takes the
    # empty-id slot: its result goes to the backend's own session and never
    # reaches this loop, so a stored entry would never be popped. A normal call is
    # stored under its toolCallId so a parallel call finishing first hands the
    # judged slot back to one still running.
    if getattr(event, "sub_session_id", ""):
        info._tool_tracker.dispatch("", state)
    else:
        info._tool_tracker.dispatch(getattr(event, "tool_call_id", "") or "", state)
    SubagentManager._retire_subagent_oracle(info)


def _note_tool_result(info: SubagentInfo, event: Any) -> None:
    """Retire a call's attribution when its TERMINAL result arrives.

    ``EVENT_TOOL_RESULT`` is also emitted for non-terminal progress updates;
    treating one as the end would drop attribution while the long silent command
    is still running and flag a healthy agent. Failed, cancelled and refused calls
    are terminal too, the set ``AcpSessionHandle`` and ``acp.client`` use.

    Through the shared :class:`InFlightToolTracker` (``result``), a finished
    judged call passes the slot to the most recently dispatched call still
    running, so a file write returning beside a ``wait`` leaves the wait judged.
    The oracle is retired only when the tracker says the judged call changed.
    """
    terminal = getattr(event, "tool_status", "") in TERMINAL_TOOL_STATUSES
    if not (event.tool_final or terminal):
        return
    judged_changed = info._tool_tracker.result(
        getattr(event, "tool_call_id", "") or "", terminal=True
    )
    if judged_changed:
        SubagentManager._retire_subagent_oracle(info)


def _clear_tool_dispatch(info: SubagentInfo) -> None:
    """Drop every in-flight tool snapshot and retire the oracle with them.

    Called at the start of each prompt, so a call the previous turn never
    closed cannot be handed the slot when a later call returns.
    """
    info._tool_tracker.clear()
    SubagentManager._retire_subagent_oracle(info)


def _retire_subagent_oracle(info: SubagentInfo) -> None:
    """Retire this agent's stall oracle and bump its generation.

    Called whenever the judged tool call changes. Swaps in a ``fresh()`` oracle
    rather than clearing in place, since a walk against the previous command holds
    the old instance; the generation bump marks a verdict that outlived its tool
    as stale. The counterpart of ``AcpSessionHandle._retire_liveness_state``; each
    owns different machinery, so retirement is not part of the shared tracker.
    """
    oracle = info._stall_oracle
    info._stall_oracle = oracle.fresh() if oracle is not None else None
    info._stall_gen += 1


async def _stall_verdict(self: SubagentManager, info: SubagentInfo) -> tuple[str, str]:
    return await self._monitor._stall_verdict_impl(info)


async def _maybe_flag_stall(
    self: SubagentManager, agent_id: str, info: SubagentInfo, now: float
) -> None:
    return await self._monitor._maybe_flag_stall_impl(agent_id, info, now)


def _record_slow_command(info: SubagentInfo, idle: float) -> None:
    """Best-effort append of a stalled subagent's slow command for analysis.

    Writes to ``~/.kiro/crew/subagents/slow_commands.jsonl`` (rotated at
    1 MiB keeping one previous generation, survives per-agent folder
    cleanup). Deliberately separate from the
    tombstone path, which marks an agent dead — a stalled agent is still
    running.
    """
    try:
        record_slow_command(
            info.id,
            last_tool=_redact(info.last_tool or ""),
            tool_count=info.tool_count,
            turns=info.turns,
            idle_secs=int(idle),
            elapsed_secs=int(time.time() - info.started),
            parent_session=info.parent_session_key or "",
            session_sharing=info._session_sharing,
        )
    except Exception:
        logger.debug("Failed to record slow command for %s", info.id, exc_info=True)


def _dependency_coordinator(self: SubagentManager) -> Any:
    return self._monitor._dependency_coordinator_impl()


def dependency_coordinator(self: SubagentManager) -> Any:
    """The manager's ONE ``DependencyCoordinator`` (or None without a store).

    Public seam for the gateway: it registers this process-wide so the
    main chat and the monitors read the shared ``retry_at`` per scope, and
    subscribes the runner adapters' waiters to the same schedule.
    """
    return self._dependency_coordinator()


async def dependency_coordinator_async(self: SubagentManager) -> Any:
    """:meth:`dependency_coordinator` for an event-loop caller.

    The FIRST build runs ``rebuild()`` over every waiting row, so a loop
    caller that may be the first one takes it on the store's writer thread.
    """
    await self._admission.ensure_coordinator_async()
    return self._dependency_coordinator()


def _taskq_pump(self: SubagentManager) -> None:
    self._monitor._taskq_pump_impl()
