"""
Legacy ingestion pipeline utilities.

DEPRECATED: This module contains the legacy IngestionPipeline which is deprecated.
Use TrackedIngestionPipeline from fetchcraft.ingestion.pipeline instead.

This module is kept for backwards compatibility and provides:
- Queue message utilities (QueueMessage, to_json, from_json)
- AsyncQueueBackend protocol
- Legacy Record, Source, Transformation, Sink protocols
- Legacy IngestionPipeline (deprecated)
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterable, Awaitable, Callable, Iterable, List, Optional, Protocol, Dict

from fetchcraft.connector import Connector
from fetchcraft.parsing.base import DocumentParser

UTC = timezone.utc

# -----------------------------
# Pipeline (async) with multiple sinks + deferred steps
# -----------------------------

MAIN_QUEUE = "ingest.main"
DEFER_QUEUE = "ingest.deferred"
ERROR_QUEUE = "ingest.error"


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


def to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def from_json(s: str) -> Any:
    return json.loads(s)


# -----------------------------
# Core data model & Protocols
# -----------------------------

@dataclass
class Record:
    id: str
    payload: dict
    meta: dict = field(default_factory=dict)


class Source(Protocol):
    async def read(self) -> AsyncIterable[Record]:  # async generator
        ...


class Transformation(Protocol):
    async def process(self, record: Record) -> Record | Iterable[Record] | None:
        ...


class Sink(Protocol):
    async def write(self, record: Record) -> None:
        ...


# -----------------------------
# Async Queue interface (pluggable)
# -----------------------------

@dataclass
class QueueMessage:
    id: str
    body: dict


class AsyncQueueBackend(Protocol):
    async def enqueue(self, queue_name: str, body: dict, delay_seconds: int = 0) -> str:
        ...

    async def lease_next(self, queue_name: str, lease_seconds: int = 30) -> Optional[QueueMessage]:
        ...

    async def ack(self, queue_name: str, message_id: str) -> None:
        ...

    async def nack(self, queue_name: str, message_id: str, requeue_delay_seconds: int = 0) -> None:
        ...

    async def has_pending(self, *queue_names: str) -> bool:
        ...


# -----------------------------
# Async workers
# -----------------------------

@dataclass
class WorkerConfig:
    queue_name: str
    lease_seconds: int = 60
    poll_interval: float = 0.2
    max_retries: int = 5
    backoff_seconds: float = 5.0


class Worker:
    def __init__(self, name: str, backend: AsyncQueueBackend, handler: Callable[[dict], Awaitable[None]], cfg: WorkerConfig, error_queue: Optional[str] = None):
        self.name = name
        self.backend = backend
        self.handler = handler
        self.cfg = cfg
        self.error_queue = error_queue
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

    async def start(self):
        self._task = asyncio.create_task(self._run(), name=f"worker:{self.name}")

    async def _run(self):
        while not self._stopped.is_set():
            msg = await self.backend.lease_next(self.cfg.queue_name, lease_seconds=self.cfg.lease_seconds)
            if not msg:
                await asyncio.sleep(self.cfg.poll_interval)
                continue
            try:
                await self.handler(msg.body)
            except Exception as e:  # noqa: BLE001
                body = msg.body
                attempts = int(body.get("__attempts__", 0)) + 1
                body["__attempts__"] = attempts
                if attempts >= self.cfg.max_retries:
                    # Send to error queue instead of dropping
                    if self.error_queue:
                        error_body = {
                            "type": "error",
                            "original_queue": self.cfg.queue_name,
                            "message_id": msg.id,
                            "attempts": attempts,
                            "error": str(e),
                            "error_type": e.__class__.__name__,
                            "original_body": body,
                        }
                        await self.backend.enqueue(self.error_queue, body=error_body)
                        print(f"[Worker {self.name}] moved {msg.id} to error queue after {attempts} attempts: {e}")
                    else:
                        print(f"[Worker {self.name}] dropping {msg.id} after {attempts} attempts: {e}")
                    await self.backend.ack(self.cfg.queue_name, msg.id)
                else:
                    await self.backend.nack(self.cfg.queue_name, msg.id, requeue_delay_seconds=int(self.cfg.backoff_seconds * attempts))
            else:
                await self.backend.ack(self.cfg.queue_name, msg.id)

    async def stop(self):
        self._stopped.set()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


@dataclass
class StepSpec:
    step: Transformation
    deferred: bool = False


class IngestionPipeline:
    def __init__(self, backend: Optional[AsyncQueueBackend] = None):
        self.backend = backend
        self._source: Optional[Source] = None
        self._steps: List[StepSpec] = []
        self._sinks: List[Sink] = []
        self._main_worker: Optional[Worker] = None
        self._defer_worker: Optional[Worker] = None

    # Builder API
    def source(self, src: Source) -> "IngestionPipeline":
        self._source = src
        return self

    def add_transformation(self, step: Transformation, deferred: bool = False) -> "IngestionPipeline":
        self._steps.append(StepSpec(step=step, deferred=deferred))
        return self

    def add_sink(self, sink: Sink) -> "IngestionPipeline":
        self._sinks.append(sink)
        return self

    def _validate(self):
        if not self._source:
            raise ValueError("source() is required")
        if not self._sinks:
            raise ValueError("At least one sink is required. Use add_sink(...)")

    async def run(self):
        self._validate()
        # Start workers
        self._main_worker = Worker("main", self.backend, self._handle_main, WorkerConfig(queue_name=MAIN_QUEUE), error_queue=ERROR_QUEUE)
        self._defer_worker = Worker("deferred", self.backend, self._handle_deferred, WorkerConfig(queue_name=DEFER_QUEUE), error_queue=ERROR_QUEUE)
        await asyncio.gather(self._main_worker.start(), self._defer_worker.start())

        # Enqueue initial records
        async for rec in self._source.read():
            await self.backend.enqueue(MAIN_QUEUE, body={"type": "record", "record": dataclasses.asdict(rec)})

    async def wait_until_idle(
        self,
        poll_interval: float = 0.5,
        grace_seconds: float = 2.0,
    ) -> None:
        """
        Block until both main and deferred queues have no ready/leased messages.
        Grace period is to avoid races where new messages are enqueued between checks.
        """
        # First: loop until we *see* empty queues
        print(f"[wait_until_idle] Waiting for queues to drain...")
        iteration = 0
        while True:
            if hasattr(self.backend, "has_pending"):
                has_work = await self.backend.has_pending(MAIN_QUEUE, DEFER_QUEUE)  # type: ignore[attr-defined]
            else:
                # Fallback for in-memory/backends without has_pending: just sleep.
                has_work = True
            if not has_work:
                print(f"[wait_until_idle] Queues empty after {iteration} checks")
                break
            if iteration % 10 == 0:
                print(f"[wait_until_idle] Still waiting... (iteration {iteration})")
            iteration += 1
            await asyncio.sleep(poll_interval)

        # Second: short grace period to be sure nothing new got enqueued
        print(f"[wait_until_idle] Starting {grace_seconds}s grace period...")
        deadline = asyncio.get_event_loop().time() + grace_seconds
        grace_check = 0
        while asyncio.get_event_loop().time() < deadline:
            if hasattr(self.backend, "has_pending"):
                has_work = await self.backend.has_pending(MAIN_QUEUE, DEFER_QUEUE)  # type: ignore[attr-defined]
            else:
                has_work = True
            if has_work:
                # new work appeared – go back to main loop
                print(f"[wait_until_idle] Work appeared during grace period (check {grace_check}), restarting...")
                return await self.wait_until_idle(poll_interval, grace_seconds)
            grace_check += 1
            await asyncio.sleep(poll_interval)
        
        print(f"[wait_until_idle] Grace period complete, queues are idle!")

    async def _start_workers(self) -> None:
        self._main_worker = Worker(
            "main",
            self.backend,
            self._handle_main,
            WorkerConfig(queue_name=MAIN_QUEUE),
            error_queue=ERROR_QUEUE,
        )
        self._defer_worker = Worker(
            "deferred",
            self.backend,
            self._handle_deferred,
            WorkerConfig(queue_name=DEFER_QUEUE),
            error_queue=ERROR_QUEUE,
        )
        await asyncio.gather(
            self._main_worker.start(),
            self._defer_worker.start(),
        )

    async def shutdown(self) -> None:
        if self._main_worker:
            await self._main_worker.stop()
        if self._defer_worker:
            await self._defer_worker.stop()

    async def run_job(self) -> None:
        """
        Fire off an ingestion job, wait until all documents (including deferred work)
        are done, then shut down the pipeline.
        """
        self._validate()

        # 1) start workers
        await self._start_workers()

        # 2) enqueue source records
        async for rec in self._source.read():  # type: ignore[union-attr]
            await self.backend.enqueue(
                MAIN_QUEUE,
                body={"type": "record", "record": dataclasses.asdict(rec)},
            )

        # 3) wait until the system is idle (no pending work)
        await self.wait_until_idle()

        # 4) stop workers and return
        await self.shutdown()

    # Message handlers
    async def _handle_main(self, body: dict):
        if body.get("type") != "record":
            return
        rec = Record(**body["record"])  # type: ignore[arg-type]

        for spec in self._steps:
            if spec.deferred:
                await self.backend.enqueue(
                    DEFER_QUEUE,
                    body={
                        "type": "deferred",
                        "step_name": spec.step.__class__.__name__,
                        "record": dataclasses.asdict(rec),
                        "pipeline_steps": self._serialize_steps(),
                    },
                )
                continue
            rec = await self._apply_step(spec.step, rec)
            if rec is None:
                return

        await self._write_to_sinks(rec)

    async def _handle_deferred(self, body: dict):
        if body.get("type") != "deferred":
            return
        rec = Record(**body["record"])  # type: ignore[arg-type]
        steps = self._deserialize_steps(body["pipeline_steps"])  # keep same instances
        target_step_name = body["step_name"]

        seen = False
        for spec in steps:
            if not seen:
                if spec.step.__class__.__name__ == target_step_name:
                    rec = await self._apply_step(spec.step, rec)
                    if rec is None:
                        return
                    seen = True
                continue
            if not spec.deferred:
                rec = await self._apply_step(spec.step, rec)
                if rec is None:
                    return
            else:
                await self.backend.enqueue(
                    DEFER_QUEUE,
                    body={
                        "type": "deferred",
                        "step_name": spec.step.__class__.__name__,
                        "record": dataclasses.asdict(rec),
                        "pipeline_steps": self._serialize_steps(),
                    },
                )
                return

        await self._write_to_sinks(rec)

    async def _apply_step(self, step: Transformation, rec: Record) -> Optional[Record]:
        # Allow sync Transformations via to_thread
        if asyncio.iscoroutinefunction(step.process):
            out = await step.process(rec)  # type: ignore[misc]
        else:
            out = await asyncio.to_thread(step.process, rec)

        if out is None:
            return None
        if isinstance(out, Record):
            return out
        # Fan-out for iterables
        for child in out:  # type: ignore[assignment]
            await self.backend.enqueue(MAIN_QUEUE, body={"type": "record", "record": dataclasses.asdict(child)})
        return None

    async def _write_to_sinks(self, rec: Record) -> None:
        # write to all sinks concurrently
        await asyncio.gather(*(self._call_sink(s, rec) for s in self._sinks))

    async def _call_sink(self, sink: Sink, rec: Record):
        try:
            if asyncio.iscoroutinefunction(sink.write):
                await sink.write(rec)  # type: ignore[misc]
            else:
                await asyncio.to_thread(sink.write, rec)
        except Exception as e:
            # Send sink errors to error queue
            error_body = {
                "type": "sink_error",
                "sink": sink.__class__.__name__,
                "error": str(e),
                "error_type": e.__class__.__name__,
                "record": dataclasses.asdict(rec),
            }
            await self.backend.enqueue(ERROR_QUEUE, body=error_body)
            print(f"Error writing to sink {sink}: {e} - sent to error queue")

    def _serialize_steps(self) -> list[dict]:
        return [
            {"cls": spec.step.__class__.__name__, "deferred": spec.deferred}
            for spec in self._steps
        ]

    def _deserialize_steps(self, cfg: list[dict]) -> List[StepSpec]:
        name_to_spec = {spec.step.__class__.__name__: spec for spec in self._steps}
        out: List[StepSpec] = []
        for c in cfg:
            spec = name_to_spec.get(c["cls"])  # type: ignore[index]
            if spec:
                out.append(StepSpec(step=spec.step, deferred=bool(c.get("deferred", False))))
        return out


class ConnectorSource(Source):
    def __init__(self, connector: Connector, parser: Optional[DocumentParser] = None, parser_map: Optional[Dict[str, DocumentParser]] = None):
        self.connector = connector
        self.parser_map = parser_map or {}
        if parser and "default" not in self.parser_map:
            self.parser_map["default"] = parser

    async def read(self) -> AsyncIterable[Record]:  # async generator
        async for file in self.connector.read():
            parser = self.parser_map.get(file.mimetype, self.parser_map.get("default"))
            documents = parser.parse(file)
            async for doc in documents:
                yield Record(id=str(file.path), payload={"document": doc.model_dump()}, meta={"path": str(file.path)})
