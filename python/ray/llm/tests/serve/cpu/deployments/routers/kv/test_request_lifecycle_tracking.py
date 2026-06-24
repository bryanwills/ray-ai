import sys
from types import SimpleNamespace

import pytest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams

import ray
import ray.cloudpickle
from ray import serve
from ray.llm._internal.serve.routing_policies.kv_aware.kv_aware_actor import (
    KV_ROUTER_ACTOR_NAME,
    KVRouterActor,
    get_worker_id,
)
from ray.llm._internal.serve.routing_policies.kv_aware.token_tracking import (
    enable_token_tracking,
)

# Cluster workers can't import this pytest module, so ship the actor classes
# defined here (and their FakeSelectionService) to actors by value, not by name.
ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])

REPLICA_UNIQUE_ID = "test-replica-uid"
WORKER_ID = get_worker_id(REPLICA_UNIQUE_ID)
# SamplingParams.max_tokens the engine reports as the request's expected output.
MAX_TOKENS = 20


def sampling(kind=RequestOutputKind.DELTA, max_tokens=MAX_TOKENS):
    return SamplingParams(output_kind=kind, max_tokens=max_tokens)


@pytest.fixture(scope="module", autouse=True)
def ray_cluster():
    if not ray.is_initialized():
        ray.init(address="auto")


def request_output(token_counts, prompt_len=4, finished=False):
    """A real vLLM RequestOutput: one CompletionOutput per entry in token_counts."""
    return RequestOutput(
        request_id="r",
        prompt=None,
        prompt_token_ids=list(range(prompt_len)),
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=i,
                text="",
                token_ids=list(range(n)),
                cumulative_logprob=None,
                logprobs=None,
            )
            for i, n in enumerate(token_counts)
        ],
        finished=finished,
    )


def delta_steps(num_tokens, prompt_len=4):
    """A DELTA-kind stream: one new token per step, last step finished."""
    return [
        request_output([1], prompt_len=prompt_len, finished=i == num_tokens - 1)
        for i in range(num_tokens)
    ]


class MockAsyncLLM:
    """Mocks AsyncLLM: ``generate`` yields a scripted output stream.

    This is the one test seam on CPU -- the GPU integration test covers the
    real engine; everything downstream of ``generate`` runs the production
    path here.
    """

    def __init__(self, script, error_after=None):
        self.script = script
        self.error_after = error_after

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        for i, output in enumerate(self.script):
            if self.error_after is not None and i == self.error_after:
                raise RuntimeError("engine failure")
            yield output


class FakeSelectionService:
    """Records the selection-service reservation calls the actor's lifecycle
    hooks make, standing in for the real pyo3 service on CPU. ``add_output_block``
    is synchronous as in the real binding; the rest are async."""

    def __init__(self):
        self.calls = []

    async def create_reservation(self, request):
        self.calls.append(
            (
                "create_reservation",
                request["reservation_id"],
                request["worker_id"],
                len(request["token_ids"]),
                request.get("expected_output_tokens"),
            )
        )

    async def prefill_complete(self, reservation_id):
        self.calls.append(("prefill_complete", reservation_id))

    def add_output_block(self, reservation_id, *, decay_fraction=None):
        self.calls.append(("add_output_block", reservation_id, decay_fraction))

    async def free_reservation(self, reservation_id):
        self.calls.append(("free_reservation", reservation_id))


@ray.remote(num_cpus=0)
class RecordingKVRouterActor(KVRouterActor):
    """In-process KVRouterActor with the event plane + Serve LongPoll stripped,
    recording the events it applies and the reservation calls it books into the
    (fake) selection service."""

    def __init__(self, block_size):
        self._block_size = block_size
        self._replica_id_by_worker = {}
        self._requests = {}
        self._pending_tasks = set()
        self._svc = FakeSelectionService()
        self._event_log = []

    async def on_lifecycle_events(self, events):
        self._event_log.extend(events)
        await super().on_lifecycle_events(events)

    def get_event_log(self):
        return self._event_log

    def get_dynamo_calls(self):
        return self._svc.calls


@ray.remote(num_cpus=0)
class RaisingActor:
    """A KV-router stand-in whose event ingest always raises, to prove the
    engine stream is never disrupted."""

    async def on_lifecycle_events(self, events):
        raise RuntimeError("actor down")


class LocalKVRouterActor(KVRouterActor):
    """In-process KVRouterActor with the event plane + Serve LongPoll stripped."""

    def __init__(self, block_size):
        self._block_size = block_size
        self._replica_id_by_worker = {}
        self._requests = {}
        self._pending_tasks = set()
        self._svc = FakeSelectionService()


@pytest.fixture
def build_token_tracking_engine(monkeypatch):
    def _build(script, actor, **engine_kwargs):
        def get_deployment_actor(name):
            assert name == KV_ROUTER_ACTOR_NAME
            return actor

        monkeypatch.setattr(serve, "get_deployment_actor", get_deployment_actor)
        monkeypatch.setattr(
            serve,
            "get_replica_context",
            lambda: SimpleNamespace(
                replica_id=SimpleNamespace(unique_id=REPLICA_UNIQUE_ID)
            ),
        )
        return enable_token_tracking(MockAsyncLLM)(script, **engine_kwargs)

    return _build


async def consume(stream, limit=None):
    """Drain ``stream``, optionally closing it early after ``limit`` outputs."""
    outputs = []
    async for output in stream:
        outputs.append(output)
        if limit is not None and len(outputs) == limit:
            await stream.aclose()
    return outputs


async def drain(engine):
    """Wait for the engine reporter's queued lifecycle batches to land."""
    await engine._kv_reporter.flush()


def decode_counts(events):
    return [args[1] for name, args in events if name == "on_decode_progress"]


def op_names(calls):
    return [c[0] for c in calls]


@pytest.mark.asyncio
async def test_basic_lifecycle(build_token_tracking_engine):
    """A streamed request reports add -> prefill -> exact decode counts -> done."""
    actor = RecordingKVRouterActor.remote(block_size=16)
    engine = build_token_tracking_engine(delta_steps(3, prompt_len=10), actor)

    prompt = {"prompt_token_ids": list(range(10))}
    outputs = await consume(engine.generate(prompt, sampling(), "req-1"))
    await drain(engine)

    assert ray.get(actor.get_event_log.remote()) == [
        ("on_request_added", ("req-1", WORKER_ID, list(range(10)), MAX_TOKENS)),
        ("on_prefill_complete", ("req-1",)),
        ("on_decode_progress", ("req-1", 1)),
        ("on_decode_progress", ("req-1", 2)),
        ("on_decode_progress", ("req-1", 3)),
        ("on_request_completed", ("req-1",)),
    ]
    assert outputs == engine.script


@pytest.mark.asyncio
async def test_in_order_reports(build_token_tracking_engine):
    """Back-to-back reports reach the actor in submission order.

    Plain fire-and-forget calls to an async actor would be executed out of order under load.
    """
    actor = RecordingKVRouterActor.remote(block_size=16)
    engine = build_token_tracking_engine(delta_steps(200), actor)

    await consume(engine.generate("p", sampling(max_tokens=200), "r"))
    await drain(engine)

    events = ray.get(actor.get_event_log.remote())
    assert events[0][0] == "on_request_added"
    assert events[1][0] == "on_prefill_complete"
    assert events[-1] == ("on_request_completed", ("r",))
    assert decode_counts(events) == list(range(1, 201))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", [RequestOutputKind.DELTA, RequestOutputKind.CUMULATIVE]
)
async def test_token_count_normalization(kind, build_token_tracking_engine):
    """DELTA and CUMULATIVE streams yield identical cumulative progress."""
    if kind is RequestOutputKind.DELTA:
        # Steps carry only new tokens: 1, then 2, then 1 -> 1, 3, 4.
        script = [request_output([n]) for n in (1, 2, 1)]
    else:
        # Steps carry the full output so far: 1, 3, 4 -> 1, 3, 4.
        script = [request_output([n]) for n in (1, 3, 4)]
    actor = RecordingKVRouterActor.remote(block_size=16)
    engine = build_token_tracking_engine(script, actor)

    await consume(engine.generate("p", sampling(kind=kind), "r"))
    await drain(engine)

    assert decode_counts(ray.get(actor.get_event_log.remote())) == [1, 3, 4]


@pytest.mark.asyncio
async def test_multi_candidate_sum(build_token_tracking_engine):
    """With n>1, progress sums the new tokens across all candidate outputs."""
    actor = RecordingKVRouterActor.remote(block_size=16)
    script = [request_output([1, 1]), request_output([1, 0], finished=True)]
    engine = build_token_tracking_engine(script, actor)

    await consume(engine.generate("p", sampling(), "r"))
    await drain(engine)

    assert decode_counts(ray.get(actor.get_event_log.remote())) == [2, 3]


@pytest.mark.asyncio
async def test_empty_steps_ignored(build_token_tracking_engine):
    """Token-less outputs (e.g. a finish-only chunk) emit no progress hooks."""
    actor = RecordingKVRouterActor.remote(block_size=16)
    script = [
        request_output([1]),
        request_output([0]),  # structural chunk: no new tokens
        request_output([1], finished=True),
    ]
    engine = build_token_tracking_engine(script, actor)

    await consume(engine.generate("p", sampling(), "r"))
    await drain(engine)

    events = ray.get(actor.get_event_log.remote())
    assert decode_counts(events) == [1, 2]
    assert [e for e in events if e[0] == "on_prefill_complete"] == [
        ("on_prefill_complete", ("r",))
    ]


@pytest.mark.asyncio
async def test_non_pretokenized_prompt(build_token_tracking_engine):
    """A non-pretokenized prompt (out-of-band engine call) reports no prompt
    tokens; the OpenAI serving layer always passes pre-tokenized prompts."""
    actor = RecordingKVRouterActor.remote(block_size=16)
    engine = build_token_tracking_engine(delta_steps(1, prompt_len=6), actor)

    await consume(engine.generate("plain text prompt", sampling(), "r"))
    await drain(engine)

    events = ray.get(actor.get_event_log.remote())
    assert events[0] == ("on_request_added", ("r", WORKER_ID, [], MAX_TOKENS))


@pytest.mark.asyncio
@pytest.mark.parametrize("early_drop", [False, True])
async def test_completed_exactly_once(early_drop, build_token_tracking_engine):
    """Completion fires exactly once on normal end and on early stream close."""
    actor = RecordingKVRouterActor.remote(block_size=16)
    engine = build_token_tracking_engine(delta_steps(3), actor)

    stream = engine.generate("p", sampling(), "r")
    await consume(stream, limit=1 if early_drop else None)
    await drain(engine)

    events = ray.get(actor.get_event_log.remote())
    assert [e for e in events if e[0] == "on_request_completed"] == [
        ("on_request_completed", ("r",))
    ]


@pytest.mark.asyncio
async def test_engine_error_still_completes(build_token_tracking_engine):
    """A mid-stream engine error propagates but still frees the request."""
    actor = RecordingKVRouterActor.remote(block_size=16)
    engine = build_token_tracking_engine(delta_steps(3), actor, error_after=1)

    with pytest.raises(RuntimeError, match="engine failure"):
        await consume(engine.generate("p", sampling(), "r"))
    await drain(engine)

    events = ray.get(actor.get_event_log.remote())
    assert events[-1] == ("on_request_completed", ("r",))


@pytest.mark.asyncio
async def test_zero_token_request(build_token_tracking_engine):
    """An output-less request (e.g. validation abort) is added and freed only."""
    actor = RecordingKVRouterActor.remote(block_size=16)
    engine = build_token_tracking_engine([request_output([0], finished=True)], actor)

    prompt = {"prompt_token_ids": [1, 2, 3]}
    await consume(engine.generate(prompt, sampling(), "r"))
    await drain(engine)

    assert ray.get(actor.get_event_log.remote()) == [
        ("on_request_added", ("r", WORKER_ID, [1, 2, 3], MAX_TOKENS)),
        ("on_request_completed", ("r",)),
    ]


@pytest.mark.asyncio
async def test_actor_failure_isolation(build_token_tracking_engine):
    """A failing actor never disrupts the engine's output stream."""
    engine = build_token_tracking_engine(delta_steps(2), RaisingActor.remote())

    outputs = await consume(engine.generate("p", sampling(), "r"))
    await drain(engine)  # the failed batches are dropped without raising

    assert len(outputs) == 2


def test_decorator_returns_subclass():
    """The decorator returns an isinstance-compatible subclass."""
    assert issubclass(enable_token_tracking(MockAsyncLLM), MockAsyncLLM)


@pytest.mark.asyncio
async def test_passthrough_without_actor(monkeypatch):
    """Outside a replica (no actor resolvable) the engine is a pure pass-through."""

    def _raise(name):
        raise RuntimeError("no actor")

    monkeypatch.setattr(serve, "get_deployment_actor", _raise)
    engine = enable_token_tracking(MockAsyncLLM)(delta_steps(2))

    outputs = await consume(engine.generate("p", sampling(), "r"))

    assert len(outputs) == 2
    assert engine._kv_reporter is None  # resolution failed; retried next call


@pytest.mark.asyncio
async def test_block_boundary_crossings():
    """Each ceil((prompt+output)/block_size) increase advances total_blocks."""
    actor = LocalKVRouterActor(block_size=16)
    await actor.on_request_added("r", 1, list(range(10)))
    assert (await actor.get_request_lifecycle("r"))["total_blocks"] == 1  # ceil(10/16)

    await actor.on_decode_progress("r", 6)  # 10+6=16 -> still 1 block
    assert (await actor.get_request_lifecycle("r"))["total_blocks"] == 1

    await actor.on_decode_progress("r", 7)  # 17 -> crosses into block 2
    assert (await actor.get_request_lifecycle("r"))["total_blocks"] == 2

    await actor.on_decode_progress("r", 39)  # 49 -> ceil=4, crosses two more at once
    snapshot = await actor.get_request_lifecycle("r")
    assert snapshot["total_blocks"] == 4
    assert snapshot["output_tokens"] == 39


@pytest.mark.asyncio
async def test_active_load_tracking():
    """Active load is per-worker; completion evicts the request entirely."""
    actor = LocalKVRouterActor(block_size=16)
    await actor.on_request_added("a", 1, list(range(8)))
    await actor.on_request_added("b", 1, [])
    await actor.on_request_added("c", 2, [])
    assert await actor.get_worker_active_load(1) == 2
    assert await actor.get_worker_active_load(2) == 1

    await actor.on_prefill_complete("a")
    await actor.on_decode_progress("a", 5)
    assert await actor.get_worker_active_load(1) == 2  # still active while decoding
    assert await actor.get_request_lifecycle("a") == {
        "worker_id": 1,
        "prompt_tokens": 8,
        "expected_output_tokens": None,
        "prefill_completed": True,
        "output_tokens": 5,
        "total_blocks": 1,
    }

    # Completion evicts (bounding memory to in-flight requests).
    await actor.on_request_completed("a")
    assert await actor.get_worker_active_load(1) == 1
    assert set(await actor.get_active_request_ids()) == {"b", "c"}
    assert await actor.get_request_lifecycle("a") is None

    # Hooks for an unknown request id are ignored.
    await actor.on_prefill_complete("missing")
    await actor.on_decode_progress("missing", 3)
    await actor.on_request_completed("missing")
    assert await actor.get_request_lifecycle("missing") is None


@pytest.mark.asyncio
async def test_end_to_end_actor_state(build_token_tracking_engine):
    """End-to-end: exact token counts land as actor block state over ``.remote``."""
    actor = RecordingKVRouterActor.remote(block_size=8)
    # prompt 12, block_size 8: baseline ceil(12/8)=2; boundaries at cumulative
    # output 5 and 13 -> 9 generated tokens cross only the first.
    engine = build_token_tracking_engine(delta_steps(9, prompt_len=12), actor)

    prompt = {"prompt_token_ids": list(range(12))}
    stream = engine.generate(prompt, sampling(), "req-e2e")
    for _ in range(9):
        await stream.__anext__()
    await drain(engine)

    # All outputs consumed but the stream is still open: the request is
    # tracked with its exact final counts.
    assert await actor.get_request_lifecycle.remote("req-e2e") == {
        "worker_id": WORKER_ID,
        "prompt_tokens": 12,
        "expected_output_tokens": MAX_TOKENS,
        "prefill_completed": True,
        "output_tokens": 9,
        "total_blocks": 3,
    }
    assert await actor.get_worker_active_load.remote(WORKER_ID) == 1

    # Stream end fires completion, which evicts the request.
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()
    await drain(engine)
    assert await actor.get_request_lifecycle.remote("req-e2e") is None
    assert await actor.get_active_request_ids.remote() == []
    assert await actor.get_worker_active_load.remote(WORKER_ID) == 0


@pytest.mark.asyncio
async def test_lifecycle_books_selection_service_load(build_token_tracking_engine):
    """A streamed request books create_reservation -> prefill_complete -> one
    add_output_block per crossed decode block -> free_reservation, in order."""
    actor = RecordingKVRouterActor.remote(block_size=8)
    # prompt 12 (2 blocks); cumulative output crosses into block 3 at 5 tokens
    # and block 4 at 13 tokens -> exactly two output blocks over 20 tokens.
    engine = build_token_tracking_engine(delta_steps(20, prompt_len=12), actor)

    prompt = {"prompt_token_ids": list(range(12))}
    await consume(engine.generate(prompt, sampling(), "req-1"))
    await drain(engine)

    calls = ray.get(actor.get_dynamo_calls.remote())
    assert op_names(calls) == (
        ["create_reservation", "prefill_complete"]
        + ["add_output_block"] * 2
        + ["free_reservation"]
    )
    assert calls[0] == ("create_reservation", "req-1", WORKER_ID, 12, MAX_TOKENS)
    assert calls[-1] == ("free_reservation", "req-1")


@pytest.mark.asyncio
async def test_decode_blocks_book_add_output_block():
    """Each crossed decode block books one add_output_block in the service."""
    actor = LocalKVRouterActor(block_size=16)
    await actor.on_request_added("r", WORKER_ID, list(range(10)))  # 1 prompt block
    await actor.on_decode_progress("r", 6)  # 16 -> still 1 block
    await actor.on_decode_progress("r", 7)  # 17 -> crosses into block 2
    await actor.on_decode_progress("r", 39)  # 49 -> ceil=4, crosses two more

    assert (
        op_names(actor._svc.calls) == ["create_reservation"] + ["add_output_block"] * 3
    )


@pytest.mark.asyncio
async def test_expected_output_tokens_sets_decay_fraction():
    """With an output-length estimate each booked decode block decays by the
    remaining fraction; without one the block carries no decay."""
    actor = LocalKVRouterActor(block_size=8)
    await actor.on_request_added(
        "r", WORKER_ID, list(range(8)), expected_output_tokens=40
    )
    await actor.on_decode_progress("r", 8)  # total 16 -> crosses into block 2

    block_calls = [c for c in actor._svc.calls if c[0] == "add_output_block"]
    assert block_calls == [("add_output_block", "r", pytest.approx(1.0 - 8 / 40))]

    bare = LocalKVRouterActor(block_size=8)
    await bare.on_request_added("r", WORKER_ID, list(range(8)))  # no estimate
    await bare.on_decode_progress("r", 8)
    assert [c for c in bare._svc.calls if c[0] == "add_output_block"] == [
        ("add_output_block", "r", None)
    ]


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
