"""A stop reaches the reply through Redis, after a Redis restart and on a Redis Cluster too.

With Redis configured, stopping a reply publishes a command on a Redis channel and every
instance's task command listener cancels the reply it runs. Three fixes keep that listener alive:

* #27104 (issue #26779): a default socket timeout cut the listener's idle subscription every
  few seconds; `REDIS_SOCKET_TIMEOUT` now defaults to unset. The pinned redis-py no longer
  applies a socket timeout to a subscription's blocking read, so the default is pinned where it
  still shows: a Redis that answers slowly is waited for. Every signed-in request asks Redis
  whether its token was revoked and lets the request through when that fails, so a read that
  gave up shows as a signed-out token accepted.
* bf3a58dbcd (#28909): the listener subscribed once, so after a Redis restart no instance heard a
  stop again. It now subscribes again whenever its stream ends, backing off while Redis is down.
* a5ea8b0b8 (PR #29165, issue #19840): on a Redis Cluster the listener asked for a subscription
  before the cluster client had learned the slot layout, so it never subscribed and a stop was
  dropped while the reply ran to the end. It now initializes the client first. The pinned
  redis-py 8.0.1 subscribes on an uninitialized cluster client as well, so the cluster test is
  journey coverage here and unit/chat/test_socket_cluster_and_packets.py keeps the narrow guard.

The cluster test is also the regression for open-webui/open-webui#33765: on dev 393dd6689 the
terminal connection hub built its own cluster client and subscribed before initializing it, so
every chat request on a Redis Cluster failed and the slow reply never reached the provider.

One instance runs on a real Redis the test can stall (`DEBUG SLEEP`), stop and start again,
another on a one-node Redis Cluster. The subscription is read off Redis itself (`PUBSUB NUMSUB`),
and a stop is proven by a slow reply that ends before its last piece.

Twin of unit/chat/test_socket_runtime.py, unit/chat/test_socket_and_redis_runtime.py and
unit/chat/test_socket_cluster_and_packets.py for the listener.

Discriminates: passes on dev ef67cc3fa; with a fixed 2 second default socket timeout the
signed-out token is accepted while Redis stalls; with the listener returning once its stream
ends, the restarted Redis never gets its subscriber back and the stop is lost. Removing the
`initialize()` call stays green with redis-py 8.0.1 (see above). The cluster test fails on dev
393dd6689 (#33765, no reply request reaches the provider) and passes with the hub fix.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Iterator

import pytest
import redis

from harness.actors import create_user
from harness.backends import RedisProcess, wait_until
from harness.chat import wait_for_reply
from harness.inflight import LAST_PIECE, start_slow_reply, wait_until_streaming
from harness.instance import free_port
from harness.socket_client import connected

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

STOP_CHANNEL = "open-webui:tasks:commands"
# longer than redis-py's own reconnect attempts, so the listener's stream really ends
OUTAGE_SECONDS = 15
STALL_SECONDS = 4


def _subscribers(url: str) -> int:
    client = redis.Redis.from_url(url)
    try:
        [(_, count)] = client.pubsub_numsub(STOP_CHANNEL)
        return count
    except redis.exceptions.RedisError:
        return 0
    finally:
        client.close()


def _stop_mid_stream(instance) -> str:
    """Start a slow reply, stop its chat, and return what the reply ended with."""
    account = create_user(instance)
    with account.client() as client, connected(account) as tab:
        turn = start_slow_reply(client, instance.upstream, chunk_delay=0.3)
        wait_until_streaming(tab, turn.chat_id)
        stopped = client.post(f"/api/tasks/chat/{turn.chat_id}/stop")
        assert stopped.status_code == 200, stopped.text
        return wait_for_reply(client, turn, timeout=30)["content"]


@pytest.fixture(scope="module")
def restartable_redis() -> Iterator[RedisProcess]:
    server = RedisProcess("--enable-debug-command", "local")
    server.start()
    yield server
    server.close()


@pytest.fixture(scope="module")
def redis_instance(instance_with, restartable_redis):
    return instance_with({"REDIS_URL": restartable_redis.url})


def _stall(url: str) -> None:
    client = redis.Redis.from_url(url)
    try:
        client.execute_command("DEBUG", "SLEEP", STALL_SECONDS)
    finally:
        client.close()


def test_a_stop_reaches_the_reply_through_redis(redis_instance, restartable_redis):
    assert wait_until(lambda: _subscribers(restartable_redis.url) >= 1, timeout=15)

    content = _stop_mid_stream(redis_instance)

    assert content.startswith("part-0"), content
    assert LAST_PIECE not in content, "the stop never reached the reply, it ran to the end"


def test_only_a_stop_command_cancels_the_reply(redis_instance, restartable_redis):
    assert wait_until(lambda: _subscribers(restartable_redis.url) >= 1, timeout=15)
    account = create_user(redis_instance)
    publisher = redis.Redis.from_url(restartable_redis.url)
    with account.client() as client:
        turn = start_slow_reply(client, redis_instance.upstream, chunk_delay=0.1)
        [task_id] = client.get(f"/api/tasks/chat/{turn.chat_id}").json()["task_ids"]
        publisher.publish(STOP_CHANNEL, json.dumps({"action": "ping", "task_id": task_id}))
        publisher.publish(STOP_CHANNEL, b"not json")
        content = wait_for_reply(client, turn, timeout=30)["content"]
    publisher.close()

    assert LAST_PIECE in content, "a command other than stop cut the reply short"


def test_a_slow_redis_answer_is_waited_for_by_default(redis_instance, restartable_redis):
    account = create_user(redis_instance)
    with account.client() as client:
        client.post("/api/v1/auths/signout").raise_for_status()
        assert client.get("/api/v1/auths/").status_code == 401, "signing out did not revoke"

        stall = threading.Thread(target=_stall, args=(restartable_redis.url,))
        stall.start()
        time.sleep(0.5)
        try:
            during_stall = client.get("/api/v1/auths/")
        finally:
            stall.join()

    assert during_stall.status_code == 401, (
        "a Redis read gave up on a slow answer, so the revocation check let a signed-out token "
        "through: the socket timeout is no longer unset by default (#26779)"
    )


def test_after_a_redis_restart_a_stop_still_reaches_the_reply(redis_instance, restartable_redis):
    assert wait_until(lambda: _subscribers(restartable_redis.url) >= 1, timeout=15)
    restartable_redis.stop()
    time.sleep(OUTAGE_SECONDS)
    restartable_redis.start()

    assert wait_until(lambda: _subscribers(restartable_redis.url) >= 1, timeout=60), (
        "after Redis came back the instance never subscribed to stop commands again (#28909)"
    )
    content = _stop_mid_stream(redis_instance)
    assert LAST_PIECE not in content, "the stop was lost after the Redis restart"


@pytest.fixture(scope="module")
def cluster_redis() -> Iterator[RedisProcess]:
    # the cluster bus defaults to the port plus 10000, which a high free port overflows
    server = RedisProcess(
        "--cluster-enabled",
        "yes",
        "--cluster-config-file",
        "nodes.conf",
        "--cluster-port",
        str(free_port()),
    )
    server.start()
    client = redis.Redis(port=server.port, decode_responses=True)
    client.execute_command("CLUSTER", "ADDSLOTSRANGE", "0", "16383")
    assert wait_until(
        lambda: "cluster_state:ok" in client.execute_command("CLUSTER", "INFO"), timeout=15
    )
    client.close()
    yield server
    server.close()


@pytest.fixture(scope="module")
def cluster_instance(instance_with, cluster_redis):
    return instance_with({"REDIS_URL": cluster_redis.url, "REDIS_CLUSTER": "true"})


def test_on_a_redis_cluster_a_stop_reaches_the_reply(cluster_instance, cluster_redis):
    assert wait_until(lambda: _subscribers(cluster_redis.url) >= 1, timeout=15), (
        "the listener never subscribed on the Redis Cluster (#19840)"
    )

    content = _stop_mid_stream(cluster_instance)

    assert content.startswith("part-0"), content
    assert LAST_PIECE not in content, "the stop was dropped and the reply ran to the end"
