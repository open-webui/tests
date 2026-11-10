"""Hold a new chat's live events until the page has taken the chat's id, as a real network does.

The first send of a new chat learns the chat's id from the answer to `/api/chat/completions`, and
the page drops every event of a chat it does not know yet. A provider that answers at once (a
pipe, a local stand-in) can stream the whole reply over the socket before a slow page has read
that answer, and the reply is then lost, which no real provider is fast enough to cause.
`hold_new_chat_events(page)` holds the chat events the socket brings while a new chat's send is
out, until the page joins the chat's room right after taking the id. It runs inside the page, so
every other frame passes untouched and nothing waits on the test's own thread.

The page still gets the browser's own sockets: the gate listens on each one as it is made and
wraps the one `send` they share. A subclass of `WebSocket` in their place is refused by noVNC,
which reads a socket's methods from its own prototype, so the Chrome viewer and the Desktop tab
never connected.
"""

from __future__ import annotations

from playwright.sync_api import Page

GATE = r"""
(() => {
    const NativeWebSocket = window.WebSocket;
    const nativeFetch = window.fetch;
    const held = [];
    const released = new WeakSet();
    let waiting = false;

    // a socket.io event frame: type 42, an optional namespace and ack id, then the arguments
    const eventOf = (data) => {
        const match = typeof data === 'string' && data.match(/^42(?:\/[^,]*,)?\d*(\[[\s\S]*\])$/);
        if (!match) return {};
        try {
            const [name, payload] = JSON.parse(match[1]);
            return { name, payload: payload ?? {} };
        } catch {
            return {};
        }
    };

    const release = () => {
        waiting = false;
        for (const [socket, data] of held.splice(0)) {
            const event = new MessageEvent('message', { data });
            released.add(event);
            socket.dispatchEvent(event);
        }
    };

    const hold = (socket) => {
        socket.addEventListener('message', (event) => {
            if (released.has(event) || !waiting) return;
            const { name, payload } = eventOf(event.data);
            if (name !== 'events' || !payload.chat_id) return;
            event.stopImmediatePropagation();
            held.push([socket, event.data]);
        });
        return socket;
    };

    // the page's sockets stay the browser's own: noVNC refuses a subclass
    window.WebSocket = new Proxy(NativeWebSocket, {
        construct: (target, args) => hold(new target(...args))
    });

    const nativeSend = NativeWebSocket.prototype.send;
    NativeWebSocket.prototype.send = function (data) {
        nativeSend.call(this, data);
        const { name, payload } = eventOf(data);
        // the page joins a chat's room right after it took the chat's id
        if (name === 'events:chat' && payload.data?.type === 'join') release();
    };

    const isNewChatSend = (input, init) => {
        const request = input instanceof Request ? input : null;
        const url = new URL(request ? request.url : String(input), location.href);
        const method = (init?.method ?? request?.method ?? 'GET').toUpperCase();
        if (method !== 'POST' || !url.pathname.endsWith('/api/chat/completions')) return false;
        try {
            return !JSON.parse(init?.body ?? '{}').chat_id;
        } catch {
            return false;
        }
    };

    window.fetch = function (input, init) {
        const answer = nativeFetch.call(this, input, init);
        if (!isNewChatSend(input, init)) return answer;
        waiting = true;
        answer.then(async (response) => {
            const json = response.ok && response.headers.get('content-type')?.includes('json');
            const chatId = json ? (await response.clone().json().catch(() => ({}))).chat_id : null;
            // a refused send or a temporary chat: the page joins no room
            if (!chatId || String(chatId).startsWith('local:')) release();
        }, release);
        return answer;
    };
})();
"""


def hold_new_chat_events(page: Page) -> None:
    """Install the gate in every document the page loads; call before its first navigation."""
    page.add_init_script(GATE)
