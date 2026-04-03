import { useEffect, useRef, useState, useCallback } from 'react';

export interface WSMessage {
  type: string;
  [key: string]: unknown;
}

export type WSStatus = 'connecting' | 'connected' | 'disconnected';

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 30000;

/**
 * Story 6.4: WebSocket hook with auto-reconnect and exponential backoff.
 */
export function useWebSocket(path: string = '/ws') {
  const [status, setStatus] = useState<WSStatus>('disconnected');
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}${path}`;
    setStatus('connecting');

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus('connected');
      retriesRef.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WSMessage;
        if (data.type !== 'pong') {
          setLastMessage(data);
        }
      } catch {
        // ignore non-JSON messages
      }
    };

    ws.onclose = () => {
      setStatus('disconnected');
      wsRef.current = null;
      // exponential backoff
      const delay = Math.min(
        RECONNECT_BASE_MS * Math.pow(2, retriesRef.current),
        RECONNECT_MAX_MS
      );
      retriesRef.current += 1;
      timerRef.current = setTimeout(connect, delay);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [path]);

  useEffect(() => {
    connect();
    // ping keepalive every 25s
    const pingInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send('ping');
      }
    }, 25000);

    return () => {
      clearInterval(pingInterval);
      if (timerRef.current) clearTimeout(timerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { status, lastMessage };
}
