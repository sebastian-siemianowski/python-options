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
  const shouldReconnectRef = useRef(false);
  const generationRef = useRef(0);

  const connect = useCallback(() => {
    if (!shouldReconnectRef.current) return;
    if (
      wsRef.current?.readyState === WebSocket.OPEN
      || wsRef.current?.readyState === WebSocket.CONNECTING
    ) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}${path}`;
    setStatus('connecting');

    const ws = new WebSocket(url);
    const generation = generationRef.current;
    wsRef.current = ws;

    ws.onopen = () => {
      if (wsRef.current !== ws || generation !== generationRef.current) return;
      setStatus('connected');
      retriesRef.current = 0;
    };

    ws.onmessage = (event) => {
      if (wsRef.current !== ws || generation !== generationRef.current) return;
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
      if (wsRef.current !== ws || generation !== generationRef.current) return;
      setStatus('disconnected');
      wsRef.current = null;
      if (!shouldReconnectRef.current) return;
      // exponential backoff
      const delay = Math.min(
        RECONNECT_BASE_MS * Math.pow(2, retriesRef.current),
        RECONNECT_MAX_MS
      );
      retriesRef.current += 1;
      timerRef.current = setTimeout(connect, delay);
    };

    ws.onerror = () => {
      if (wsRef.current !== ws || generation !== generationRef.current) return;
      ws.close();
    };
  }, [path]);

  useEffect(() => {
    shouldReconnectRef.current = true;
    generationRef.current += 1;
    connect();
    // ping keepalive every 25s
    const pingInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send('ping');
      }
    }, 25000);

    return () => {
      shouldReconnectRef.current = false;
      generationRef.current += 1;
      clearInterval(pingInterval);
      if (timerRef.current) clearTimeout(timerRef.current);
      const ws = wsRef.current;
      wsRef.current = null;
      ws?.close();
    };
  }, [connect]);

  return { status, lastMessage };
}
