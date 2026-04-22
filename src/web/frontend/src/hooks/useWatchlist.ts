import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, type WatchlistResponse } from '../api';

const WATCHLIST_KEY = ['watchlist'] as const;

/**
 * React Query hook for the server-side watchlist.
 *
 * Returns the sorted list of persisted symbols plus helpers for adding and
 * removing entries. Mutations use optimistic updates so the UI feels instant
 * and roll back automatically if the server rejects the change.
 */
export function useWatchlist() {
  const qc = useQueryClient();

  const query = useQuery<WatchlistResponse>({
    queryKey: WATCHLIST_KEY,
    queryFn: api.watchlistGet,
    staleTime: 60_000,
  });

  const add = useMutation<WatchlistResponse, Error, string, { previous?: WatchlistResponse }>({
    mutationFn: (symbol: string) => api.watchlistAdd(symbol),
    onMutate: async (symbol) => {
      const sym = symbol.trim().toUpperCase();
      await qc.cancelQueries({ queryKey: WATCHLIST_KEY });
      const previous = qc.getQueryData<WatchlistResponse>(WATCHLIST_KEY);
      const prior = previous?.symbols ?? [];
      if (!prior.includes(sym)) {
        qc.setQueryData<WatchlistResponse>(WATCHLIST_KEY, { symbols: [...prior, sym] });
      }
      return { previous };
    },
    onError: (_err, _sym, ctx) => {
      if (ctx?.previous) qc.setQueryData(WATCHLIST_KEY, ctx.previous);
    },
    onSuccess: (data) => {
      qc.setQueryData(WATCHLIST_KEY, data);
    },
  });

  const remove = useMutation<WatchlistResponse, Error, string, { previous?: WatchlistResponse }>({
    mutationFn: (symbol: string) => api.watchlistRemove(symbol),
    onMutate: async (symbol) => {
      const sym = symbol.trim().toUpperCase();
      await qc.cancelQueries({ queryKey: WATCHLIST_KEY });
      const previous = qc.getQueryData<WatchlistResponse>(WATCHLIST_KEY);
      const prior = previous?.symbols ?? [];
      qc.setQueryData<WatchlistResponse>(WATCHLIST_KEY, {
        symbols: prior.filter((s) => s !== sym),
      });
      return { previous };
    },
    onError: (_err, _sym, ctx) => {
      if (ctx?.previous) qc.setQueryData(WATCHLIST_KEY, ctx.previous);
    },
    onSuccess: (data) => {
      qc.setQueryData(WATCHLIST_KEY, data);
    },
  });

  return {
    symbols: query.data?.symbols ?? [],
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    add,
    remove,
  };
}
