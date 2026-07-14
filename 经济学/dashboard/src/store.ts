import { create } from 'zustand';
import type { VisData } from './types';

interface AppState {
  data: VisData | null;
  loading: boolean;
  error: string | null;

  setData: (data: VisData) => void;
  setLoading: (v: boolean) => void;
  setError: (e: string | null) => void;
}

export const useStore = create<AppState>((set) => ({
  data: null,
  loading: true,
  error: null,

  setData: (data) => set({ data, loading: false }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error, loading: false }),
}));
