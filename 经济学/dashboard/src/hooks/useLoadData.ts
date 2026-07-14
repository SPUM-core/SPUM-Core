import { useEffect } from 'react';
import { useStore } from '../store';
import type { VisData } from '../types';

export function useLoadData() {
  const setData = useStore((s) => s.setData);
  const setLoading = useStore((s) => s.setLoading);
  const setError = useStore((s) => s.setError);

  useEffect(() => {
    fetch('/vis_data.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d: VisData) => {
        setData(d);
      })
      .catch((e) => {
        setError(e.message);
      });
  }, [setData, setLoading, setError]);
}
