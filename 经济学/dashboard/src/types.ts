export interface Stock {
  sym: string;
  name: string;
  listing: string;
  s0: number[];
  label: string;
  s0_normalized: number[];
}

export interface Group {
  label: string;
  s0: number[];
  members: string[];
  size: number;
}

export interface Pair {
  label: string;
  target: string;
  target_sym: string;
  source: string;
  source_sym: string;
  h1_ridge: number | null;
  h1_xgb: number | null;
  h5_ridge: number | null;
  h5_xgb: number | null;
}

export interface VolSeries {
  dates: string[];
  vol: number[];
}

export interface Stats {
  total_stocks: number;
  total_groups: number;
  multi_groups: number;
  total_pairs: number;
  xgb_avg: number;
  ridge_avg: number;
  xgb_max: number;
  xgb_min: number;
}

export interface BacktestMetrics {
  ic_cross_sectional?: {
    ic_mean: number;
    ic_std: number;
    ic_ir: number;
    n: number;
    by_label?: Record<string, number>;
  };
  ic_time_series?: {
    n_stocks: number;
    ic_mean: number;
    ic_median: number;
    ic_pos_ratio: number;
  };
  strategy_comparison?: StrategyRow[];
  export_time?: string;
  version?: string;
}

export interface StrategyRow {
  name: string;
  sharpe: number;
  total_return: number;
  max_drawdown: number;
  calmar: number;
  hit_rate: number;
  annual_return: number;
  annual_vol: number;
}

export interface VisData {
  stocks: Stock[];
  groups: Group[];
  pairs: Pair[];
  vol_data: Record<string, VolSeries>;
  stats: Stats;
  backtest?: BacktestMetrics;
  export_time: string;
}

export const WUXING = ['木', '火', '土', '金', '水'] as const;

export const WUXING_COLORS: Record<string, string> = {
  '木': '#4ade80',
  '火': '#f87171',
  '土': '#fbbf24',
  '金': '#e2e8f0',
  '水': '#60a5fa',
};

export const WUXING_COLORS_HEX: Record<string, string> = {
  '木': '#4ade80',
  '火': '#f87171',
  '土': '#fbbf24',
  '金': '#e2e8f0',
  '水': '#60a5fa',
};
