package backtest

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// WalkForwardConfig controls the rolling optimise-then-test procedure.
type WalkForwardConfig struct {
	TrainDays int      // in-sample window used to pick parameters
	TestDays  int      // out-of-sample window traded with those parameters
	Grid      []Params // candidate thresholds
}

// ParseGrid parses a grid specification of the form "entry=start:stop:step,exit=start:stop:step"
// into candidate parameter pairs with Exit < Entry.
func ParseGrid(spec string) ([]Params, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return nil, fmt.Errorf("empty grid specification")
	}

	parts := strings.Split(spec, ",")
	if len(parts) != 2 {
		return nil, fmt.Errorf("expected 'entry=start:stop:step,exit=start:stop:step', got %q", spec)
	}

	var entryVals, exitVals []float64
	seenEntry, seenExit := false, false

	for _, p := range parts {
		kv := strings.SplitN(strings.TrimSpace(p), "=", 2)
		if len(kv) != 2 {
			return nil, fmt.Errorf("invalid component %q (expected key=val)", p)
		}
		key, val := strings.ToLower(strings.TrimSpace(kv[0])), strings.TrimSpace(kv[1])
		switch key {
		case "entry":
			if seenEntry {
				return nil, fmt.Errorf("duplicate entry parameter")
			}
			vals, err := parseRange(val)
			if err != nil {
				return nil, fmt.Errorf("invalid entry spec: %w", err)
			}
			entryVals = vals
			seenEntry = true
		case "exit":
			if seenExit {
				return nil, fmt.Errorf("duplicate exit parameter")
			}
			vals, err := parseRange(val)
			if err != nil {
				return nil, fmt.Errorf("invalid exit spec: %w", err)
			}
			exitVals = vals
			seenExit = true
		default:
			return nil, fmt.Errorf("unknown parameter %q (must be 'entry' or 'exit')", key)
		}
	}

	if !seenEntry || !seenExit {
		return nil, fmt.Errorf("both 'entry' and 'exit' specifications are required")
	}

	var grid []Params
	for _, entry := range entryVals {
		for _, exit := range exitVals {
			if exit < entry {
				grid = append(grid, Params{Entry: entry, Exit: exit})
			}
		}
	}

	if len(grid) == 0 {
		return nil, fmt.Errorf("grid has no valid points with exit < entry")
	}

	return grid, nil
}

func parseRange(s string) ([]float64, error) {
	parts := strings.Split(s, ":")
	if len(parts) != 3 {
		return nil, fmt.Errorf("expected start:stop:step, got %q", s)
	}
	start, err := strconv.ParseFloat(parts[0], 64)
	if err != nil {
		return nil, fmt.Errorf("invalid range start %q: %w", parts[0], err)
	}
	stop, err := strconv.ParseFloat(parts[1], 64)
	if err != nil {
		return nil, fmt.Errorf("invalid range stop %q: %w", parts[1], err)
	}
	step, err := strconv.ParseFloat(parts[2], 64)
	if err != nil {
		return nil, fmt.Errorf("invalid range step %q: %w", parts[2], err)
	}
	if step <= 0 {
		return nil, fmt.Errorf("step must be positive, got %v", step)
	}
	if start > stop {
		return nil, fmt.Errorf("start (%v) must not be greater than stop (%v)", start, stop)
	}

	var values []float64
	const eps = 1e-9
	for v := start; v <= stop+eps; v += step {
		rounded := math.Round(v*1e6) / 1e6
		values = append(values, rounded)
	}
	if len(values) == 0 {
		return nil, fmt.Errorf("range %q produced no values", s)
	}
	return values, nil
}

// DefaultGrid is every Entry/Exit pair from a small grid with Exit < Entry.
func DefaultGrid() []Params {
	var grid []Params
	for _, entry := range []float64{0.1, 0.2, 0.3, 0.4, 0.5} {
		for _, exit := range []float64{-0.5, -0.3, -0.2, -0.1, 0.0, 0.1} {
			if exit < entry {
				grid = append(grid, Params{Entry: entry, Exit: exit})
			}
		}
	}
	return grid
}

// DefaultWalkForward optimises on 120 days and trades the next 30.
func DefaultWalkForward() WalkForwardConfig {
	return WalkForwardConfig{TrainDays: 120, TestDays: 30, Grid: DefaultGrid()}
}

// Fold is one optimise-then-test step.
type Fold struct {
	TrainStart  int     `json:"train_start"`
	TestStart   int     `json:"test_start"`
	TestEnd     int     `json:"test_end"` // exclusive
	Chosen      Params  `json:"chosen"`
	TrainSharpe float64 `json:"train_sharpe"`
	TestReturn  float64 `json:"test_return_pct"`
}

// WalkForwardResult compares, over the same out-of-sample days, the optimised
// strategy with the fixed default thresholds and with buy-and-hold.
type WalkForwardResult struct {
	CoinID    string    `json:"coin_id"`
	OOSStart  int       `json:"oos_start"` // first out-of-sample day
	Folds     []Fold    `json:"folds"`
	Optimized Result    `json:"optimized"`
	Fixed     Result    `json:"fixed_default"`
	BuyHold   Result    `json:"buy_hold"`
	Prices    []float64 `json:"-"`
}

// WalkForward rolls a train/test split across the series. For each fold it picks
// the grid point with the best in-sample Sharpe ratio, then trades the following
// test window with it. Only test windows are scored, so every reported number is
// out-of-sample. Each test window starts flat and ends in cash.
func WalkForward(eng Predictor, cfg WalkForwardConfig) (WalkForwardResult, error) {
	prices := eng.Prices()
	n := len(prices)
	oosStart := WarmupPeriod + cfg.TrainDays
	if cfg.TrainDays <= 0 || cfg.TestDays <= 1 || len(cfg.Grid) == 0 {
		return WalkForwardResult{}, fmt.Errorf("invalid walk-forward config %+v", cfg)
	}
	if n < oosStart+cfg.TestDays {
		return WalkForwardResult{}, fmt.Errorf(
			"walk-forward needs at least %d prices (warmup %d + train %d + test %d), have %d",
			oosStart+cfg.TestDays, WarmupPeriod, cfg.TrainDays, cfg.TestDays, n)
	}

	scores := Scores(eng)
	res := WalkForwardResult{OOSStart: oosStart, Prices: prices}
	capital := InitialCapital
	var stitched simulation

	for testStart := oosStart; testStart < n-1; testStart += cfg.TestDays {
		testEnd := min(testStart+cfg.TestDays, n)
		trainStart := testStart - cfg.TrainDays

		best, bestSharpe := cfg.Grid[0], math.Inf(-1)
		for _, p := range cfg.Grid {
			train := simulate(prices, scores, p, trainStart, testStart, InitialCapital)
			if s := sharpeRatio(dailyReturns(train.equity)); s > bestSharpe {
				best, bestSharpe = p, s
			}
		}

		test := simulate(prices, scores, best, testStart, testEnd, capital)
		stitched.equity = append(stitched.equity, test.equity...)
		stitched.trades = append(stitched.trades, test.trades...)
		end := test.equity[len(test.equity)-1]
		res.Folds = append(res.Folds, Fold{
			TrainStart: trainStart, TestStart: testStart, TestEnd: testEnd,
			Chosen: best, TrainSharpe: bestSharpe, TestReturn: pctChange(capital, end),
		})
		capital = end
	}

	res.Optimized = summarize(stitched, InitialCapital)
	res.Fixed = summarize(simulate(prices, scores, DefaultParams, oosStart, n, InitialCapital), InitialCapital)
	res.BuyHold = summarize(buyAndHold(prices, oosStart, n), InitialCapital)

	bh := pctChange(prices[oosStart], prices[n-1])
	for _, r := range []*Result{&res.Optimized, &res.Fixed, &res.BuyHold} {
		r.BuyHoldReturn = bh
		r.DataPoints = n - oosStart
	}
	return res, nil
}

// buyAndHold buys at the close of day `from` and marks to market until `to`.
func buyAndHold(prices []float64, from, to int) simulation {
	units := InitialCapital * (1 - CommissionRate) / prices[from]
	sim := simulation{equity: make([]float64, 0, to-from)}
	for i := from; i < to; i++ {
		sim.equity = append(sim.equity, units*prices[i])
	}
	return sim
}
