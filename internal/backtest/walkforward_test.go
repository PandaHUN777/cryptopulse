package backtest

import (
	"math"
	"testing"

	"github.com/wyplerszymon0-lab/cryptopulse/internal/predictor"
)

// wavePrices is a trend with a slow oscillation, so indicators produce both
// buy and sell signals and the grid has something to choose between.
func wavePrices(n int) []float64 {
	p := make([]float64, n)
	for i := range p {
		x := float64(i)
		p[i] = 100 + 0.1*x + 12*math.Sin(x/9) + 3*math.Sin(x/2.3)
	}
	return p
}

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestSimulate_FillsNextDayWithCommission(t *testing.T) {
	prices := []float64{100, 110, 120, 130, 140}
	nan := math.NaN()
	scores := []float64{1, nan, -1, nan, nan} // buy on day 0, sell on day 2

	sim := simulate(prices, scores, DefaultParams, 0, len(prices), 1000)

	if len(sim.trades) != 1 {
		t.Fatalf("trades = %d, want 1", len(sim.trades))
	}
	tr := sim.trades[0]
	if tr.EntryIndex != 1 || tr.ExitIndex != 3 {
		t.Errorf("entry/exit = %d/%d, want 1/3 (next-day fills)", tr.EntryIndex, tr.ExitIndex)
	}
	units := 1000 * (1 - CommissionRate) / 110
	want := units * 130 * (1 - CommissionRate)
	if got := sim.equity[len(sim.equity)-1]; !approx(got, want) {
		t.Errorf("final equity = %v, want %v", got, want)
	}
	if !approx(sim.equity[2], units*120) {
		t.Errorf("day-2 equity = %v, want mark-to-market %v", sim.equity[2], units*120)
	}
}

func TestSimulate_ClosesOpenPositionAtWindowEnd(t *testing.T) {
	prices := []float64{100, 100, 200, 300}
	scores := []float64{1, math.NaN(), math.NaN(), math.NaN()}
	sim := simulate(prices, scores, DefaultParams, 0, 3, 1000) // window ends on day 2

	if len(sim.trades) != 1 || sim.trades[0].ExitIndex != 2 {
		t.Fatalf("want one trade closed on day 2, got %+v", sim.trades)
	}
	if len(sim.equity) != 3 {
		t.Errorf("equity length = %d, want 3", len(sim.equity))
	}
}

func TestSharpe_AnnualisedWith365Days(t *testing.T) {
	returns := []float64{0.01, -0.02, 0.03, 0.0}
	m := mean(returns)
	want := m / stdDev(returns, m) * math.Sqrt(365)
	if got := sharpeRatio(returns); !approx(got, want) {
		t.Errorf("sharpe = %v, want %v", got, want)
	}
}

func TestSortino_DownsideDeviationOverAllPeriods(t *testing.T) {
	returns := []float64{0.02, -0.01, 0.03, -0.03}
	downside := math.Sqrt((0.01*0.01 + 0.03*0.03) / 4) // divide by all 4 periods, not 2
	want := mean(returns) / downside * math.Sqrt(365)
	if got := sortinoRatio(returns); !approx(got, want) {
		t.Errorf("sortino = %v, want %v", got, want)
	}
}

func TestWalkForward_FoldsTileTheOutOfSamplePeriod(t *testing.T) {
	prices := wavePrices(300)
	cfg := WalkForwardConfig{TrainDays: 90, TestDays: 30, Grid: DefaultGrid()}
	res, err := WalkForward(predictor.NewEngine(prices), cfg)
	if err != nil {
		t.Fatal(err)
	}

	next := WarmupPeriod + cfg.TrainDays
	for _, f := range res.Folds {
		if f.TestStart != next || f.TrainStart != f.TestStart-cfg.TrainDays {
			t.Errorf("fold %+v does not continue at %d", f, next)
		}
		next = f.TestEnd
	}
	if next != len(prices) {
		t.Errorf("folds end at %d, want %d", next, len(prices))
	}
	oosDays := len(prices) - res.OOSStart
	for name, r := range map[string]Result{"optimized": res.Optimized, "fixed": res.Fixed, "buy_hold": res.BuyHold} {
		if len(r.Equity) != oosDays {
			t.Errorf("%s equity length = %d, want %d", name, len(r.Equity), oosDays)
		}
	}
	if res.Optimized.TotalTrades == 0 {
		t.Error("expected the oscillating series to produce trades")
	}
}

func TestWalkForward_FoldReturnsCompoundToTotal(t *testing.T) {
	res, err := WalkForward(predictor.NewEngine(wavePrices(300)), DefaultWalkForward())
	if err != nil {
		t.Fatal(err)
	}
	growth := 1.0
	for _, f := range res.Folds {
		growth *= 1 + f.TestReturn/100
	}
	if got := (growth - 1) * 100; math.Abs(got-res.Optimized.TotalReturn) > 1e-6 {
		t.Errorf("compounded folds = %.6f%%, total = %.6f%%", got, res.Optimized.TotalReturn)
	}
}

func TestWalkForward_NoLookahead(t *testing.T) {
	// Changing prices after a fold's test window must not change the parameters
	// chosen for that fold or its out-of-sample result.
	prices := wavePrices(300)
	cfg := DefaultWalkForward()
	base, err := WalkForward(predictor.NewEngine(prices), cfg)
	if err != nil {
		t.Fatal(err)
	}
	first := base.Folds[0]

	altered := append([]float64(nil), prices...)
	for i := first.TestEnd; i < len(altered); i++ {
		altered[i] *= 3
	}
	alt, err := WalkForward(predictor.NewEngine(altered), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if alt.Folds[0].Chosen != first.Chosen || !approx(alt.Folds[0].TestReturn, first.TestReturn) {
		t.Errorf("fold 0 changed when only future prices changed: %+v vs %+v", alt.Folds[0], first)
	}
}

func TestWalkForward_RejectsShortSeries(t *testing.T) {
	_, err := WalkForward(predictor.NewEngine(wavePrices(150)), DefaultWalkForward())
	if err == nil {
		t.Error("expected an error when warmup + train + test exceeds the series")
	}
}

func TestDefaultGrid_ExitBelowEntry(t *testing.T) {
	grid := DefaultGrid()
	if len(grid) == 0 {
		t.Fatal("empty grid")
	}
	found := false
	for _, p := range grid {
		if p.Exit >= p.Entry {
			t.Errorf("grid point %+v has exit >= entry", p)
		}
		found = found || p == DefaultParams
	}
	if !found {
		t.Error("grid should contain the default parameters")
	}
}

func TestParseGrid_Valid(t *testing.T) {
	spec := "entry=0.1:0.5:0.1,exit=-0.5:0.1:0.1"
	grid, err := ParseGrid(spec)
	if err != nil {
		t.Fatalf("unexpected error parsing valid grid: %v", err)
	}
	if len(grid) == 0 {
		t.Fatal("expected non-empty grid")
	}
	for _, p := range grid {
		if p.Exit >= p.Entry {
			t.Errorf("grid point %+v has exit >= entry", p)
		}
		if p.Entry < 0.1 || p.Entry > 0.5 {
			t.Errorf("entry out of bounds: %v", p.Entry)
		}
		if p.Exit < -0.5 || p.Exit > 0.1 {
			t.Errorf("exit out of bounds: %v", p.Exit)
		}
	}

	// Verify order independence
	revSpec := "exit=-0.5:0.1:0.1,entry=0.1:0.5:0.1"
	revGrid, err := ParseGrid(revSpec)
	if err != nil {
		t.Fatalf("unexpected error parsing reversed spec: %v", err)
	}
	if len(grid) != len(revGrid) {
		t.Errorf("len(grid) = %d, len(revGrid) = %d", len(grid), len(revGrid))
	}
}

func TestParseGrid_Invalid(t *testing.T) {
	tests := []struct {
		name string
		spec string
	}{
		{"empty", ""},
		{"single component", "entry=0.1:0.5:0.1"},
		{"missing entry", "exit=-0.5:0.1:0.1,foo=1:2:1"},
		{"missing exit", "entry=0.1:0.5:0.1,foo=1:2:1"},
		{"unknown key", "entry=0.1:0.5:0.1,other=1:2:1"},
		{"duplicate entry", "entry=0.1:0.5:0.1,entry=0.2:0.4:0.1"},
		{"duplicate exit", "exit=-0.5:0.1:0.1,exit=-0.2:0.0:0.1"},
		{"not three parts", "entry=0.1:0.5,exit=-0.5:0.1:0.1"},
		{"non-numeric start", "entry=abc:0.5:0.1,exit=-0.5:0.1:0.1"},
		{"non-numeric stop", "entry=0.1:xyz:0.1,exit=-0.5:0.1:0.1"},
		{"non-numeric step", "entry=0.1:0.5:step,exit=-0.5:0.1:0.1"},
		{"zero step", "entry=0.1:0.5:0,exit=-0.5:0.1:0.1"},
		{"negative step", "entry=0.1:0.5:-0.1,exit=-0.5:0.1:0.1"},
		{"start greater than stop", "entry=0.5:0.1:0.1,exit=-0.5:0.1:0.1"},
		{"no points with exit < entry", "entry=0.1:0.2:0.1,exit=0.3:0.4:0.1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseGrid(tt.spec)
			if err == nil {
				t.Errorf("expected error for spec %q, got nil", tt.spec)
			}
		})
	}
}
