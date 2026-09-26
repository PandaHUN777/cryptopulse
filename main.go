package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/wyplerszymon0-lab/cryptopulse/internal/api"
	"github.com/wyplerszymon0-lab/cryptopulse/internal/backtest"
	"github.com/wyplerszymon0-lab/cryptopulse/internal/predictor"
	"github.com/wyplerszymon0-lab/cryptopulse/internal/report"
)

const version = "2.1.0"

func main() {
	coinsStr := flag.String("coins", "bitcoin,ethereum,solana", "comma-separated CoinGecko coin IDs")
	days := flag.Int("days", 90, "historical days to fetch (30–365)")
	asJSON := flag.Bool("json", false, "output as JSON (pipe-friendly)")
	verbose := flag.Bool("verbose", false, "enable debug logging")
	workers := flag.Int("workers", 3, "concurrent fetch workers")
	doBacktest := flag.Bool("backtest", false, "backtest the default signal thresholds over the whole period")
	doOptimize := flag.Bool("optimize", false, "walk-forward optimisation: tune thresholds in-sample, score out-of-sample")
	trainDays := flag.Int("train-days", 120, "walk-forward training window (days)")
	testDays := flag.Int("test-days", 30, "walk-forward test window (days)")
	gridStr := flag.String("grid", "", "custom threshold grid (e.g. entry=0.1:0.5:0.1,exit=-0.5:0.1:0.1)")
	exportPath := flag.String("export", "", "write analysis + walk-forward results with price history to this JSON file")
	versionFlag := flag.Bool("version", false, "print version and exit")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, `CryptoPulse v%s — Real-Time Technical Analysis & Backtesting Engine

Usage:
  predictor [flags]

Flags:
`, version)
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, `
Examples:
  predictor                                    Analyse BTC, ETH, SOL (90 days)
  predictor --coins bitcoin,ethereum           Custom coin list
  predictor --backtest --days 365              1-year backtest of the default thresholds
  predictor --optimize --days 365              walk-forward optimisation (out-of-sample)
  predictor --json | jq '.[] | .signal'        JSON output for scripting
  predictor --coins cardano --days 60          60-day analysis

Any CoinGecko coin ID is valid: bitcoin, ethereum, solana, cardano, dogecoin, …
`)
	}

	flag.Parse()

	if *versionFlag {
		fmt.Printf("CryptoPulse v%s\n", version)
		os.Exit(0)
	}

	level := slog.LevelWarn
	if *verbose {
		level = slog.LevelDebug
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level})))

	if *days < 30 || *days > 365 {
		fmt.Fprintf(os.Stderr, "error: --days must be between 30 and 365\n")
		os.Exit(1)
	}

	coins := parseCoins(*coinsStr)
	if len(coins) == 0 {
		fmt.Fprintf(os.Stderr, "error: --coins must contain at least one ID\n")
		os.Exit(1)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	client := api.NewCoinGeckoClient()
	wf := backtest.DefaultWalkForward()
	wf.TrainDays, wf.TestDays = *trainDays, *testDays
	if *gridStr != "" {
		grid, err := backtest.ParseGrid(*gridStr)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		wf.Grid = grid
	}

	if (*doOptimize || *exportPath != "") && *days < backtest.WarmupPeriod+wf.TrainDays+wf.TestDays {
		fmt.Fprintf(os.Stderr, "error: walk-forward needs --days >= %d (warmup %d + train %d + test %d)\n",
			backtest.WarmupPeriod+wf.TrainDays+wf.TestDays, backtest.WarmupPeriod, wf.TrainDays, wf.TestDays)
		os.Exit(1)
	}

	switch {
	case *exportPath != "":
		runExport(ctx, client, coins, *days, *workers, wf, *exportPath)
	case *doOptimize:
		runOptimize(ctx, client, coins, *days, *workers, wf, *asJSON)
	case *doBacktest:
		runBacktest(ctx, client, coins, *days, *asJSON)
	default:
		runAnalysis(ctx, client, coins, *days, *workers, *asJSON)
	}
}

// coinResult carries the outcome of a single fetch.
type coinResult struct {
	coinID string
	prices []float64
	times  []int64
	err    error
}

// fetchAll fetches price series for all coins concurrently, bounded by workers.
func fetchAll(ctx context.Context, client *api.CoinGeckoClient, coins []string, days, workers int) []coinResult {
	results := make([]coinResult, len(coins))
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup

	for i, coin := range coins {
		wg.Add(1)
		i, coin := i, coin
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			slog.Debug("fetching", "coin", coin, "days", days)
			series, err := client.FetchSeries(ctx, coin, days)
			results[i] = coinResult{coinID: coin, prices: series.Prices, times: series.Times, err: err}
		}()
	}

	wg.Wait()
	return results
}

func runAnalysis(ctx context.Context, client *api.CoinGeckoClient, coins []string, days, workers int, asJSON bool) {
	if !asJSON {
		report.PrintHeader(version)
	}

	start := time.Now()
	datasets := fetchAll(ctx, client, coins, days, workers)
	elapsed := time.Since(start)

	if !asJSON {
		fmt.Printf("Fetched %d coin(s) in %s\n", len(coins), elapsed.Round(time.Millisecond))
	}

	var results []predictor.Result
	for _, d := range datasets {
		if d.err != nil {
			slog.Error("fetch failed", "coin", d.coinID, "err", d.err)
			fmt.Fprintf(os.Stderr, "warning: skipping %s: %v\n", d.coinID, d.err)
			continue
		}
		engine := predictor.NewEngine(d.prices)
		result, err := engine.Predict()
		if err != nil {
			slog.Error("prediction failed", "coin", d.coinID, "err", err)
			fmt.Fprintf(os.Stderr, "warning: skipping %s: %v\n", d.coinID, err)
			continue
		}
		result.CoinID = d.coinID
		results = append(results, result)
	}

	if asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(results); err != nil {
			fmt.Fprintf(os.Stderr, "json encode: %v\n", err)
			os.Exit(1)
		}
		return
	}

	for _, r := range results {
		report.PrintResult(r)
	}
}

func runBacktest(ctx context.Context, client *api.CoinGeckoClient, coins []string, days int, asJSON bool) {
	if !asJSON {
		report.PrintHeader(version)
		fmt.Printf("Running walk-forward backtest: %d coin(s) × %d days…\n", len(coins), days)
	}

	datasets := fetchAll(ctx, client, coins, days, len(coins))

	var btResults []backtest.Result
	for _, d := range datasets {
		if d.err != nil {
			fmt.Fprintf(os.Stderr, "warning: skipping %s: %v\n", d.coinID, d.err)
			continue
		}
		engine := predictor.NewEngine(d.prices)
		bt, err := backtest.Run(engine)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: backtest failed for %s: %v\n", d.coinID, err)
			continue
		}
		bt.CoinID = d.coinID
		btResults = append(btResults, bt)
	}

	if asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(btResults); err != nil {
			fmt.Fprintf(os.Stderr, "json encode: %v\n", err)
			os.Exit(1)
		}
		return
	}

	report.PrintBacktestSummary(btResults)
}

func runOptimize(ctx context.Context, client *api.CoinGeckoClient, coins []string, days, workers int,
	cfg backtest.WalkForwardConfig, asJSON bool) {
	if !asJSON {
		report.PrintHeader(version)
		fmt.Printf("Walk-forward optimisation: %d coin(s) × %d days, train %d / test %d, %d parameter sets…\n",
			len(coins), days, cfg.TrainDays, cfg.TestDays, len(cfg.Grid))
	}

	var results []backtest.WalkForwardResult
	for _, d := range fetchAll(ctx, client, coins, days, workers) {
		if d.err != nil {
			fmt.Fprintf(os.Stderr, "warning: skipping %s: %v\n", d.coinID, d.err)
			continue
		}
		res, err := backtest.WalkForward(predictor.NewEngine(d.prices), cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: walk-forward failed for %s: %v\n", d.coinID, err)
			continue
		}
		res.CoinID = d.coinID
		results = append(results, res)
	}

	if asJSON {
		writeJSON(os.Stdout, results)
		return
	}
	report.PrintWalkForward(results)
}

// exportCoin is one coin's entry in the --export file consumed by the web dashboard.
type exportCoin struct {
	CoinID      string                     `json:"coin_id"`
	Times       []int64                    `json:"times"`
	Prices      []float64                  `json:"prices"`
	Analysis    predictor.Result           `json:"analysis"`
	WalkForward backtest.WalkForwardResult `json:"walk_forward"`
}

type exportFile struct {
	Version     string       `json:"version"`
	GeneratedAt time.Time    `json:"generated_at"`
	Days        int          `json:"days"`
	TrainDays   int          `json:"train_days"`
	TestDays    int          `json:"test_days"`
	GridSize    int          `json:"grid_size"`
	Commission  float64      `json:"commission"`
	Coins       []exportCoin `json:"coins"`
}

func runExport(ctx context.Context, client *api.CoinGeckoClient, coins []string, days, workers int,
	cfg backtest.WalkForwardConfig, path string) {
	out := exportFile{
		Version: version, GeneratedAt: time.Now().UTC(), Days: days,
		TrainDays: cfg.TrainDays, TestDays: cfg.TestDays, GridSize: len(cfg.Grid),
		Commission: backtest.CommissionRate,
	}
	for _, d := range fetchAll(ctx, client, coins, days, workers) {
		if d.err != nil {
			fmt.Fprintf(os.Stderr, "warning: skipping %s: %v\n", d.coinID, d.err)
			continue
		}
		engine := predictor.NewEngine(d.prices)
		analysis, err := engine.Predict()
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: skipping %s: %v\n", d.coinID, err)
			continue
		}
		analysis.CoinID = d.coinID
		wfRes, err := backtest.WalkForward(engine, cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: skipping %s: %v\n", d.coinID, err)
			continue
		}
		wfRes.CoinID = d.coinID
		out.Coins = append(out.Coins, exportCoin{
			CoinID: d.coinID, Times: d.times, Prices: d.prices, Analysis: analysis, WalkForward: wfRes,
		})
	}
	if len(out.Coins) == 0 {
		fmt.Fprintln(os.Stderr, "error: no coin could be exported")
		os.Exit(1)
	}

	f, err := os.Create(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()
	writeJSON(f, out)
	fmt.Printf("wrote %d coin(s) to %s\n", len(out.Coins), path)
}

func writeJSON(w io.Writer, v any) {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		fmt.Fprintf(os.Stderr, "json encode: %v\n", err)
		os.Exit(1)
	}
}

func parseCoins(s string) []string {
	var coins []string
	for _, c := range strings.Split(s, ",") {
		c = strings.TrimSpace(strings.ToLower(c))
		if c != "" {
			coins = append(coins, c)
		}
	}
	return coins
}
