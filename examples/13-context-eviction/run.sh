#!/usr/bin/env bash
# 13 Context Window Eviction Tester — flood memory with noise, probe what survived.
set -euo pipefail
cd "$(dirname "$0")"
E="uv run --no-sync evalkit context-eviction"

echo "== sweep: 5 strategies x 4 noise levels, 3 paired trials, 400-token budget =="
$E run --strategies full,fifo,window,summary,retrieval --noise 0,50,200,800 --trials 3 --seed 0
echo
echo "== tight budget (120 tokens), 8 facts, 30% hard noise =="
$E run --strategies window,summary,retrieval --noise 50,800 --budget 120 --facts 8 --hard-noise 0.3
echo
echo "== per-probe view: sliding window on the checked-in 120-noise scenario =="
$E inspect --scenario scenario-noise120.json --strategy window --budget 200
echo
echo "== same scenario, retrieval memory =="
$E inspect --scenario scenario-noise120.json --strategy retrieval --budget 200 | tail -n 10
