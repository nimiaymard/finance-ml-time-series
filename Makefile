.PHONY: setup data arima lstm backtest powerbi test

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

data:
	python scripts/fetch_data.py --tickers AAPL MSFT --start 2018-01-01 --end 2025-10-01

arima:
	python scripts/train_arima.py --ticker AAPL --target Close --order 5 1 0

lstm:
	python scripts/train_lstm.py --ticker AAPL --target Close --seq-len 60 --epochs 5

backtest:
	python scripts/backtest.py --ticker AAPL --model arima --target Close

powerbi:
	python scripts/export_powerbi.py --tickers AAPL MSFT --target Close

test:
	pytest -q
