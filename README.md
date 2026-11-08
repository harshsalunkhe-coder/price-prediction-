# HKEX / KRX Sentiment Research Pipeline

A Python project for collecting public investor discussion, recent Exa news, and hourly prices for five Hong Kong and five Korean equities. It aligns timestamps in UTC, builds sentiment and market features, evaluates one-bar-ahead returns chronologically, and writes a PDF briefing.

## Current status

The pipeline uses LIHKG, AASTOCKS, and ETNet for HKEX, and Naver Finance and public Telegram for KRX. Exa supplies dated news, yfinance supplies hourly OHLCV, and Gemini returns structured Cantonese/Korean/English sentiment labels. A local multilingual lexicon remains available only when explicitly selected with `SENTIMENT_PROVIDER=lexicon`. Synthetic post/news fallbacks have been removed; failed sources, scoring errors, and short histories produce `N/A` results instead of invented posts or forecasts.

Generated reports are produced by the current pipeline and carry a draft warning unless all ten configured tickers have at least two distinct ticker-relevant public sources aligned to the available bars, aligned Exa news, successful sentiment scoring, and out-of-sample test samples. No web server or Streamlit app is used.

The requested 10-calendar-day window is a small sample for forecasting. Any walk-forward metrics are exploratory and should not be presented as evidence of a dependable trading edge.

## Coverage and source configuration

The tickers and public-source URLs are in [`config/tickers.yaml`](config/tickers.yaml).

| Exchange | Public sources |
| --- | --- | --- |
| HKEX | [LIHKG](https://lihkg.com/) through the community Actor [`claude_code_reviewer/lihkg-scraper-en`](https://apify.com/claude_code_reviewer/lihkg-scraper-en); [AASTOCKS ticker news](https://www.aastocks.com/tc/stocks/news/aafn/); [ETNet ticker commentary](https://www.etnet.com.hk/www/tc/stocks/realtime/quote_commentary.php) |
| KRX | [Naver Finance stock discussion](https://stock.naver.com/domestic/stock/005930/discussion), read from its public discussion feed; [Korean stocks Telegram channel](https://t.me/s/FastStockNews) through [`innolabs/telegram-scraper`](https://apify.com/innolabs/telegram-scraper) |

Apify collects LIHKG and public Telegram posts. AASTOCKS news cards and ETNet ticker-commentary articles are fetched from their public ticker pages; these are market-news/commentary signals, not equivalent to retail-forum posts. AASTOCKS reader-vote totals are excluded because the public page does not show when votes were cast. Naver's legacy board URL now redirects to Npay Securities; its public read-only feed is fetched directly because generic Apify browser actors request full access to the entire Apify account. The Naver request is limited to the selected ticker and at most 50 recent records. No Naver login, cookies, user profile, or author name is collected. Telegram collection is restricted to public channels. The code does not log in to private groups. All source timestamps are normalized to UTC before bar alignment.

## Setup

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and enter your Apify, Exa, and Gemini keys locally. Keep that file private; do not commit it or paste keys into chat. The pipeline uses the free `yfinance` package for price history. Apify Actor usage, Exa searches, and Gemini requests can use account credits or incur charges; check each provider's current usage and spending limits before collecting.

## First connection check

Check one Hong Kong ticker's live data sources and Gemini scoring before running all ten:

```powershell
python run_pipeline.py --check-sources 0700.HK
```

This fetches a bounded sample from market data, configured public sources, and Exa, then sends the collected text through the configured sentiment scorer. If those sources return no text, it uses a clearly labelled one-item Gemini connection probe that is not added to the report. `--check-market TICKER` is available for a price-only check.

## Full pipeline

```powershell
python run_pipeline.py
```

Expected outputs after a successful run:

- `outputs/sentiment_report.pdf` — executive summary, exchange rankings, selected ticker charts, and validation status.
- `outputs/summary.png` — image preview of the current run.
- `outputs/predictions.csv` — one row per ticker, including data and validation status.
- `outputs/charts/` — price, support/resistance, and sentiment plots.
- `data/raw/` — versioned, live-source response caches for reproducibility.

An unavailable source is recorded as missing; it is not replaced with generated content. A ticker forecast is withheld unless at least two distinct ticker-relevant configured sources align to the price history, dated Exa news aligns to the history, Gemini scoring succeeds, and the history meets the expanding-window validation minimum.

To re-score the last saved public collection and rebuild the CSV, charts, and PDF without calling yfinance, Exa, or social APIs again, run:

```powershell
python run_pipeline.py --use-cached-data
```

This replay mode requires the prior `outputs/predictions.csv` and versioned data caches to be present and less than 24 hours old. Gemini may still be called for records not yet present in its sentiment cache. To refresh only the public Hong Kong commentary pages while reusing the other cached inputs, run:

```powershell
python run_pipeline.py --use-cached-data --refresh-hk-commentary
```

This makes public read-only requests to AASTOCKS and ETNet for the five HKEX tickers. It does not call Apify, Exa, or yfinance during that refresh.

## Method notes

- Price, social, and news timestamps are converted to UTC. An observation is assigned to the first available bar at or after its publication time; observations later than the newest price bar are excluded from historical features.
- Support and resistance features are recalculated from data available up through each bar, to prevent future price extrema from entering earlier model rows.
- Gemini structured output classifies each post and article with polarity, confidence, relevance, spam flag, and a short rationale; prompts identify whether a record is a forum post, public news item, or analyst commentary. Results are cached by model, context, and text to avoid repeating identical requests. The optional lexicon scorer is transparent and selected explicitly.
- Sentiment features include raw and engagement/confidence/relevance-weighted polarity, dispersion, volume, and news mood. The report includes hit rate, return MAE/RMSE, information coefficient, and a per-bar Sharpe diagnostic where enough test data exists.
- The validator uses an expanding training window and predicts each test bar once in time order. Results are compared with a zero-return baseline for error and a training-period majority-direction baseline for directional accuracy.
- No accuracy, alpha, or Sharpe figure is hard-coded. The report shows `N/A` where the run did not produce an actual metric.

## Reproducibility checklist

Before sharing a report, confirm that each ticker has fresh price bars, at least two distinct relevant public sources with timestamps inside the price history, Exa results with source URLs, and a nonzero out-of-sample test count. Save the generated timestamp and commit the code/configuration, not `.env` or raw personal data. The Actor IDs and public source URLs are listed above and in `config/tickers.yaml`.
