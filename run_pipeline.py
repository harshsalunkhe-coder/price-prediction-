"""
Master Pipeline Runner: Hang Seng & KRX Sentiment Price Prediction Agent.
Executes the end-to-end 6-stage alternative quantitative data pipeline:
Collect -> Clean/Align -> Sentiment -> Features + S/R Levels -> Predict n+1 -> Visual PDF Report
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import json
import logging
import os
import sys
import yaml

from dotenv import load_dotenv
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
(PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)

from src.collect import MarketDataCollector, ExaNewsCollector, ApifySocialCollector
from src.process import DataAligner
from src.sentiment import SentimentScorer
from src.features import FeatureBuilder
from src.model import ModelValidator
from src.report import ChartGenerator, PDFReportBuilder, SummaryImageBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "outputs" / "pipeline.log", mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
# Reduce verbosity for third party loggers in console
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logger = logging.getLogger("PipelineRunner")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(highlight=False)


def print_banner():
    """Prints the Bloomberg-terminal style CLI banner."""
    banner_text = (
        "[bold cyan]CROWDWISDOM TRADING[/bold cyan] — [bold white]Alternative Sentiment & n+1 Price Prediction Agent[/bold white]\n"
        "[dim]Hang Seng (HKEX) & Korea (KRX) Multi-Asset Quantitative Pipeline | 10-Day Walk-Forward Engine[/dim]"
    )
    console.print(Panel(banner_text, border_style="cyan", padding=(1, 2)))


def main(argv=None):
    print_banner()

    # Load environment variables
    load_dotenv(PROJECT_ROOT / ".env")

    # Load configuration
    config_path = PROJECT_ROOT / "config" / "tickers.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    settings = config.get("settings", {})
    lookback_days = settings.get("lookback_days", 10)
    bar_interval = settings.get("bar_interval", "1h")
    output_dir = PROJECT_ROOT / settings.get("output_dir", "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run the HKEX/KRX sentiment research pipeline.")
    parser.add_argument(
        "--check-market",
        metavar="TICKER",
        help="Make a fresh yfinance request for one ticker and print the returned bar count and latest close.",
    )
    parser.add_argument(
        "--check-sources",
        metavar="TICKER",
        help="Run a bounded one-ticker check of yfinance, both configured Apify sources, and Exa.",
    )
    parser.add_argument(
        "--use-cached-data",
        action="store_true",
        help="Re-score and rebuild reports from the last saved public collection without calling market, Exa, or social APIs.",
    )
    parser.add_argument(
        "--refresh-hk-commentary",
        action="store_true",
        help="With --use-cached-data, refresh only public AASTOCKS and ETNet inputs for HKEX tickers.",
    )
    args = parser.parse_args(argv)

    if args.use_cached_data and (args.check_market or args.check_sources):
        parser.error("--use-cached-data cannot be combined with a live connection check")
    if args.refresh_hk_commentary and not args.use_cached_data:
        parser.error("--refresh-hk-commentary requires --use-cached-data")

    cached_run_metadata = {}
    if args.use_cached_data:
        predictions_path = output_dir / "predictions.csv"
        if not predictions_path.exists():
            parser.error("--use-cached-data requires an existing outputs/predictions.csv from a prior run")
        prior_predictions = pd.read_csv(predictions_path)
        cached_run_metadata = {
            str(row["ticker"]): row.to_dict()
            for _, row in prior_predictions.iterrows()
            if row.get("ticker")
        }
        raw_dir = PROJECT_ROOT / "data" / "raw"
        cache_issues = []
        for exchange_info in config.get("exchanges", {}).values():
            for ticker_info in exchange_info.get("tickers", []):
                symbol = str(ticker_info.get("symbol", ""))
                key = symbol.replace(".", "_")
                price_files = [
                    raw_dir / f"prices_live_v1_{key}_{bar_interval}.parquet",
                    raw_dir / f"prices_live_v1_{key}_{bar_interval}.csv",
                ]
                price_path = next((path for path in price_files if path.exists()), None)
                required_paths = [
                    raw_dir / f"social_live_v1_{key}.json",
                    raw_dir / f"news_live_v1_{key}.json",
                ]
                if symbol not in cached_run_metadata:
                    cache_issues.append(f"{symbol}: missing prior prediction metadata")
                for input_path in [price_path, *required_paths]:
                    if input_path is None or not input_path.exists():
                        cache_issues.append(f"{symbol}: missing cached input")
                    elif datetime.now(timezone.utc).timestamp() - input_path.stat().st_mtime > timedelta(hours=24).total_seconds():
                        cache_issues.append(f"{symbol}: stale cached input ({input_path.name})")
        if cache_issues:
            parser.error(
                "Cached replay stopped before changing outputs because inputs are incomplete or stale: "
                + "; ".join(cache_issues[:12])
            )

    # Initialize Modules
    collector_market = MarketDataCollector(cache_dir=str(PROJECT_ROOT / "data" / "raw"))
    if args.check_market:
        bars = collector_market.fetch_bars(
            args.check_market,
            lookback_days=lookback_days,
            interval=bar_interval,
            force_refresh=True,
        )
        latest_time = bars.index[-1].strftime("%Y-%m-%d %H:%M UTC")
        console.print(
            f"[bold green]Market data connection succeeded[/bold green] — "
            f"{args.check_market}: {len(bars)} bars, latest close {bars['Close'].iloc[-1]:.4f} at {latest_time}"
        )
        return

    collector_news = ExaNewsCollector(cache_dir=str(PROJECT_ROOT / "data" / "raw"))
    collector_social = ApifySocialCollector(cache_dir=str(PROJECT_ROOT / "data" / "raw"))
    if args.check_sources:
        match = next(
            (
                (exchange_code, exchange_info, ticker_info)
                for exchange_code, exchange_info in config.get("exchanges", {}).items()
                for ticker_info in exchange_info.get("tickers", [])
                if ticker_info.get("symbol") == args.check_sources
            ),
            None,
        )
        if match is None:
            parser.error(f"Unknown configured ticker: {args.check_sources}")

        exchange_code, exchange_info, ticker_info = match
        source_definitions = []
        for configured_source in exchange_info.get("social_sources", []):
            source = dict(configured_source)
            actor_env = source.get("actor_env")
            if actor_env and os.getenv(actor_env):
                source["actor_id"] = os.environ[actor_env]
            if source.get("type") == "naver_forum":
                source["pages"] = min(int(source.get("pages", 2)), 2)
            source_definitions.append(source)

        missing_credentials = [
            name for name, value in (
                ("APIFY_API_TOKEN", collector_social.api_token),
                ("EXA_API_KEY", collector_news.api_key),
            ) if not value or value.startswith("add-your-") or value.startswith("your_")
        ]
        if os.getenv("SENTIMENT_PROVIDER", "gemini").strip().lower() == "gemini":
            gemini_key = os.getenv("GEMINI_API_KEY")
            if not gemini_key or gemini_key.startswith("add-your-") or gemini_key.startswith("your_"):
                missing_credentials.append("GEMINI_API_KEY")
        if missing_credentials:
            console.print(
                "[bold yellow]Source connection check skipped[/bold yellow] — add "
                + ", ".join(missing_credentials)
                + f" to {PROJECT_ROOT / '.env'} and rerun."
            )
            return

        ticker = ticker_info["symbol"]
        price_bars = collector_market.fetch_bars(
            ticker,
            lookback_days=lookback_days,
            interval=bar_interval,
            force_refresh=True,
        )
        news_results = collector_news.fetch_news(
            ticker,
            ticker_info["name"],
            ticker_info.get("keywords", [ticker]),
            lookback_days=lookback_days,
            max_results=3,
        )
        social_results = collector_social.fetch_social_posts(
            ticker,
            ticker_info["name"],
            exchange_code,
            ticker_info.get("keywords", [ticker]),
            ticker_info.get("telegram_tags", []),
            lookback_days=lookback_days,
            max_posts=5,
            social_sources=source_definitions,
        )
        console.print(
            f"[bold green]Market[/bold green]: {len(price_bars)} bars; latest close "
            f"{price_bars['Close'].iloc[-1]:.4f} at {price_bars.index[-1].strftime('%Y-%m-%d %H:%M UTC')}"
        )
        console.print(f"[bold]Exa[/bold]: {collector_news.last_status}, {len(news_results)} dated item(s)")
        for status in collector_social.last_source_statuses:
            console.print(
                f"[bold]{status.get('source')}[/bold]: {status.get('status')}, "
                f"{status.get('items', 0)} in-window record(s)"
            )
        scorer = SentimentScorer()
        smoke_posts = social_results
        if not smoke_posts and not news_results and scorer.provider == "gemini":
            smoke_text = (
                "삼성전자 실적 호조로 주가 상승을 기대합니다."
                if exchange_code == "KRX"
                else "騰訊業績勝預期，睇好後市。"
            )
            smoke_posts = [{"post_id": "gemini-connection-smoke", "text": smoke_text}]
        try:
            scored_check_posts, scored_check_news = scorer.score_records(
                smoke_posts, news_results, ticker=ticker, asset_name=ticker_info["name"], exchange=exchange_code
            )
            check_count = len(scored_check_posts) + len(scored_check_news)
            suffix = " (connection probe only)" if smoke_posts and not social_results else ""
            console.print(
                f"[bold]Sentiment[/bold]: {scorer.provider}, {scorer.last_status}, "
                f"{check_count} record(s) classified{suffix}"
            )
        except Exception as exc:
            console.print(
                f"[bold red]Sentiment check failed[/bold red] ({type(exc).__name__}); "
                "predictions will be withheld."
            )
        return

    aligner = DataAligner()
    scorer = SentimentScorer()
    feature_builder = FeatureBuilder()
    validator = ModelValidator(test_size_ratio=config["model"].get("test_size_ratio", 0.25))
    chart_gen = ChartGenerator(output_dir=str(output_dir / "charts"))
    pdf_builder = PDFReportBuilder(output_pdf_path=str(output_dir / "sentiment_report.pdf"))
    summary_img_builder = SummaryImageBuilder(output_path=str(output_dir / "summary.png"))

    exchanges = config.get("exchanges", {})
    all_summaries = []
    hk_summaries = []
    kr_summaries = []
    chart_paths = {}
    all_metrics = []
    feature_importances_accum = {}
    prediction_records = []

    # Process all tickers
    total_tickers = sum(len(ex["tickers"]) for ex in exchanges.values())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Running Ingestion, NLP & Predictive Agents...", total=total_tickers)

        for ex_code, ex_info in exchanges.items():
            ex_name = ex_info["name"]
            tickers = ex_info["tickers"]

            for t_info in tickers:
                symbol = t_info["symbol"]
                name = t_info["name"]
                keywords = t_info.get("keywords", [symbol])
                tg_tags = t_info.get("telegram_tags", [f"#{symbol.split('.')[0]}"])
                social_sources = []
                for configured_source in ex_info.get("social_sources", []):
                    source = dict(configured_source)
                    actor_env = source.get("actor_env")
                    if actor_env and os.getenv(actor_env):
                        source["actor_id"] = os.environ[actor_env]
                    social_sources.append(source)

                progress.update(task, description=f"[cyan]Processing {symbol} ({name})...")

                try:
                    if args.use_cached_data:
                        # Reuse the exact public inputs and collection statuses
                        # from the prior run, keeping scoring/report repairs
                        # reproducible without another paid collection pass.
                        key = symbol.replace(".", "_")
                        raw_dir = PROJECT_ROOT / "data" / "raw"
                        price_parquet = raw_dir / f"prices_live_v1_{key}_{bar_interval}.parquet"
                        price_csv = raw_dir / f"prices_live_v1_{key}_{bar_interval}.csv"
                        price_path = price_parquet if price_parquet.exists() else price_csv
                        social_path = raw_dir / f"social_live_v1_{key}.json"
                        news_path = raw_dir / f"news_live_v1_{key}.json"
                        cached_meta = cached_run_metadata.get(symbol)
                        if not (price_path.exists() and social_path.exists() and news_path.exists() and cached_meta):
                            raise FileNotFoundError(f"Complete cached run inputs are unavailable for {symbol}")
                        for input_path in (price_path, social_path, news_path):
                            if datetime.now(timezone.utc).timestamp() - input_path.stat().st_mtime > timedelta(hours=24).total_seconds():
                                raise ValueError(f"Cached input for {symbol} is older than 24 hours: {input_path.name}")

                        price_df = (
                            pd.read_parquet(price_path)
                            if price_path.suffix == ".parquet"
                            else pd.read_csv(price_path, index_col=0, parse_dates=True)
                        )
                        price_df.index = pd.to_datetime(price_df.index, utc=True)
                        price_df.index.name = "timestamp_utc"
                        social_payload = json.loads(social_path.read_text(encoding="utf-8"))
                        news_payload = json.loads(news_path.read_text(encoding="utf-8"))
                        social_posts = social_payload.get("items", [])
                        news_items = news_payload.get("items", [])
                        prior_source_statuses = cached_meta.get("social_source_statuses", "[]")
                        cached_source_statuses = (
                            json.loads(prior_source_statuses)
                            if isinstance(prior_source_statuses, str)
                            else prior_source_statuses
                        )
                        if not isinstance(cached_source_statuses, list):
                            cached_source_statuses = []
                        if args.refresh_hk_commentary and ex_code == "HKEX":
                            commentary_types = {"aastocks_commentary", "etnet_commentary"}
                            commentary_sources = [
                                source for source in social_sources
                                if source.get("type") in commentary_types
                            ]
                            retained_source_names = {
                                str(source.get("name")) for source in social_sources
                                if source.get("type") not in commentary_types
                            }
                            cutoff_utc = datetime.now(timezone.utc) - timedelta(days=lookback_days)
                            retained_posts = []
                            for post in social_posts:
                                if str(post.get("source")) not in retained_source_names:
                                    continue
                                try:
                                    post_time = datetime.fromisoformat(
                                        str(post["timestamp_utc"]).replace("Z", "+00:00")
                                    )
                                except (KeyError, TypeError, ValueError):
                                    continue
                                if cutoff_utc <= post_time <= datetime.now(timezone.utc):
                                    retained_posts.append(post)

                            refreshed_posts = collector_social.fetch_social_posts(
                                symbol,
                                name,
                                ex_code,
                                keywords,
                                tg_tags,
                                lookback_days=lookback_days,
                                max_posts=50,
                                force_refresh=True,
                                social_sources=commentary_sources,
                            )
                            refreshed_statuses = list(collector_social.last_source_statuses)
                            merged_posts = {
                                str(post.get("post_id")): post
                                for post in [*retained_posts, *refreshed_posts]
                                if post.get("post_id")
                            }
                            social_posts = sorted(
                                merged_posts.values(), key=lambda item: item["timestamp_utc"]
                            )
                            retained_statuses = [
                                status for status in cached_source_statuses
                                if str(status.get("source")) in retained_source_names
                            ]
                            collector_social.last_source_statuses = [
                                *retained_statuses,
                                *refreshed_statuses,
                            ]
                            social_payload.update({
                                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                                "lookback_days": lookback_days,
                                "items": social_posts,
                            })
                            social_path.write_text(
                                json.dumps(social_payload, indent=2, ensure_ascii=False),
                                encoding="utf-8",
                            )
                            logger.info(
                                "Refreshed HK commentary for %s (%d AASTOCKS/ETNet records)",
                                symbol,
                                len(refreshed_posts),
                            )
                        else:
                            collector_social.last_source_statuses = cached_source_statuses
                        collector_news.last_status = str(cached_meta.get("news_collection_status", "unknown"))
                        logger.info("Loaded saved public inputs for %s (%d posts, %d news items)", symbol, len(social_posts), len(news_items))
                    else:
                        # 1. Fetch Market Bars
                        price_df = collector_market.fetch_bars(
                            symbol, lookback_days=lookback_days, interval=bar_interval
                        )

                        # 2. Fetch Exa News
                        news_items = collector_news.fetch_news(
                            symbol, name, keywords, lookback_days=lookback_days
                        )

                        # 3. Fetch public sentiment records from configured sources.
                        social_posts = collector_social.fetch_social_posts(
                            symbol, name, ex_code, keywords, tg_tags,
                            lookback_days=lookback_days,
                            social_sources=social_sources,
                        )

                    # 4. Multilingual structured sentiment, batched per ticker.
                    # Keep the full-window scored records for the executive
                    # summary; only time-aligned copies enter model features.
                    scoring_status = "complete"
                    scoring_error = None
                    try:
                        scored_posts_all, scored_news_all = scorer.score_records(
                            social_posts,
                            news_items,
                            ticker=symbol,
                            asset_name=name,
                            exchange=ex_code,
                        )
                    except Exception as exc:
                        scoring_status = "failed"
                        scoring_error = type(exc).__name__
                        scored_posts_all, scored_news_all = [], []
                        logger.error("Sentiment scoring failed for %s (%s); forecasts will be withheld", symbol, scoring_error)

                    # 5. UTC Anti-Leakage Alignment. Future observations are
                    # omitted by DataAligner and cannot enter historical rows.
                    scored_posts = aligner.align_social_to_bars(price_df, scored_posts_all)
                    scored_news = aligner.align_news_to_bars(price_df, scored_news_all)

                    # 6. Feature Engineering & S/R Extraction
                    features_df, summary = feature_builder.build_dataset_for_ticker(
                        symbol, name, ex_code, price_df, scored_posts, scored_news,
                        summary_posts=scored_posts_all,
                        summary_news=scored_news_all,
                    )
                    summary["social_sources_expected"] = len(social_sources)
                    summary["social_source_statuses"] = list(collector_social.last_source_statuses)
                    summary["sentiment_provider"] = scorer.provider
                    summary["sentiment_scoring_status"] = scoring_status
                    summary["sentiment_scoring_error"] = scoring_error
                    summary["news_collection_status"] = collector_news.last_status
                    summary["total_social_mentions"] = len(social_posts)
                    summary["social_source_names"] = sorted({str(p.get("source")) for p in social_posts if p.get("source")})
                    raw_sources_with_posts = {
                        str(status.get("source"))
                        for status in collector_social.last_source_statuses
                        if status.get("status") == "available" and int(status.get("items", 0)) > 0
                    }
                    aligned_sources_with_posts = {
                        str(p.get("source"))
                        for p in scored_posts
                        if p.get("source")
                        and not p.get("is_spam", False)
                        and float(p.get("relevance_score", 1.0)) >= 0.25
                    }
                    summary["social_sources_with_posts"] = len(raw_sources_with_posts)
                    summary["social_sources_with_aligned_posts"] = len(aligned_sources_with_posts)
                    summary["aligned_social_post_count"] = len(scored_posts)
                    if scoring_status != "complete":
                        summary["news_data_status"] = "scoring_failed"
                    elif scored_news:
                        summary["news_data_status"] = "available"
                    elif news_items and collector_news.last_status == "available":
                        summary["news_data_status"] = "outside_price_window"
                    else:
                        summary["news_data_status"] = collector_news.last_status
                    if not raw_sources_with_posts:
                        summary["social_data_status"] = "no_data"
                    elif summary["social_sources_with_posts"] < 2:
                        summary["social_data_status"] = "partial"
                    else:
                        summary["social_data_status"] = "complete"

                    # 7. Model Out-of-Sample Validation & Predictor Training
                    required_sources = 2
                    if scoring_status != "complete":
                        predictor = None
                        metrics = {"ticker": symbol, "status": "sentiment_scoring_failed", "error_type": scoring_error}
                    elif summary["social_sources_with_aligned_posts"] < required_sources:
                        predictor = None
                        metrics = {
                            "ticker": symbol,
                            "status": "insufficient_aligned_social_sources",
                            "sources_with_posts": summary["social_sources_with_aligned_posts"],
                            "required_sources": required_sources,
                        }
                    elif not scored_news:
                        predictor = None
                        metrics = {"ticker": symbol, "status": "insufficient_news"}
                    else:
                        predictor, metrics = validator.evaluate_ticker(symbol, features_df)
                    summary["validation_status"] = metrics.get("status")
                    summary["validation_test_samples"] = metrics.get("test_samples")
                    if metrics.get("status") == "validated":
                        all_metrics.append(metrics)
                        for f_name, imp_val in metrics.get("feature_importance", {}).items():
                            feature_importances_accum[f_name] = feature_importances_accum.get(f_name, 0.0) + imp_val
                    else:
                        logger.warning("%s forecast/validation withheld: %s", symbol, metrics.get("status"))

                    # 8. Single-step n+1 Forecast. Do not invent one when the
                    # out-of-sample validation gate did not pass.
                    curr_price = summary["current_price"]
                    if predictor is not None:
                        pred_n1, ret_pct, pred_dir = predictor.predict_next_price(curr_price, features_df)
                    else:
                        pred_n1, ret_pct, pred_dir = None, None, "NOT VALIDATED"
                    summary["pred_n1_price"] = pred_n1
                    summary["pred_return_pct"] = ret_pct
                    summary["pred_direction"] = pred_dir

                    # 9. Generate Vector Chart
                    c_path = chart_gen.generate_ticker_chart(
                        symbol, name, features_df, summary["support_level"], summary["resistance_level"], pred_n1, pred_dir
                    )
                    chart_paths[symbol] = c_path

                    if ex_code == "HKEX":
                        hk_summaries.append(summary)
                    else:
                        kr_summaries.append(summary)
                    all_summaries.append(summary)

                    # Save prediction record
                    prediction_records.append({
                        "ticker": symbol,
                        "name": name,
                        "exchange": ex_code,
                        "current_price": curr_price,
                        "support_level": summary["support_level"],
                        "resistance_level": summary["resistance_level"],
                        "total_mentions": summary["total_social_mentions"],
                        "weighted_sentiment": summary["weighted_sentiment"],
                        "social_data_status": summary["social_data_status"],
                        "social_post_count": summary["total_social_mentions"],
                        "social_sources_with_posts": summary["social_sources_with_posts"],
                        "social_sources_expected": summary["social_sources_expected"],
                        "social_sources_with_aligned_posts": summary["social_sources_with_aligned_posts"],
                        "aligned_social_post_count": summary["aligned_social_post_count"],
                        "news_data_status": summary["news_data_status"],
                        "news_collection_status": summary["news_collection_status"],
                        "news_item_count": len(news_items),
                        "social_source_statuses": json.dumps(summary["social_source_statuses"], ensure_ascii=False),
                        "sentiment_provider": summary["sentiment_provider"],
                        "sentiment_scoring_status": summary["sentiment_scoring_status"],
                        "sentiment_scoring_error": summary["sentiment_scoring_error"],
                        "sentiment_direction": summary["direction"],
                        "suggested_position": summary["position"],
                        "predicted_n1_price": pred_n1,
                        "predicted_return_pct": ret_pct,
                        "predicted_direction": pred_dir,
                        "out_of_sample_hit_rate": metrics.get("model_hit_rate_pct"),
                        "baseline_hit_rate": metrics.get("majority_direction_baseline_hit_rate_pct"),
                        "model_mae": metrics.get("model_mae"),
                        "baseline_mae": metrics.get("persistence_mae"),
                        "model_rmse": metrics.get("model_rmse"),
                        "baseline_rmse": metrics.get("persistence_rmse"),
                        "out_of_sample_ic": metrics.get("model_ic"),
                        "strategy_sharpe_per_bar": metrics.get("strategy_sharpe_ratio_per_bar"),
                        "validation_status": metrics.get("status"),
                        "validation_test_samples": metrics.get("test_samples"),
                    })

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}", exc_info=True)

                progress.advance(task)

    # Export Predictions CSV
    pred_df = pd.DataFrame(prediction_records)
    csv_out_path = output_dir / "predictions.csv"
    pred_df.to_csv(csv_out_path, index=False)
    logger.info(f"Saved {len(pred_df)} prediction records to {csv_out_path}")

    # Compute averages only from actual out-of-sample evaluation results.
    mean_hit_rate = float(pd.Series([m["model_hit_rate_pct"] for m in all_metrics]).mean()) if all_metrics else None
    mean_baseline_hit_rate = float(pd.Series([m["majority_direction_baseline_hit_rate_pct"] for m in all_metrics]).mean()) if all_metrics else None
    mean_m_mae = float(pd.Series([m["model_mae"] for m in all_metrics]).mean()) if all_metrics else None
    mean_n_mae = float(pd.Series([m["persistence_mae"] for m in all_metrics]).mean()) if all_metrics else None
    mean_m_rmse = float(pd.Series([m["model_rmse"] for m in all_metrics]).mean()) if all_metrics else None
    mean_n_rmse = float(pd.Series([m["persistence_rmse"] for m in all_metrics]).mean()) if all_metrics else None
    ticker_ics = [m["model_ic"] for m in all_metrics if m.get("model_ic") is not None]
    mean_ic = float(pd.Series(ticker_ics).mean()) if ticker_ics else None
    mean_sharpe = float(pd.Series([m["strategy_sharpe_ratio_per_bar"] for m in all_metrics]).mean()) if all_metrics else None

    overall_metrics = {
        "mean_hit_rate_pct": round(mean_hit_rate, 2) if mean_hit_rate is not None else None,
        "mean_baseline_hit_rate_pct": round(mean_baseline_hit_rate, 2) if mean_baseline_hit_rate is not None else None,
        "mean_model_mae": round(mean_m_mae, 5) if mean_m_mae is not None else None,
        "mean_naive_mae": round(mean_n_mae, 5) if mean_n_mae is not None else None,
        "mean_model_rmse": round(mean_m_rmse, 5) if mean_m_rmse is not None else None,
        "mean_naive_rmse": round(mean_n_rmse, 5) if mean_n_rmse is not None else None,
        "mean_ic": round(mean_ic, 4) if mean_ic is not None else None,
        "mean_sharpe": round(mean_sharpe, 2) if mean_sharpe is not None else None,
        "sentiment_provider": scorer.provider,
        "validated_tickers": len(all_metrics),
    }

    # Generate Feature Importance Plot
    avg_importances = {k: v / len(all_metrics) for k, v in feature_importances_accum.items()} if all_metrics else {}
    feat_chart_path = chart_gen.generate_feature_importance_chart(avg_importances) if avg_importances else None

    # Generate PDF Report
    pdf_path = pdf_builder.build_report(
        hk_summaries=hk_summaries,
        kr_summaries=kr_summaries,
        chart_paths=chart_paths,
        feature_importance_path=feat_chart_path,
        overall_metrics=overall_metrics,
    )

    # Generate Summary Image for README
    summary_img_path = summary_img_builder.build_summary_image(
        hk_summaries=hk_summaries,
        kr_summaries=kr_summaries,
        overall_metrics=overall_metrics,
    )

    # Print Terminal Executive Tables
    _print_terminal_tables(hk_summaries, kr_summaries, overall_metrics, pdf_path, summary_img_path)


def _print_terminal_tables(hk_sums, kr_sums, metrics, pdf_path, img_path):
    """Renders formatted tables to the terminal using rich."""
    console.print("\n")
    console.rule("[bold green]EXECUTIVE VISUAL SUMMARY (Sorted by Mentions & Sentiment)[/bold green]")

    for title, sums in [("Hang Seng (HKEX)", hk_sums), ("South Korea (KRX)", kr_sums)]:
        table = Table(title=f"[bold cyan]{title}[/bold cyan]", header_style="bold magenta", border_style="dim")
        table.add_column("Ticker", style="bold yellow")
        table.add_column("Asset Name", style="white")
        table.add_column("Price", justify="right")
        table.add_column("Mentions", justify="right", style="cyan")
        table.add_column("Sources", justify="right", style="cyan")
        table.add_column("Sentiment", justify="right")
        table.add_column("Direction", justify="center")
        table.add_column("Position", justify="center")
        table.add_column("Support", justify="right", style="green")
        table.add_column("Resistance", justify="right", style="red")
        table.add_column("n+1 Target", justify="right", style="bold green")

        sorted_sums = sorted(
            sums,
            key=lambda x: (x["total_social_mentions"], x.get("weighted_sentiment") if x.get("weighted_sentiment") is not None else -2),
            reverse=True,
        )
        for s in sorted_sums:
            dir_str = s["direction"]
            if dir_str == "BULLISH":
                dir_styled = "[bold green][+] BULL[/bold green]"
            elif dir_str == "BEARISH":
                dir_styled = "[bold red][-] BEAR[/bold red]"
            elif dir_str == "NEUTRAL":
                dir_styled = "[dim][=] NEUT[/dim]"
            else:
                dir_styled = "[dim]N/A[/dim]"

            pos_str = s.get("position", "HOLD")
            pos_styled = f"[bold]{pos_str}[/bold]" if "LONG" in pos_str or "SHORT" in pos_str else f"[dim]{pos_str}[/dim]"

            sent_val = s.get("weighted_sentiment")
            if sent_val is None:
                sent_styled = "N/A"
            else:
                sent_styled = f"[green]+{sent_val:.2f}[/green]" if sent_val >= 0 else f"[red]{sent_val:.2f}[/red]"

            table.add_row(
                s["ticker"],
                s["asset_name"][:14],
                f"{s['current_price']:.2f}",
                str(s["total_social_mentions"]),
                f"{s.get('social_sources_with_posts', 0)}/{s.get('social_sources_expected', 0)}",
                sent_styled,
                dir_styled,
                pos_styled,
                f"{s['support_level']:.2f}",
                f"{s['resistance_level']:.2f}",
                f"{s['pred_n1_price']:.2f}" if s.get("pred_n1_price") is not None else "N/A",
            )
        console.print(table)
        console.print("")

    # Benchmark Summary Card
    bench_table = Table(title="[bold blue]Out-of-Sample Walk-Forward Validation vs Baselines[/bold blue]", border_style="blue")
    bench_table.add_column("Metric", style="bold white")
    bench_table.add_column("Agent Model", style="bold green")
    bench_table.add_column("Baseline", style="dim")
    bench_table.add_column("Alpha Edge", style="bold cyan")

    if metrics["mean_hit_rate_pct"] is None:
        bench_table.add_row("Validation", "N/A", "N/A", "Insufficient dated observations")
    else:
        bench_table.add_row(
            "Directional Hit Rate",
            f"{metrics['mean_hit_rate_pct']:.1f}%",
            f"{metrics['mean_baseline_hit_rate_pct']:.1f}% majority direction",
            f"{metrics['mean_hit_rate_pct'] - metrics['mean_baseline_hit_rate_pct']:+.1f}%",
        )
        bench_table.add_row(
            "Return MAE",
            f"{metrics['mean_model_mae']:.5f}",
            f"{metrics['mean_naive_mae']:.5f} zero-return",
            f"{((metrics['mean_naive_mae'] - metrics['mean_model_mae']) / metrics['mean_naive_mae']) * 100:+.1f}%",
        )
        bench_table.add_row(
            "Strategy Sharpe (per bar)",
            f"{metrics['mean_sharpe']:.2f}",
            "N/A",
            "Unannualized",
        )
    console.print(bench_table)

    console.print("\n[bold green]✓ Pipeline Execution Complete![/bold green]")
    console.print(f"  • Executive PDF Briefing: [bold underline]{pdf_path}[/bold underline]")
    console.print(f"  • Summary Image (README): [bold underline]{img_path}[/bold underline]")
    console.print(f"  • CSV Predictions:        [bold underline]{PROJECT_ROOT / 'outputs' / 'predictions.csv'}[/bold underline]")


if __name__ == "__main__":
    main()
