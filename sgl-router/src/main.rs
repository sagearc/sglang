// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use clap::Parser;
use sgl_router::{
    config::{Cli, Config, LogFormat},
    routing,
    server::{app::build_router, app_context::AppContext, shutdown::drain_for_termination},
    state::load_monitor::router_inflight_load::{
        spawn_janitor, JanitorHandle, RouterInflightLoadRegistry, SystemTimeClock,
    },
    tokenizer::TokenizerRegistry,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    net::TcpListener,
    signal::unix::{signal, Signal, SignalKind},
    sync::{oneshot, watch},
};
use tokio_util::sync::CancellationToken;

const DRAIN_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
/// Heartbeat escalates INFO -> WARN here: earlier is a routine rollout draining
/// a long response; later the pod risks SIGKILL with work still open.
const DRAIN_WARN_AFTER: Duration = Duration::from_secs(30);

#[tokio::main]
async fn main() -> Result<()> {
    // Resolve CLI configuration and set up startup logging.
    let cli = Cli::parse();
    let routing_config = cli.routing.chat_routing;
    init_tracing(&cli.server.log_level, cli.server.log_format)?;
    #[cfg(feature = "fips")]
    install_fips_crypto_provider()?;
    let config = cli
        .into_config()
        .context("resolve configuration from CLI flags")?;

    // Buffer termination signals before tokenizer loading or discovery can block startup.
    let (sigterm, sigint) = install_signal_handlers()?;
    log_startup(&config);

    // Load tokenizers used to prepare requests for routing.
    let tokenizers =
        Arc::new(TokenizerRegistry::load_from_config(&config).context("load tokenizers")?);

    let (local_inflight_requests, inflight_cleanup) = start_local_inflight_tracker(&config);
    let (app_context, workers) =
        routing::start(&config, routing_config, tokenizers, local_inflight_requests).await?;
    app_context.mark_ready();

    // Serve HTTP requests until shutdown, allowing in-flight requests to finish.
    let listen_addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = TcpListener::bind(&listen_addr)
        .await
        .with_context(|| format!("bind {listen_addr}"))?;
    tracing::info!("listening on {listen_addr}");
    let grpc_listener = match config.server.grpc_port {
        Some(port) => {
            let grpc_addr = format!("{}:{port}", config.server.host);
            let listener = TcpListener::bind(&grpc_addr)
                .await
                .with_context(|| format!("bind {grpc_addr}"))?;
            tracing::info!("serving native gRPC on {grpc_addr}");
            Some(listener)
        }
        None => None,
    };
    let outcome = serve(listener, grpc_listener, app_context, sigterm, sigint).await;

    // Stop background tasks once the HTTP server has finished draining.
    workers.shutdown().await;
    inflight_cleanup.shutdown().await;
    log_shutdown(&outcome.result, outcome.inflight_drain_secs);
    outcome.result
}

// Respect RUST_LOG and tolerate an already-installed subscriber.
fn init_tracing(default_level: &str, format: LogFormat) -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));
    let install_result = match format {
        LogFormat::Json => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .json()
            .try_init(),
        LogFormat::Text => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .try_init(),
    };
    if let Err(e) = install_result {
        tracing::debug!(
            default_level = %default_level,
            ?format,
            error = %e,
            "tracing subscriber already installed; continuing"
        );
    }
    Ok(())
}

/// Make aws-lc-rs's FIPS module the process-wide rustls provider before any TLS
/// client is built (reqwest and kube take the installed default), and refuse to
/// start unless the module runs in FIPS mode.
#[cfg(feature = "fips")]
fn install_fips_crypto_provider() -> Result<()> {
    rustls::crypto::default_fips_provider()
        .install_default()
        .map_err(|_| {
            anyhow::anyhow!("a rustls crypto provider was installed before the FIPS one")
        })?;
    aws_lc_rs::try_fips_mode()
        .map_err(|e| anyhow::anyhow!("aws-lc-rs is not in FIPS mode: {e}"))?;
    anyhow::ensure!(
        rustls::crypto::CryptoProvider::get_default().is_some_and(|provider| provider.fips()),
        "the installed rustls crypto provider is not FIPS-approved"
    );
    tracing::info!("FIPS mode: aws-lc-rs FIPS module is the rustls crypto provider");
    Ok(())
}

fn install_signal_handlers() -> Result<(Signal, Signal)> {
    let sigterm = signal(SignalKind::terminate()).context("install SIGTERM handler")?;
    let sigint = signal(SignalKind::interrupt()).context("install SIGINT handler")?;
    Ok((sigterm, sigint))
}

fn log_startup(config: &Config) {
    // Resolve (and log, or warn about) SGLANG_ROUTER_MAX_BODY_BYTES with the
    // rest of the startup configuration, not when the router is first built.
    sgl_router::server::routes::chat::max_chat_body_bytes();
    if let Some(advisory) = sgl_router::config::shutdown_drain_advisory(
        config.server.shutdown_drain_secs,
        config.server.termination_grace_secs,
    ) {
        tracing::warn!(
            shutdown_drain_secs = advisory.shutdown_drain_secs,
            termination_grace_secs = advisory.termination_grace_secs,
            grace_declared = advisory.grace_declared,
            "shutdown drain leaves no room under terminationGracePeriodSeconds for the \
             in-flight drain that follows it; raise the grace period to at least the drain \
             plus in-flight request time, or lower the drain. If the grace period is already \
             higher, declare it with --termination-grace-secs",
        );
    }

    tracing::info!(
        "sgl-router {} starting on {}:{}",
        env!("CARGO_PKG_VERSION"),
        config.server.host,
        config.server.port
    );
}

fn start_local_inflight_tracker(
    config: &Config,
) -> (Arc<RouterInflightLoadRegistry>, JanitorHandle) {
    let timeout_secs = config.router_inflight_load.stale_request_timeout_secs;
    let local_inflight_requests = RouterInflightLoadRegistry::new(
        Arc::new(SystemTimeClock),
        Duration::from_secs(timeout_secs),
    );
    // Reap stale requests at one tenth of their timeout, bounded to 1–60 seconds.
    let sweep_interval = Duration::from_secs((timeout_secs / 10).clamp(1, 60));
    let inflight_cleanup = spawn_janitor(Arc::clone(&local_inflight_requests), sweep_interval);
    (local_inflight_requests, inflight_cleanup)
}

/// How serving ended; `inflight_drain_secs` is `None` when the server stopped
/// without ever reaching the in-flight drain.
struct ServeOutcome {
    result: Result<()>,
    inflight_drain_secs: Option<u64>,
}

async fn serve(
    listener: TcpListener,
    grpc_listener: Option<TcpListener>,
    app_context: Arc<AppContext>,
    sigterm: Signal,
    sigint: Signal,
) -> ServeOutcome {
    let app = build_router(Arc::clone(&app_context));
    let drain = app_context.config.server.shutdown_drain();
    let (drain_tx, drain_rx) = watch::channel(None);
    let heartbeat = tokio::spawn(report_drain_progress(
        Arc::clone(&app_context),
        drain_rx.clone(),
    ));
    // gRPC stops accepting with HTTP, or when HTTP serving fails.
    let grpc_stop = CancellationToken::new();
    let grpc = grpc_listener.map(|listener| {
        let stop = grpc_stop.clone();
        let app_context = Arc::clone(&app_context);
        tokio::spawn(async move {
            let shutdown = async move { stop.cancelled().await };
            let result = sgl_router::server::grpc::serve(listener, app_context, shutdown).await;
            if let Err(error) = &result {
                tracing::error!(%error, "native gRPC server stopped");
            }
            result
        })
    });
    let stop = grpc_stop.clone();
    let mut result = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal(sigterm, sigint, app_context, drain).await;
            let _ = drain_tx.send(Some(Instant::now()));
            stop.cancel();
        })
        .await
        .context("axum serve");
    grpc_stop.cancel();
    if let Some(grpc) = grpc {
        let grpc_result = grpc
            .await
            .context("gRPC server task")
            .and_then(|r| r.context("gRPC serve"));
        result = result.and(grpc_result);
    }
    heartbeat.abort();
    let inflight_drain_secs = drain_rx.borrow().map(|at| at.elapsed().as_secs());
    ServeOutcome {
        result,
        inflight_drain_secs,
    }
}

async fn report_drain_progress(
    app_context: Arc<AppContext>,
    mut drain_rx: watch::Receiver<Option<Instant>>,
) {
    // Start reporting only after the readiness pause, when axum begins draining requests.
    let Ok(started) = drain_rx
        .wait_for(Option::is_some)
        .await
        .map(|at| at.expect("wait_for only resolves once the instant is published"))
    else {
        return;
    };
    let mut ticker = tokio::time::interval(DRAIN_HEARTBEAT_INTERVAL);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    ticker.tick().await; // the first tick completes immediately

    macro_rules! heartbeat {
        ($level:ident, $elapsed:expr) => {
            tracing::$level!(
                elapsed_secs = $elapsed,
                inflight_http = app_context.inflight_http.count(),
                inflight_proxied = app_context.router_inflight_load.inflight_count(),
                "still draining in-flight requests; this phase is unbounded and ends at \
                 SIGKILL when terminationGracePeriodSeconds expires",
            )
        };
    }
    loop {
        ticker.tick().await;
        let elapsed = started.elapsed();
        if elapsed < DRAIN_WARN_AFTER {
            heartbeat!(info, elapsed.as_secs());
        } else {
            heartbeat!(warn, elapsed.as_secs());
        }
    }
}

fn log_shutdown(result: &Result<()>, inflight_drain_secs: Option<u64>) {
    match (result, inflight_drain_secs) {
        (Ok(()), Some(inflight_drain_secs)) => {
            tracing::info!(inflight_drain_secs, "shutdown complete")
        }
        (Ok(()), None) => {
            tracing::info!("shutdown complete; the server stopped without a termination signal")
        }
        (Err(e), Some(inflight_drain_secs)) => tracing::error!(
            error = %e,
            inflight_drain_secs,
            "shutdown complete, but the server exited with an error",
        ),
        (Err(e), None) => tracing::error!(
            error = %e,
            "the server exited with an error before any termination signal",
        ),
    }
}

// SIGTERM pauses for readiness propagation; SIGINT goes straight to the in-flight drain.
async fn shutdown_signal(
    mut sigterm: Signal,
    mut sigint: Signal,
    app_context: Arc<AppContext>,
    drain: Duration,
) {
    let sigterm_first = tokio::select! {
        _ = sigterm.recv() => {
            tracing::info!("got SIGTERM, shutting down");
            true
        }
        _ = sigint.recv() => {
            tracing::info!("got SIGINT, shutting down without the readiness drain");
            false
        }
    };

    let (expedite_tx, expedite_rx) = oneshot::channel::<()>();
    let mut expedite_tx = sigterm_first.then_some(expedite_tx);
    // Keep consuming signals during the in-flight drain; tokio never restores default handlers.
    tokio::spawn(async move {
        loop {
            let delivered = tokio::select! {
                delivered = sigterm.recv() => delivered,
                delivered = sigint.recv() => delivered,
            };
            if delivered.is_none() {
                return;
            }
            handle_further_signal(&mut expedite_tx, sigterm_first);
        }
    });

    if sigterm_first {
        let expedite = async move {
            let _ = expedite_rx.await;
        };
        drain_for_termination(&app_context, drain, expedite).await;
    }
}

#[derive(Debug, PartialEq, Eq)]
enum FurtherSignal {
    Expedited,
    Ignored,
}

fn handle_further_signal(
    expedite_tx: &mut Option<oneshot::Sender<()>>,
    sigterm_first: bool,
) -> FurtherSignal {
    // A failed `send` means the pause already elapsed; it must fall through to
    // the notice below. Discarding the `Err` once swallowed the first
    // post-pause signal (see the regression test).
    if let Some(tx) = expedite_tx.take() {
        if tx.send(()).is_ok() {
            return FurtherSignal::Expedited;
        }
    }
    if sigterm_first {
        tracing::warn!(
            "further termination signal ignored: the readiness pause is over and the \
             in-flight drain cannot be cut short; send SIGKILL to force an immediate exit",
        );
    } else {
        tracing::warn!(
            "further termination signal ignored: SIGINT skips the readiness pause and the \
             in-flight drain cannot be cut short; send SIGKILL to force an immediate exit",
        );
    }
    FurtherSignal::Ignored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn install_signal_handlers_returns_both() {
        assert!(install_signal_handlers().is_ok());
    }

    #[test]
    fn a_further_signal_expedites_a_running_pause() {
        let (tx, mut rx) = oneshot::channel::<()>();
        let mut expedite_tx = Some(tx);
        assert_eq!(
            handle_further_signal(&mut expedite_tx, true),
            FurtherSignal::Expedited,
        );
        assert!(
            expedite_tx.is_none(),
            "the sender is spent once the pause has been expedited",
        );
        assert!(
            rx.try_recv().is_ok(),
            "the running pause must actually be notified",
        );
    }

    #[test]
    fn the_first_signal_after_the_pause_ends_is_reported_not_swallowed() {
        let (tx, rx) = oneshot::channel::<()>();
        drop(rx); // the pause elapsed and dropped its receiver
        let mut expedite_tx = Some(tx);
        assert_eq!(
            handle_further_signal(&mut expedite_tx, true),
            FurtherSignal::Ignored,
            "a send into a dropped receiver expedites nothing and must say so",
        );
        assert_eq!(
            handle_further_signal(&mut expedite_tx, true),
            FurtherSignal::Ignored,
        );
    }

    #[test]
    fn a_further_signal_on_the_sigint_path_is_always_ignored() {
        let mut expedite_tx = None;
        assert_eq!(
            handle_further_signal(&mut expedite_tx, false),
            FurtherSignal::Ignored,
        );
    }

    #[test]
    fn init_tracing_is_idempotent() {
        let _ = init_tracing("info", LogFormat::Text);
        let _ = init_tracing("info", LogFormat::Text);
    }

    #[test]
    fn init_tracing_accepts_json_format() {
        assert!(init_tracing("info", LogFormat::Json).is_ok());
    }
}
