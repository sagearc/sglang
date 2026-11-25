// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::http::{Request as HttpRequest, Response as HttpResponse};
use futures::future::{ready, BoxFuture, Ready};
use futures::stream::BoxStream;
use std::collections::HashMap;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::body::Body as GrpcBody;
use tonic::server::{Grpc, NamedService, ServerStreamingService, UnaryService};
use tonic::Status;
use tonic_prost::ProstCodec;

pub type Replies<T> = BoxStream<'static, Result<T, Status>>;
type Handler =
    Arc<dyn Fn(HttpRequest<GrpcBody>) -> BoxFuture<'static, HttpResponse<GrpcBody>> + Send + Sync>;

/// An engine serving the RPCs it is given, by method name.
#[derive(Clone, Default)]
pub struct MockEngine(HashMap<&'static str, Handler>);

/// Adapts a closure to a tonic service; records nothing itself.
struct Svc<F>(Arc<F>);

impl<Req, Rep, F> ServerStreamingService<Req> for Svc<F>
where
    F: Fn(tonic::Request<Req>) -> Replies<Rep> + Send + Sync + 'static,
    Rep: Send + 'static,
{
    type Response = Rep;
    type ResponseStream = Replies<Rep>;
    type Future = Ready<Result<tonic::Response<Replies<Rep>>, Status>>;

    fn call(&mut self, request: tonic::Request<Req>) -> Self::Future {
        ready(Ok(tonic::Response::new((self.0)(request))))
    }
}

struct Unary<F>(Arc<F>);

impl<Req, Rep, F> UnaryService<Req> for Unary<F>
where
    F: Fn(Req) -> Rep + Send + Sync + 'static,
{
    type Response = Rep;
    type Future = Ready<Result<tonic::Response<Rep>, Status>>;

    fn call(&mut self, request: tonic::Request<Req>) -> Self::Future {
        ready(Ok(tonic::Response::new((self.0)(request.into_inner()))))
    }
}

impl MockEngine {
    pub fn stream<Req, Rep>(
        mut self,
        method: &'static str,
        reply: impl Fn(tonic::Request<Req>) -> Replies<Rep> + Send + Sync + 'static,
    ) -> Self
    where
        Req: prost::Message + Default + Send + 'static,
        Rep: prost::Message + Send + 'static,
    {
        let reply = Arc::new(reply);
        let handler: Handler = Arc::new(move |request| {
            let svc = Svc(Arc::clone(&reply));
            Box::pin(async move {
                let codec = ProstCodec::<Rep, Req>::default();
                Grpc::new(codec).server_streaming(svc, request).await
            })
        });
        self.0.insert(method, handler);
        self
    }

    pub fn unary<Req, Rep>(
        mut self,
        method: &'static str,
        reply: impl Fn(Req) -> Rep + Send + Sync + 'static,
    ) -> Self
    where
        Req: prost::Message + Default + Send + 'static,
        Rep: prost::Message + Send + 'static,
    {
        let reply = Arc::new(reply);
        let handler: Handler = Arc::new(move |request| {
            let svc = Unary(Arc::clone(&reply));
            Box::pin(async move {
                let codec = ProstCodec::<Rep, Req>::default();
                Grpc::new(codec).unary(svc, request).await
            })
        });
        self.0.insert(method, handler);
        self
    }

    /// Serve on 127.0.0.1 and return the port.
    pub async fn start(self) -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tonic::transport::Server::builder().add_service(self);
        tokio::spawn(server.serve_with_incoming(TcpListenerStream::new(listener)));
        port
    }
}

impl NamedService for MockEngine {
    const NAME: &'static str = "sglang.api.v1.SglangService";
}

impl tower::Service<HttpRequest<GrpcBody>> for MockEngine {
    type Response = HttpResponse<GrpcBody>;
    type Error = std::convert::Infallible;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: HttpRequest<GrpcBody>) -> Self::Future {
        let method = request.uri().path().rsplit('/').next().unwrap_or_default();
        let Some(handler) = self.0.get(method).cloned() else {
            return Box::pin(async {
                Ok(Status::unimplemented("mock RPC is not served").into_http())
            });
        };
        Box::pin(async move { Ok(handler(request).await) })
    }
}
