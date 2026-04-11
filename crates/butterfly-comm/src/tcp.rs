//! TCP-based communication backend for distributed node communication.
//!
//! Provides reliable, ordered message delivery using TCP sockets with:
//! - Automatic connection management and reconnection
//! - Length-prefixed framing for message boundaries
//! - Connection pooling for efficiency
//! - Exponential backoff retry logic

use crate::{CommunicationBackend, CommunicationError, Message};
use butterfly_core::NodeId;
use bytes::{Buf, BufMut, BytesMut};
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio_util::codec::{Decoder, Encoder, Framed};
use tracing::{debug, error, info, warn};

/// Maximum message size (100 MB)
const MAX_MESSAGE_SIZE: usize = 100 * 1024 * 1024;

/// Initial retry delay in milliseconds
const INITIAL_RETRY_DELAY_MS: u64 = 100;

/// Maximum retry delay in milliseconds
const MAX_RETRY_DELAY_MS: u64 = 32000;

/// Connection pool size per peer
const CONNECTIONS_PER_PEER: usize = 1;

/// TCP-based communication backend
pub struct TcpBackend {
    /// Local node ID
    node_id: NodeId,

    /// Address this backend listens on
    bind_addr: SocketAddr,

    /// Active connections to peers (NodeId -> Connection)
    connections: Arc<DashMap<NodeId, Connection>>,

    /// Peer addresses (NodeId -> SocketAddr)
    peer_addrs: Arc<DashMap<NodeId, SocketAddr>>,

    /// Incoming message queue (wrapped in Mutex for interior mutability)
    incoming: Arc<Mutex<mpsc::UnboundedReceiver<Message>>>,

    /// Sender for incoming messages (cloned to listener task)
    incoming_tx: mpsc::UnboundedSender<Message>,

    /// Listener task handle
    _listener_handle: Option<tokio::task::JoinHandle<()>>,
}

impl TcpBackend {
    /// Create a new TCP backend listening on the specified address
    pub async fn new(node_id: NodeId, bind_addr: SocketAddr) -> Result<Self, CommunicationError> {
        let listener = TcpListener::bind(bind_addr)
            .await
            .map_err(|e| CommunicationError::Network(format!("Failed to bind: {}", e)))?;

        let actual_addr = listener.local_addr()
            .map_err(|e| CommunicationError::Network(format!("Failed to get local addr: {}", e)))?;

        info!("TCP backend listening on {}", actual_addr);

        let (incoming_tx, incoming_rx) = mpsc::unbounded_channel();
        let connections = Arc::new(DashMap::new());
        let peer_addrs = Arc::new(DashMap::new());

        // Spawn listener task
        let listener_handle = tokio::spawn(Self::accept_loop(
            listener,
            connections.clone(),
            incoming_tx.clone(),
        ));

        Ok(Self {
            node_id,
            bind_addr: actual_addr,
            connections,
            peer_addrs,
            incoming: Arc::new(Mutex::new(incoming_rx)),
            incoming_tx,
            _listener_handle: Some(listener_handle),
        })
    }

    /// Register a peer's address for outbound connections
    pub fn register_peer(&self, node_id: NodeId, addr: SocketAddr) {
        debug!("Registering peer {:?} at {}", node_id, addr);
        self.peer_addrs.insert(node_id, addr);
    }

    /// Accept incoming connections in a loop
    async fn accept_loop(
        listener: TcpListener,
        connections: Arc<DashMap<NodeId, Connection>>,
        incoming_tx: mpsc::UnboundedSender<Message>,
    ) {
        loop {
            match listener.accept().await {
                Ok((stream, peer_addr)) => {
                    debug!("Accepted connection from {}", peer_addr);

                    let incoming_tx = incoming_tx.clone();
                    let connections = connections.clone();

                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_connection(stream, incoming_tx, connections).await {
                            warn!("Connection error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Accept error: {}", e);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }

    /// Handle an incoming connection
    async fn handle_connection(
        stream: TcpStream,
        incoming_tx: mpsc::UnboundedSender<Message>,
        _connections: Arc<DashMap<NodeId, Connection>>,
    ) -> Result<(), CommunicationError> {
        let mut framed = Framed::new(stream, MessageCodec);

        while let Some(result) = framed.next().await {
            match result {
                Ok(message) => {
                    debug!("Received message: {:?}", message);
                    if incoming_tx.send(message).is_err() {
                        warn!("Failed to queue incoming message");
                        break;
                    }
                }
                Err(e) => {
                    warn!("Frame decode error: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

    /// Get or create a connection to a peer
    async fn get_connection(&self, target: NodeId) -> Result<Connection, CommunicationError> {
        // Check if we already have a connection
        if let Some(conn) = self.connections.get(&target) {
            return Ok(conn.clone());
        }

        // Get peer address
        let addr = self.peer_addrs
            .get(&target)
            .ok_or_else(|| CommunicationError::Unreachable(target))?
            .value()
            .clone();

        // Create new connection
        let conn = Connection::connect(addr).await?;
        self.connections.insert(target, conn.clone());

        Ok(conn)
    }
}

#[async_trait::async_trait]
impl CommunicationBackend for TcpBackend {
    async fn send(&self, target: NodeId, message: Message) -> Result<(), CommunicationError> {
        let mut retry_delay = INITIAL_RETRY_DELAY_MS;
        let mut attempts = 0;

        loop {
            match self.get_connection(target).await {
                Ok(conn) => {
                    match conn.send(message.clone()).await {
                        Ok(_) => {
                            debug!("Sent message to {:?}", target);
                            return Ok(());
                        }
                        Err(e) => {
                            warn!("Send failed to {:?}: {}, retrying...", target, e);
                            self.connections.remove(&target);
                        }
                    }
                }
                Err(e) => {
                    warn!("Connection failed to {:?}: {}", target, e);
                }
            }

            attempts += 1;
            if attempts >= 5 {
                return Err(CommunicationError::Unreachable(target));
            }

            tokio::time::sleep(Duration::from_millis(retry_delay)).await;
            retry_delay = (retry_delay * 2).min(MAX_RETRY_DELAY_MS);
        }
    }

    async fn receive(&self) -> Result<Message, CommunicationError> {
        let mut incoming = self.incoming.lock().await;
        incoming
            .recv()
            .await
            .ok_or_else(|| CommunicationError::Network("Channel closed".to_string()))
    }

    async fn broadcast(&self, message: Message) -> Result<(), CommunicationError> {
        let mut errors = Vec::new();

        for entry in self.peer_addrs.iter() {
            let node_id = *entry.key();
            if let Err(e) = self.send(node_id, message.clone()).await {
                errors.push((node_id, e));
            }
        }

        if !errors.is_empty() {
            warn!("Broadcast failed to {} nodes", errors.len());
            // Return error if majority failed
            if errors.len() * 2 > self.peer_addrs.len() {
                return Err(CommunicationError::Network(format!(
                    "Broadcast failed to majority ({}/{})",
                    errors.len(),
                    self.peer_addrs.len()
                )));
            }
        }

        Ok(())
    }
}

/// A connection to a peer node
#[derive(Clone)]
struct Connection {
    sender: Arc<RwLock<mpsc::UnboundedSender<Message>>>,
}

impl Connection {
    /// Connect to a peer at the given address
    async fn connect(addr: SocketAddr) -> Result<Self, CommunicationError> {
        let stream = TcpStream::connect(addr)
            .await
            .map_err(|e| CommunicationError::Network(format!("Connect failed: {}", e)))?;

        debug!("Connected to {}", addr);

        let framed = Framed::new(stream, MessageCodec);
        let (mut sink, _stream) = framed.split();

        let (tx, mut rx) = mpsc::unbounded_channel::<Message>();

        // Spawn task to write messages to the stream
        tokio::spawn(async move {
            while let Some(message) = rx.recv().await {
                if let Err(e) = sink.send(message).await {
                    warn!("Failed to send message: {}", e);
                    break;
                }
            }
        });

        Ok(Self {
            sender: Arc::new(RwLock::new(tx)),
        })
    }

    /// Send a message through this connection
    async fn send(&self, message: Message) -> Result<(), CommunicationError> {
        let sender = self.sender.read().await;
        sender
            .send(message)
            .map_err(|_| CommunicationError::Network("Connection closed".to_string()))
    }
}

/// Codec for length-prefixed message framing
struct MessageCodec;

impl Decoder for MessageCodec {
    type Item = Message;
    type Error = std::io::Error;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        if src.len() < 4 {
            // Not enough data to read length prefix
            return Ok(None);
        }

        // Read length prefix
        let mut length_bytes = [0u8; 4];
        length_bytes.copy_from_slice(&src[..4]);
        let length = u32::from_be_bytes(length_bytes) as usize;

        if length > MAX_MESSAGE_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Message too large: {} bytes", length)
            ));
        }

        if src.len() < 4 + length {
            // Not enough data yet
            src.reserve(4 + length - src.len());
            return Ok(None);
        }

        // We have a complete message
        src.advance(4);
        let data = src.split_to(length);

        // Deserialize
        let message = crate::serialization::deserialize_message(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        Ok(Some(message))
    }
}

impl Encoder<Message> for MessageCodec {
    type Error = std::io::Error;

    fn encode(&mut self, item: Message, dst: &mut BytesMut) -> Result<(), Self::Error> {
        // Serialize message
        let data = crate::serialization::serialize_message(&item)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        if data.len() > MAX_MESSAGE_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Message too large: {} bytes", data.len())
            ));
        }

        // Write length prefix
        let length = data.len() as u32;
        dst.reserve(4 + data.len());
        dst.put_u32(length);
        dst.put_slice(&data);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tcp_backend_creation() {
        let addr = "127.0.0.1:0".parse().unwrap();
        let backend = TcpBackend::new(NodeId(1), addr).await.unwrap();
        assert!(backend.bind_addr.port() > 0);
    }

    #[tokio::test]
    async fn test_peer_registration() {
        let addr = "127.0.0.1:0".parse().unwrap();
        let backend = TcpBackend::new(NodeId(1), addr).await.unwrap();

        let peer_addr = "127.0.0.1:9000".parse().unwrap();
        backend.register_peer(NodeId(2), peer_addr);

        assert!(backend.peer_addrs.contains_key(&NodeId(2)));
    }
}
