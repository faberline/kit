//! agentkit-graph: a stateful, typed graph runtime for agents.
//!
//! The runtime is a typed `State`, a directed graph of `Node`s, and a `run` loop
//! that walks the configured edges. See [`Graph`] for the core type.
//!
//! This first slice supports linear paths only: the loop refuses to run on a
//! graph that has more than one outgoing edge from any node.

// HANDWRITE-BEGIN reason: no Rust-runtime generator for an async graph
// engine yet. Once a `graph-engine` section type lands that can emit
// `Graph` / `Node` / `Edge` from the lifecycle/dependency YAML, swap to
// CODEGEN markers.

pub mod checkpoint;
pub mod events;
pub mod graph;
pub mod node;

pub use checkpoint::{Checkpoint, FileCheckpoint, MemoryCheckpoint};
pub use events::{GraphEvent, GraphEventBus};
pub use graph::{Edge, EdgePicker, Graph, GraphError};
pub use node::{FnNode, Node, NodeRef};

// HANDWRITE-END
