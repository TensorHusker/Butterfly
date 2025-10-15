//! State management types and traits for Butterfly distributed system
//!
//! This module provides the foundational types for managing state across
//! all components of the Butterfly system, ensuring consistency and
//! enabling formal reasoning about system behavior.

pub mod system;
pub mod component;
pub mod transition;
pub mod invariants;

pub use system::{
    SystemState, Phase, TransitionId, TransitionTrigger, StateTransitionRecord,
};
pub use component::{
    ComponentState, StateMachine, PersistentStateMachine, StateError,
};
pub use transition::{
    TransitionRule, TransitionRules, ProofObligation, TransitionContext,
};
pub use invariants::{
    StateInvariant, InvariantViolation, Severity,
};
