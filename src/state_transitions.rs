//! Controlled state transitions module.
//!
//! Enforces explicit, validated transitions between task states.
//! Prevents skipping phases (e.g., can't execute before plan is approved,
//! can't finish without validation).

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Task lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum State {
    Draft,
    PlanDefined,
    PlanApproved,
    InProgress,
    Implemented,
    Validated,
    Done,
    Paused,
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            State::Draft => write!(f, "Draft"),
            State::PlanDefined => write!(f, "PlanDefined"),
            State::PlanApproved => write!(f, "PlanApproved"),
            State::InProgress => write!(f, "InProgress"),
            State::Implemented => write!(f, "Implemented"),
            State::Validated => write!(f, "Validated"),
            State::Done => write!(f, "Done"),
            State::Paused => write!(f, "Paused"),
        }
    }
}

/// Result of an attempted transition.
#[derive(Debug, PartialEq, Eq)]
pub enum TransitionResult {
    Ok,
    Denied { from: State, to: State, reason: String },
}

/// Records a completed transition.
#[derive(Debug, Clone)]
pub struct TransitionRecord {
    pub from: State,
    pub to: State,
}

/// A controlled state machine with explicit allowed transitions.
#[derive(Debug)]
pub struct ControlledStateMachine {
    current: State,
    /// Map from source state to set of allowed target states.
    allowed_transitions: HashMap<State, HashSet<State>>,
    /// State saved before pausing, so we can resume to the correct state.
    paused_from: Option<State>,
    /// Log of all transitions.
    history: Vec<TransitionRecord>,
}

impl ControlledStateMachine {
    /// Creates a state machine with default lifecycle transitions.
    ///
    /// Default allowed transitions:
    /// - Draft → PlanDefined
    /// - PlanDefined → PlanApproved
    /// - PlanApproved → InProgress
    /// - InProgress → Implemented
    /// - Implemented → Validated
    /// - Validated → Done
    /// - Any non-terminal state → Paused (pause)
    /// - Paused → (restored state) (resume)
    pub fn new() -> Self {
        let mut allowed = HashMap::new();

        allowed.insert(
            State::Draft,
            HashSet::from([State::PlanDefined, State::Paused]),
        );
        allowed.insert(
            State::PlanDefined,
            HashSet::from([State::PlanApproved, State::Paused]),
        );
        allowed.insert(
            State::PlanApproved,
            HashSet::from([State::InProgress, State::Paused]),
        );
        allowed.insert(
            State::InProgress,
            HashSet::from([State::Implemented, State::Paused]),
        );
        allowed.insert(
            State::Implemented,
            HashSet::from([State::Validated, State::Paused]),
        );
        allowed.insert(
            State::Validated,
            HashSet::from([State::Done, State::Paused]),
        );
        // Done is terminal — no transitions allowed
        allowed.insert(State::Done, HashSet::new());
        // Paused can only resume (handled specially)
        allowed.insert(State::Paused, HashSet::new());

        Self {
            current: State::Draft,
            allowed_transitions: allowed,
            paused_from: None,
            history: Vec::new(),
        }
    }

    /// Creates a state machine with custom allowed transitions.
    pub fn with_transitions(transitions: HashMap<State, HashSet<State>>) -> Self {
        Self {
            current: State::Draft,
            allowed_transitions: transitions,
            paused_from: None,
            history: Vec::new(),
        }
    }

    /// Returns the current state.
    pub fn current(&self) -> State {
        self.current
    }

    /// Returns allowed target states from the current state.
    pub fn allowed_from_current(&self) -> Vec<State> {
        self.allowed_transitions
            .get(&self.current)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Attempts a transition. Returns `TransitionResult::Denied` if not allowed.
    pub fn transition(&mut self, target: State) -> TransitionResult {
        if target == State::Paused {
            return self.pause();
        }

        let allowed = self
            .allowed_transitions
            .get(&self.current)
            .map(|s| s.contains(&target))
            .unwrap_or(false);

        if !allowed {
            return TransitionResult::Denied {
                from: self.current,
                to: target,
                reason: format!(
                    "Transition from {} to {} is not allowed. Allowed targets: {:?}",
                    self.current,
                    target,
                    self.allowed_from_current()
                ),
            };
        }

        let record = TransitionRecord {
            from: self.current,
            to: target,
        };
        self.history.push(record);
        self.current = target;
        TransitionResult::Ok
    }

    /// Pauses the state machine, saving the current state.
    fn pause(&mut self) -> TransitionResult {
        if self.current == State::Done {
            return TransitionResult::Denied {
                from: self.current,
                to: State::Paused,
                reason: "Cannot pause: task is already done.".to_string(),
            };
        }
        if self.current == State::Paused {
            return TransitionResult::Denied {
                from: self.current,
                to: State::Paused,
                reason: "Already paused.".to_string(),
            };
        }

        self.paused_from = Some(self.current);
        let record = TransitionRecord {
            from: self.current,
            to: State::Paused,
        };
        self.history.push(record);
        self.current = State::Paused;
        TransitionResult::Ok
    }

    /// Resumes from paused state back to the state before pausing.
    pub fn resume(&mut self) -> TransitionResult {
        if self.current != State::Paused {
            return TransitionResult::Denied {
                from: self.current,
                to: self.current,
                reason: "Cannot resume: not currently paused.".to_string(),
            };
        }

        let restored = self.paused_from.take().expect("paused_from must be set");
        let record = TransitionRecord {
            from: State::Paused,
            to: restored,
        };
        self.history.push(record);
        self.current = restored;
        TransitionResult::Ok
    }

    /// Returns the full transition history.
    pub fn history(&self) -> &[TransitionRecord] {
        &self.history
    }

    /// Returns true if the task is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        self.current == State::Done
    }
}

impl Default for ControlledStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_happy_path() {
        let mut sm = ControlledStateMachine::new();
        assert_eq!(sm.current(), State::Draft);

        assert_eq!(sm.transition(State::PlanDefined), TransitionResult::Ok);
        assert_eq!(sm.transition(State::PlanApproved), TransitionResult::Ok);
        assert_eq!(sm.transition(State::InProgress), TransitionResult::Ok);
        assert_eq!(sm.transition(State::Implemented), TransitionResult::Ok);
        assert_eq!(sm.transition(State::Validated), TransitionResult::Ok);
        assert_eq!(sm.transition(State::Done), TransitionResult::Ok);

        assert!(sm.is_terminal());
        assert_eq!(sm.history().len(), 6);
    }

    #[test]
    fn test_skip_plan_denied() {
        let mut sm = ControlledStateMachine::new();
        // Try to jump from Draft directly to InProgress
        let result = sm.transition(State::InProgress);
        assert!(matches!(result, TransitionResult::Denied { .. }));
        assert_eq!(sm.current(), State::Draft); // state unchanged
    }

    #[test]
    fn test_skip_validation_denied() {
        let mut sm = ControlledStateMachine::new();
        sm.transition(State::PlanDefined);
        sm.transition(State::PlanApproved);
        sm.transition(State::InProgress);
        sm.transition(State::Implemented);

        // Try to skip validation and go directly to Done
        let result = sm.transition(State::Done);
        assert!(matches!(result, TransitionResult::Denied { .. }));
        assert_eq!(sm.current(), State::Implemented);
    }

    #[test]
    fn test_cannot_execute_before_plan_approved() {
        let mut sm = ControlledStateMachine::new();
        sm.transition(State::PlanDefined);

        // PlanDefined → InProgress is not allowed (must go through PlanApproved)
        let result = sm.transition(State::InProgress);
        assert!(matches!(result, TransitionResult::Denied { .. }));
    }

    #[test]
    fn test_pause_and_resume() {
        let mut sm = ControlledStateMachine::new();
        sm.transition(State::PlanDefined);
        sm.transition(State::PlanApproved);
        sm.transition(State::InProgress);

        // Pause
        assert_eq!(sm.transition(State::Paused), TransitionResult::Ok);
        assert_eq!(sm.current(), State::Paused);

        // Resume
        assert_eq!(sm.resume(), TransitionResult::Ok);
        assert_eq!(sm.current(), State::InProgress);

        // Can continue after resume
        assert_eq!(sm.transition(State::Implemented), TransitionResult::Ok);
    }

    #[test]
    fn test_cannot_pause_when_done() {
        let mut sm = ControlledStateMachine::new();
        sm.transition(State::PlanDefined);
        sm.transition(State::PlanApproved);
        sm.transition(State::InProgress);
        sm.transition(State::Implemented);
        sm.transition(State::Validated);
        sm.transition(State::Done);

        let result = sm.transition(State::Paused);
        assert!(matches!(result, TransitionResult::Denied { .. }));
    }

    #[test]
    fn test_cannot_resume_when_not_paused() {
        let mut sm = ControlledStateMachine::new();
        let result = sm.resume();
        assert!(matches!(result, TransitionResult::Denied { .. }));
    }

    #[test]
    fn test_double_pause_denied() {
        let mut sm = ControlledStateMachine::new();
        sm.transition(State::Paused);
        let result = sm.transition(State::Paused);
        assert!(matches!(result, TransitionResult::Denied { .. }));
    }

    #[test]
    fn test_denied_reason_contains_info() {
        let mut sm = ControlledStateMachine::new();
        if let TransitionResult::Denied { from, to, reason } = sm.transition(State::Done) {
            assert_eq!(from, State::Draft);
            assert_eq!(to, State::Done);
            assert!(reason.contains("not allowed"));
        } else {
            panic!("Expected Denied");
        }
    }

    #[test]
    fn test_allowed_from_current() {
        let sm = ControlledStateMachine::new();
        let allowed = sm.allowed_from_current();
        assert!(allowed.contains(&State::PlanDefined));
        assert!(allowed.contains(&State::Paused));
        assert!(!allowed.contains(&State::Done));
    }

    #[test]
    fn test_transition_after_done_denied() {
        let mut sm = ControlledStateMachine::new();
        sm.transition(State::PlanDefined);
        sm.transition(State::PlanApproved);
        sm.transition(State::InProgress);
        sm.transition(State::Implemented);
        sm.transition(State::Validated);
        sm.transition(State::Done);

        // No transitions from Done
        let result = sm.transition(State::Draft);
        assert!(matches!(result, TransitionResult::Denied { .. }));
    }
}
