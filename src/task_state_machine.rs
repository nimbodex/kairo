//! Task State Machine module.
//!
//! Models a task lifecycle as a finite state machine with phases:
//! Planning → Execution → Validation → Done.
//! Supports pausing at any phase and resuming without losing context.

use std::fmt;

/// The phases a task can be in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskPhase {
    Planning,
    Execution,
    Validation,
    Done,
}

impl fmt::Display for TaskPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskPhase::Planning => write!(f, "Planning"),
            TaskPhase::Execution => write!(f, "Execution"),
            TaskPhase::Validation => write!(f, "Validation"),
            TaskPhase::Done => write!(f, "Done"),
        }
    }
}

/// Expected action the agent should perform in the current phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpectedAction {
    DefinePlan,
    ApprovePlan,
    Implement,
    Validate,
    Finalize,
    None,
}

impl fmt::Display for ExpectedAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExpectedAction::DefinePlan => write!(f, "Define the plan"),
            ExpectedAction::ApprovePlan => write!(f, "Approve the plan"),
            ExpectedAction::Implement => write!(f, "Implement the solution"),
            ExpectedAction::Validate => write!(f, "Validate the result"),
            ExpectedAction::Finalize => write!(f, "Finalize the task"),
            ExpectedAction::None => write!(f, "No action required"),
        }
    }
}

/// A snapshot of the task state, used for pause/resume.
#[derive(Debug, Clone)]
pub struct TaskSnapshot {
    pub phase: TaskPhase,
    pub step: usize,
    pub expected_action: ExpectedAction,
    pub paused: bool,
    pub notes: Vec<String>,
}

/// The task state machine.
#[derive(Debug)]
pub struct TaskStateMachine {
    phase: TaskPhase,
    step: usize,
    expected_action: ExpectedAction,
    paused: bool,
    /// Notes accumulated during the task (e.g., plan details, validation results).
    notes: Vec<String>,
    /// History of phase transitions.
    transition_log: Vec<(TaskPhase, TaskPhase)>,
}

impl TaskStateMachine {
    /// Creates a new task in the Planning phase.
    pub fn new() -> Self {
        Self {
            phase: TaskPhase::Planning,
            step: 1,
            expected_action: ExpectedAction::DefinePlan,
            paused: false,
            notes: Vec::new(),
            transition_log: Vec::new(),
        }
    }

    /// Returns the current phase.
    pub fn phase(&self) -> TaskPhase {
        self.phase
    }

    /// Returns the current step number within the phase.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Returns the expected action for the current state.
    pub fn expected_action(&self) -> &ExpectedAction {
        &self.expected_action
    }

    /// Returns whether the task is paused.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Returns whether the task is completed.
    pub fn is_done(&self) -> bool {
        self.phase == TaskPhase::Done
    }

    /// Advances to the next phase. Returns an error if the transition is invalid.
    pub fn advance(&mut self) -> Result<TaskPhase, String> {
        if self.paused {
            return Err("Cannot advance: task is paused. Resume first.".to_string());
        }
        if self.phase == TaskPhase::Done {
            return Err("Task is already done.".to_string());
        }

        let old_phase = self.phase;
        let new_phase = match self.phase {
            TaskPhase::Planning => TaskPhase::Execution,
            TaskPhase::Execution => TaskPhase::Validation,
            TaskPhase::Validation => TaskPhase::Done,
            TaskPhase::Done => unreachable!(),
        };

        self.transition_log.push((old_phase, new_phase));
        self.phase = new_phase;
        self.step = 1;
        self.expected_action = match new_phase {
            TaskPhase::Planning => ExpectedAction::DefinePlan,
            TaskPhase::Execution => ExpectedAction::Implement,
            TaskPhase::Validation => ExpectedAction::Validate,
            TaskPhase::Done => ExpectedAction::None,
        };

        Ok(new_phase)
    }

    /// Increments the step counter within the current phase.
    pub fn next_step(&mut self) {
        self.step += 1;
    }

    /// Pauses the task at the current phase/step.
    pub fn pause(&mut self) -> TaskSnapshot {
        self.paused = true;
        self.snapshot()
    }

    /// Resumes from a paused state. The task continues from exactly where it was paused.
    pub fn resume(&mut self) -> Result<(), String> {
        if !self.paused {
            return Err("Task is not paused.".to_string());
        }
        self.paused = false;
        Ok(())
    }

    /// Adds a note to the task log.
    pub fn add_note(&mut self, note: &str) {
        self.notes.push(note.to_string());
    }

    /// Returns all accumulated notes.
    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    /// Returns the full transition log.
    pub fn transition_log(&self) -> &[(TaskPhase, TaskPhase)] {
        &self.transition_log
    }

    /// Takes a snapshot of the current state.
    pub fn snapshot(&self) -> TaskSnapshot {
        TaskSnapshot {
            phase: self.phase,
            step: self.step,
            expected_action: self.expected_action.clone(),
            paused: self.paused,
            notes: self.notes.clone(),
        }
    }

    /// Builds context string for the assistant prompt (so it knows the current state).
    pub fn context_prompt(&self) -> String {
        let pause_info = if self.paused { " [PAUSED]" } else { "" };
        format!(
            "Task state: phase={}, step={}, expected_action=\"{}\"{}\nNotes so far: {}",
            self.phase,
            self.step,
            self.expected_action,
            pause_info,
            if self.notes.is_empty() {
                "(none)".to_string()
            } else {
                self.notes.join("; ")
            }
        )
    }
}

impl Default for TaskStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let task = TaskStateMachine::new();
        assert_eq!(task.phase(), TaskPhase::Planning);
        assert_eq!(task.step(), 1);
        assert_eq!(*task.expected_action(), ExpectedAction::DefinePlan);
        assert!(!task.is_paused());
        assert!(!task.is_done());
    }

    #[test]
    fn test_full_lifecycle() {
        let mut task = TaskStateMachine::new();

        assert_eq!(task.advance().unwrap(), TaskPhase::Execution);
        assert_eq!(*task.expected_action(), ExpectedAction::Implement);

        assert_eq!(task.advance().unwrap(), TaskPhase::Validation);
        assert_eq!(*task.expected_action(), ExpectedAction::Validate);

        assert_eq!(task.advance().unwrap(), TaskPhase::Done);
        assert_eq!(*task.expected_action(), ExpectedAction::None);
        assert!(task.is_done());
    }

    #[test]
    fn test_cannot_advance_past_done() {
        let mut task = TaskStateMachine::new();
        task.advance().unwrap();
        task.advance().unwrap();
        task.advance().unwrap();

        let result = task.advance();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Task is already done.");
    }

    #[test]
    fn test_pause_and_resume() {
        let mut task = TaskStateMachine::new();
        task.advance().unwrap(); // → Execution
        task.next_step();
        task.next_step(); // step = 3

        let snapshot = task.pause();
        assert!(task.is_paused());
        assert_eq!(snapshot.phase, TaskPhase::Execution);
        assert_eq!(snapshot.step, 3);

        // Cannot advance while paused
        let result = task.advance();
        assert!(result.is_err());

        // Resume and continue
        task.resume().unwrap();
        assert!(!task.is_paused());
        assert_eq!(task.phase(), TaskPhase::Execution);
        assert_eq!(task.step(), 3); // step preserved

        // Can advance after resume
        assert_eq!(task.advance().unwrap(), TaskPhase::Validation);
    }

    #[test]
    fn test_resume_when_not_paused() {
        let mut task = TaskStateMachine::new();
        let result = task.resume();
        assert!(result.is_err());
    }

    #[test]
    fn test_notes() {
        let mut task = TaskStateMachine::new();
        task.add_note("Plan: use microservices architecture");
        task.add_note("Constraint: must support PostgreSQL");
        assert_eq!(task.notes().len(), 2);
    }

    #[test]
    fn test_transition_log() {
        let mut task = TaskStateMachine::new();
        task.advance().unwrap();
        task.advance().unwrap();

        let log = task.transition_log();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0], (TaskPhase::Planning, TaskPhase::Execution));
        assert_eq!(log[1], (TaskPhase::Execution, TaskPhase::Validation));
    }

    #[test]
    fn test_step_counter() {
        let mut task = TaskStateMachine::new();
        assert_eq!(task.step(), 1);
        task.next_step();
        assert_eq!(task.step(), 2);
        task.next_step();
        assert_eq!(task.step(), 3);

        // Advancing resets step
        task.advance().unwrap();
        assert_eq!(task.step(), 1);
    }

    #[test]
    fn test_context_prompt() {
        let mut task = TaskStateMachine::new();
        task.add_note("Use REST API");
        let prompt = task.context_prompt();
        assert!(prompt.contains("Planning"));
        assert!(prompt.contains("Define the plan"));
        assert!(prompt.contains("Use REST API"));
    }

    #[test]
    fn test_pause_preserves_notes() {
        let mut task = TaskStateMachine::new();
        task.add_note("Important decision");
        let snapshot = task.pause();
        assert_eq!(snapshot.notes, vec!["Important decision"]);

        task.resume().unwrap();
        assert_eq!(task.notes(), &["Important decision"]);
    }
}
