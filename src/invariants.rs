//! Invariants and state constraints module.
//!
//! Stores invariants (architectural decisions, tech stack constraints,
//! business rules) separately from the dialog. Validates proposals
//! against invariants and rejects violations with explanations.

use std::fmt;

/// Category of an invariant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InvariantCategory {
    Architecture,
    TechStack,
    BusinessRule,
    Custom(String),
}

impl fmt::Display for InvariantCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvariantCategory::Architecture => write!(f, "Architecture"),
            InvariantCategory::TechStack => write!(f, "TechStack"),
            InvariantCategory::BusinessRule => write!(f, "BusinessRule"),
            InvariantCategory::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// A single invariant constraint.
#[derive(Debug, Clone)]
pub struct Invariant {
    pub id: String,
    pub category: InvariantCategory,
    pub description: String,
    /// Keywords that signal a potential violation.
    pub violation_keywords: Vec<String>,
}

impl Invariant {
    pub fn new(
        id: &str,
        category: InvariantCategory,
        description: &str,
        violation_keywords: Vec<&str>,
    ) -> Self {
        Self {
            id: id.to_string(),
            category,
            description: description.to_string(),
            violation_keywords: violation_keywords.into_iter().map(String::from).collect(),
        }
    }
}

/// A violation found when checking a proposal against invariants.
#[derive(Debug, Clone)]
pub struct Violation {
    pub invariant_id: String,
    pub category: InvariantCategory,
    pub invariant_description: String,
    pub reason: String,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] Invariant '{}' violated: {}. Constraint: {}",
            self.category, self.invariant_id, self.reason, self.invariant_description
        )
    }
}

/// Result of validating a proposal against invariants.
#[derive(Debug)]
pub enum ValidationResult {
    /// No violations found.
    Ok,
    /// One or more violations detected.
    Rejected(Vec<Violation>),
}

impl ValidationResult {
    pub fn is_ok(&self) -> bool {
        matches!(self, ValidationResult::Ok)
    }

    pub fn violations(&self) -> &[Violation] {
        match self {
            ValidationResult::Ok => &[],
            ValidationResult::Rejected(v) => v,
        }
    }
}

/// The invariant store. Holds all invariants and validates proposals against them.
#[derive(Debug, Default)]
pub struct InvariantStore {
    invariants: Vec<Invariant>,
}

impl InvariantStore {
    pub fn new() -> Self {
        Self {
            invariants: Vec::new(),
        }
    }

    /// Adds an invariant to the store.
    pub fn add(&mut self, invariant: Invariant) {
        self.invariants.push(invariant);
    }

    /// Removes an invariant by id.
    pub fn remove(&mut self, id: &str) -> bool {
        let len_before = self.invariants.len();
        self.invariants.retain(|inv| inv.id != id);
        self.invariants.len() < len_before
    }

    /// Returns all stored invariants.
    pub fn all(&self) -> &[Invariant] {
        &self.invariants
    }

    /// Returns invariants by category.
    pub fn by_category(&self, category: &InvariantCategory) -> Vec<&Invariant> {
        self.invariants
            .iter()
            .filter(|inv| &inv.category == category)
            .collect()
    }

    /// Validates a proposal text against all invariants.
    /// Uses keyword matching to detect potential violations.
    pub fn validate(&self, proposal: &str) -> ValidationResult {
        let proposal_lower = proposal.to_lowercase();
        let mut violations = Vec::new();

        for invariant in &self.invariants {
            for keyword in &invariant.violation_keywords {
                if proposal_lower.contains(&keyword.to_lowercase()) {
                    violations.push(Violation {
                        invariant_id: invariant.id.clone(),
                        category: invariant.category.clone(),
                        invariant_description: invariant.description.clone(),
                        reason: format!(
                            "Proposal contains '{}' which conflicts with this invariant",
                            keyword
                        ),
                    });
                    break; // one violation per invariant is enough
                }
            }
        }

        if violations.is_empty() {
            ValidationResult::Ok
        } else {
            ValidationResult::Rejected(violations)
        }
    }

    /// Builds a context block for the assistant prompt listing all active invariants.
    pub fn context_prompt(&self) -> String {
        if self.invariants.is_empty() {
            return "No active invariants.".to_string();
        }

        let mut lines = vec!["Active invariants (must not be violated):".to_string()];
        for inv in &self.invariants {
            lines.push(format!("  - [{}] {}: {}", inv.category, inv.id, inv.description));
        }
        lines.join("\n")
    }

    /// Builds a refusal explanation when violations are found.
    pub fn explain_refusal(violations: &[Violation]) -> String {
        let mut lines = vec![
            "The proposed solution cannot be accepted because it violates the following invariants:"
                .to_string(),
        ];
        for v in violations {
            lines.push(format!("  - {}", v));
        }
        lines.push(
            "Please revise the proposal to comply with all active constraints.".to_string(),
        );
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_store() -> InvariantStore {
        let mut store = InvariantStore::new();
        store.add(Invariant::new(
            "arch-microservices",
            InvariantCategory::Architecture,
            "System must use microservices architecture",
            vec!["monolith", "monolithic"],
        ));
        store.add(Invariant::new(
            "tech-postgres",
            InvariantCategory::TechStack,
            "Database must be PostgreSQL",
            vec!["mysql", "mongodb", "sqlite"],
        ));
        store.add(Invariant::new(
            "biz-gdpr",
            InvariantCategory::BusinessRule,
            "All user data must be GDPR compliant",
            vec!["without consent", "skip gdpr", "ignore privacy"],
        ));
        store
    }

    #[test]
    fn test_valid_proposal() {
        let store = sample_store();
        let result = store.validate(
            "Let's build a microservice that stores user data in PostgreSQL with full GDPR compliance.",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_architecture_violation() {
        let store = sample_store();
        let result = store.validate("I suggest we build this as a monolithic application for simplicity.");
        assert!(!result.is_ok());
        assert_eq!(result.violations().len(), 1);
        assert_eq!(result.violations()[0].invariant_id, "arch-microservices");
    }

    #[test]
    fn test_tech_stack_violation() {
        let store = sample_store();
        let result = store.validate("We should use MongoDB for better flexibility.");
        assert!(!result.is_ok());
        assert_eq!(result.violations()[0].invariant_id, "tech-postgres");
    }

    #[test]
    fn test_business_rule_violation() {
        let store = sample_store();
        let result = store.validate("We can skip GDPR checks for internal users.");
        assert!(!result.is_ok());
        assert_eq!(result.violations()[0].invariant_id, "biz-gdpr");
    }

    #[test]
    fn test_multiple_violations() {
        let store = sample_store();
        let result = store.validate(
            "Build a monolithic app with MySQL and skip GDPR for speed.",
        );
        assert!(!result.is_ok());
        assert_eq!(result.violations().len(), 3);
    }

    #[test]
    fn test_case_insensitive() {
        let store = sample_store();
        let result = store.validate("Use MONGODB for the database.");
        assert!(!result.is_ok());
    }

    #[test]
    fn test_add_and_remove() {
        let mut store = InvariantStore::new();
        store.add(Invariant::new(
            "test-inv",
            InvariantCategory::Custom("Test".to_string()),
            "Test invariant",
            vec!["forbidden"],
        ));
        assert_eq!(store.all().len(), 1);

        assert!(store.remove("test-inv"));
        assert_eq!(store.all().len(), 0);

        // Removing non-existent returns false
        assert!(!store.remove("nonexistent"));
    }

    #[test]
    fn test_by_category() {
        let store = sample_store();
        let arch = store.by_category(&InvariantCategory::Architecture);
        assert_eq!(arch.len(), 1);
        let biz = store.by_category(&InvariantCategory::BusinessRule);
        assert_eq!(biz.len(), 1);
    }

    #[test]
    fn test_context_prompt() {
        let store = sample_store();
        let prompt = store.context_prompt();
        assert!(prompt.contains("microservices"));
        assert!(prompt.contains("PostgreSQL"));
        assert!(prompt.contains("GDPR"));
    }

    #[test]
    fn test_empty_store_validates_anything() {
        let store = InvariantStore::new();
        let result = store.validate("Do anything you want with monolith and MySQL.");
        assert!(result.is_ok());
    }

    #[test]
    fn test_explain_refusal() {
        let store = sample_store();
        if let ValidationResult::Rejected(violations) =
            store.validate("Use a monolithic architecture")
        {
            let explanation = InvariantStore::explain_refusal(&violations);
            assert!(explanation.contains("cannot be accepted"));
            assert!(explanation.contains("monolith"));
        } else {
            panic!("Expected violation");
        }
    }
}
