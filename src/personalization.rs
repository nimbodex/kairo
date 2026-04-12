//! Personalization module.
//!
//! Adds user profiles on top of the memory model. Each profile describes
//! the user's preferences (response style, format, constraints) and is
//! injected into every request so the assistant adapts automatically.

use std::collections::HashMap;
use std::fmt;

/// Response style preference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResponseStyle {
    Concise,
    Detailed,
    StepByStep,
    Conversational,
}

impl fmt::Display for ResponseStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResponseStyle::Concise => write!(f, "concise"),
            ResponseStyle::Detailed => write!(f, "detailed"),
            ResponseStyle::StepByStep => write!(f, "step-by-step"),
            ResponseStyle::Conversational => write!(f, "conversational"),
        }
    }
}

/// Preferred response format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResponseFormat {
    PlainText,
    Markdown,
    BulletPoints,
    CodeFirst,
}

impl fmt::Display for ResponseFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResponseFormat::PlainText => write!(f, "plain text"),
            ResponseFormat::Markdown => write!(f, "markdown"),
            ResponseFormat::BulletPoints => write!(f, "bullet points"),
            ResponseFormat::CodeFirst => write!(f, "code-first with explanations after"),
        }
    }
}

/// Experience level affects how technical the responses should be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExperienceLevel {
    Beginner,
    Intermediate,
    Expert,
}

impl fmt::Display for ExperienceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExperienceLevel::Beginner => write!(f, "beginner"),
            ExperienceLevel::Intermediate => write!(f, "intermediate"),
            ExperienceLevel::Expert => write!(f, "expert"),
        }
    }
}

/// A user profile with preferences and constraints.
#[derive(Debug, Clone)]
pub struct UserProfile {
    pub name: String,
    pub role: Option<String>,
    pub experience_level: ExperienceLevel,
    pub preferred_style: ResponseStyle,
    pub preferred_format: ResponseFormat,
    pub language: String,
    /// Domain-specific constraints (e.g., "avoid deprecated APIs", "prefer functional style").
    pub constraints: Vec<String>,
    /// Custom key-value preferences.
    pub custom: HashMap<String, String>,
}

impl UserProfile {
    /// Creates a new profile with the given name and sensible defaults.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            role: None,
            experience_level: ExperienceLevel::Intermediate,
            preferred_style: ResponseStyle::Concise,
            preferred_format: ResponseFormat::Markdown,
            language: "en".to_string(),
            constraints: Vec::new(),
            custom: HashMap::new(),
        }
    }

    pub fn with_role(mut self, role: &str) -> Self {
        self.role = Some(role.to_string());
        self
    }

    pub fn with_experience(mut self, level: ExperienceLevel) -> Self {
        self.experience_level = level;
        self
    }

    pub fn with_style(mut self, style: ResponseStyle) -> Self {
        self.preferred_style = style;
        self
    }

    pub fn with_format(mut self, format: ResponseFormat) -> Self {
        self.preferred_format = format;
        self
    }

    pub fn with_language(mut self, lang: &str) -> Self {
        self.language = lang.to_string();
        self
    }

    pub fn with_constraint(mut self, constraint: &str) -> Self {
        self.constraints.push(constraint.to_string());
        self
    }

    pub fn with_custom(mut self, key: &str, value: &str) -> Self {
        self.custom.insert(key.to_string(), value.to_string());
        self
    }

    /// Generates a system prompt fragment from this profile.
    pub fn to_system_prompt(&self) -> String {
        let mut lines = Vec::new();

        lines.push(format!("User: {}", self.name));
        if let Some(role) = &self.role {
            lines.push(format!("Role: {}", role));
        }
        lines.push(format!("Experience level: {}", self.experience_level));
        lines.push(format!("Preferred response style: {}", self.preferred_style));
        lines.push(format!("Preferred format: {}", self.preferred_format));
        lines.push(format!("Language: {}", self.language));

        if !self.constraints.is_empty() {
            lines.push("Constraints:".to_string());
            for c in &self.constraints {
                lines.push(format!("  - {}", c));
            }
        }

        if !self.custom.is_empty() {
            lines.push("Additional preferences:".to_string());
            let mut keys: Vec<_> = self.custom.keys().collect();
            keys.sort();
            for key in keys {
                lines.push(format!("  {}: {}", key, self.custom[key]));
            }
        }

        lines.join("\n")
    }
}

/// Manages multiple user profiles and selects the active one.
#[derive(Debug)]
pub struct ProfileManager {
    profiles: HashMap<String, UserProfile>,
    active: Option<String>,
}

impl ProfileManager {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            active: None,
        }
    }

    /// Adds a profile. If it's the first one, it becomes active automatically.
    pub fn add_profile(&mut self, profile: UserProfile) {
        let name = profile.name.clone();
        let is_first = self.profiles.is_empty();
        self.profiles.insert(name.clone(), profile);
        if is_first {
            self.active = Some(name);
        }
    }

    /// Switches the active profile.
    pub fn set_active(&mut self, name: &str) -> Result<(), String> {
        if !self.profiles.contains_key(name) {
            return Err(format!("Profile '{}' not found", name));
        }
        self.active = Some(name.to_string());
        Ok(())
    }

    /// Returns the active profile.
    pub fn active_profile(&self) -> Option<&UserProfile> {
        self.active
            .as_ref()
            .and_then(|name| self.profiles.get(name))
    }

    /// Returns the system prompt for the active profile.
    pub fn active_system_prompt(&self) -> Option<String> {
        self.active_profile().map(|p| p.to_system_prompt())
    }

    /// Returns all profile names.
    pub fn profile_names(&self) -> Vec<&String> {
        self.profiles.keys().collect()
    }

    /// Removes a profile. Cannot remove the active profile.
    pub fn remove_profile(&mut self, name: &str) -> Result<(), String> {
        if self.active.as_deref() == Some(name) {
            return Err("Cannot remove the active profile. Switch to another first.".to_string());
        }
        if self.profiles.remove(name).is_none() {
            return Err(format!("Profile '{}' not found", name));
        }
        Ok(())
    }

    /// Returns a profile by name.
    pub fn get_profile(&self, name: &str) -> Option<&UserProfile> {
        self.profiles.get(name)
    }
}

impl Default for ProfileManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev_profile() -> UserProfile {
        UserProfile::new("Alice")
            .with_role("Senior Backend Developer")
            .with_experience(ExperienceLevel::Expert)
            .with_style(ResponseStyle::Concise)
            .with_format(ResponseFormat::CodeFirst)
            .with_language("en")
            .with_constraint("Avoid deprecated APIs")
            .with_constraint("Prefer async/await over callbacks")
            .with_custom("editor", "Neovim")
    }

    fn beginner_profile() -> UserProfile {
        UserProfile::new("Bob")
            .with_role("CS Student")
            .with_experience(ExperienceLevel::Beginner)
            .with_style(ResponseStyle::StepByStep)
            .with_format(ResponseFormat::Markdown)
            .with_language("en")
    }

    #[test]
    fn test_profile_creation() {
        let profile = dev_profile();
        assert_eq!(profile.name, "Alice");
        assert_eq!(profile.experience_level, ExperienceLevel::Expert);
        assert_eq!(profile.preferred_style, ResponseStyle::Concise);
        assert_eq!(profile.constraints.len(), 2);
    }

    #[test]
    fn test_system_prompt_generation() {
        let profile = dev_profile();
        let prompt = profile.to_system_prompt();

        assert!(prompt.contains("Alice"));
        assert!(prompt.contains("Senior Backend Developer"));
        assert!(prompt.contains("expert"));
        assert!(prompt.contains("concise"));
        assert!(prompt.contains("code-first"));
        assert!(prompt.contains("Avoid deprecated APIs"));
        assert!(prompt.contains("editor: Neovim"));
    }

    #[test]
    fn test_different_profiles_different_prompts() {
        let expert = dev_profile();
        let beginner = beginner_profile();

        let expert_prompt = expert.to_system_prompt();
        let beginner_prompt = beginner.to_system_prompt();

        assert_ne!(expert_prompt, beginner_prompt);
        assert!(expert_prompt.contains("expert"));
        assert!(beginner_prompt.contains("beginner"));
        assert!(expert_prompt.contains("concise"));
        assert!(beginner_prompt.contains("step-by-step"));
    }

    #[test]
    fn test_profile_manager_add_and_active() {
        let mut pm = ProfileManager::new();
        pm.add_profile(dev_profile());

        assert!(pm.active_profile().is_some());
        assert_eq!(pm.active_profile().unwrap().name, "Alice");
    }

    #[test]
    fn test_profile_manager_switch() {
        let mut pm = ProfileManager::new();
        pm.add_profile(dev_profile());
        pm.add_profile(beginner_profile());

        pm.set_active("Bob").unwrap();
        assert_eq!(pm.active_profile().unwrap().name, "Bob");
    }

    #[test]
    fn test_profile_manager_switch_nonexistent() {
        let mut pm = ProfileManager::new();
        pm.add_profile(dev_profile());
        let result = pm.set_active("Charlie");
        assert!(result.is_err());
    }

    #[test]
    fn test_profile_manager_first_auto_active() {
        let mut pm = ProfileManager::new();
        assert!(pm.active_profile().is_none());

        pm.add_profile(dev_profile());
        assert_eq!(pm.active_profile().unwrap().name, "Alice");

        // Adding second doesn't change active
        pm.add_profile(beginner_profile());
        assert_eq!(pm.active_profile().unwrap().name, "Alice");
    }

    #[test]
    fn test_remove_active_profile_denied() {
        let mut pm = ProfileManager::new();
        pm.add_profile(dev_profile());
        let result = pm.remove_profile("Alice");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("active"));
    }

    #[test]
    fn test_remove_inactive_profile() {
        let mut pm = ProfileManager::new();
        pm.add_profile(dev_profile());
        pm.add_profile(beginner_profile());
        assert!(pm.remove_profile("Bob").is_ok());
        assert_eq!(pm.profile_names().len(), 1);
    }

    #[test]
    fn test_active_system_prompt() {
        let mut pm = ProfileManager::new();
        pm.add_profile(dev_profile());
        let prompt = pm.active_system_prompt().unwrap();
        assert!(prompt.contains("Alice"));
    }

    #[test]
    fn test_empty_manager_no_prompt() {
        let pm = ProfileManager::new();
        assert!(pm.active_system_prompt().is_none());
    }

    #[test]
    fn test_minimal_profile() {
        let profile = UserProfile::new("Minimal");
        let prompt = profile.to_system_prompt();
        assert!(prompt.contains("Minimal"));
        assert!(prompt.contains("intermediate")); // default
        assert!(!prompt.contains("Constraints")); // no constraints
    }

    #[test]
    fn test_custom_preferences() {
        let profile = UserProfile::new("Test")
            .with_custom("theme", "dark")
            .with_custom("font_size", "14");
        let prompt = profile.to_system_prompt();
        assert!(prompt.contains("font_size: 14"));
        assert!(prompt.contains("theme: dark"));
    }
}
