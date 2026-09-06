//! Access to the [FCA repository](https://fcarepository.org/) of published formal contexts.
//!
//! The repository is a collection of formal contexts that have appeared in the FCA
//! literature. It is served as plain files from GitHub: a catalogue, `contexts.yaml`,
//! describing every dataset, and one Burmeister (`.cxt`) file per context.
//!
//! Downloading is async: this module also compiles for `wasm32`, where the only
//! available transport is the browser's fetch API, which has no blocking form.
//!
//! ```no_run
//! # async fn run() -> Result<(), odis::repository::RepositoryError> {
//! for entry in odis::repository::fetch_catalog().await? {
//!     println!("{} — {}", entry.filename, entry.title);
//! }
//!
//! let ctx = odis::repository::fetch_context("livingbeings_en.cxt").await?;
//! assert_eq!(ctx.objects.len(), 8);
//! # Ok(())
//! # }
//! ```
//!
//! [`parse_catalog`] and [`context_url`] are also public on their own, for a caller
//! that has already obtained the bytes some other way.

use crate::data_structures::formal_context::{FormalContext, FormatError};

/// URL of the repository catalogue, a YAML file describing every available context.
pub const CATALOG_URL: &str =
    "https://raw.githubusercontent.com/fcatools/contexts/main/contexts.yaml";

/// Prefix the individual context files are served under.
const CONTEXT_BASE_URL: &str = "https://raw.githubusercontent.com/fcatools/contexts/main/contexts/";

/// Error returned when the repository cannot be read.
#[derive(Debug)]
pub enum RepositoryError {
    /// The catalogue is not valid UTF-8, or contains no entries.
    InvalidCatalog,
    /// A downloaded context file is not valid Burmeister format.
    Format(FormatError),
    /// The download failed. Carries the underlying client's message.
    Http(String),
}

impl From<FormatError> for RepositoryError {
    fn from(err: FormatError) -> RepositoryError {
        RepositoryError::Format(err)
    }
}

impl std::fmt::Display for RepositoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RepositoryError::InvalidCatalog => {
                write!(f, "the repository catalogue could not be parsed")
            }
            RepositoryError::Format(err) => {
                write!(f, "the context file is not valid Burmeister format: {err:?}")
            }
            RepositoryError::Http(message) => write!(f, "the download failed: {message}"),
        }
    }
}

impl std::error::Error for RepositoryError {}

/// One dataset in the repository catalogue.
///
/// Every field but [`filename`](Self::filename) and [`title`](Self::title) is optional
/// in the catalogue and may come back empty.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RepositoryEntry {
    /// File name of the context, e.g. `livingbeings_en.cxt`. Pass to [`fetch_context`].
    pub filename: String,
    /// Human-readable name of the dataset.
    pub title: String,
    /// Bibliographic references the context was published in.
    pub source: Vec<String>,
    /// Number of objects, as stated by the catalogue.
    pub objects: Option<usize>,
    /// Number of attributes, as stated by the catalogue.
    pub attributes: Option<usize>,
    /// Language the object and attribute names are written in.
    pub language: Option<String>,
    /// Short description of what the context is about.
    pub description: Option<String>,
    /// Further remarks, e.g. how this context relates to another one.
    pub note: Vec<String>,
}

impl RepositoryEntry {
    /// URL the context file is downloaded from.
    pub fn url(&self) -> String {
        context_url(&self.filename)
    }

    /// Assigns a catalogue field. Unknown keys are ignored, so that a catalogue
    /// gaining new fields keeps parsing.
    fn set(&mut self, key: &str, value: String) {
        match key {
            "title" => self.title = value,
            "source" => self.source.push(value),
            "objects" => self.objects = value.parse().ok(),
            "attributes" => self.attributes = value.parse().ok(),
            "language" => self.language = Some(value),
            "description" => self.description = Some(value),
            "note" => self.note.push(value),
            _ => {}
        }
    }
}

/// URL the given context file is downloaded from.
///
/// # Examples
///
/// ```
/// use odis::repository::context_url;
///
/// assert_eq!(
///     context_url("livingbeings_en.cxt"),
///     "https://raw.githubusercontent.com/fcatools/contexts/main/contexts/livingbeings_en.cxt"
/// );
/// ```
pub fn context_url(filename: &str) -> String {
    format!("{CONTEXT_BASE_URL}{filename}")
}

/// Splits `key: value`, but only where the key looks like a catalogue field name.
///
/// This keeps a quoted scalar that happens to contain a colon — several sources do,
/// e.g. `"Mahn, M. (2014). Gewürze: Das Standardwerk"` — from being read as a field.
fn split_field(line: &str) -> Option<(&str, &str)> {
    let (key, value) = line.split_once(':')?;
    if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphabetic() || c == '_') {
        return None;
    }
    Some((key, value.trim()))
}

/// Strips the surrounding double quotes YAML scalars in the catalogue may carry.
fn unquote(value: &str) -> String {
    value
        .strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .unwrap_or(value)
        .to_string()
}

/// Parses the repository catalogue.
///
/// Use this when the bytes were obtained elsewhere; [`fetch_catalog`] downloads and
/// parses in one step.
///
/// The catalogue is generated, and its entries take one of two shapes: a field is
/// either given directly, or as a list of items below the field name. Both are accepted,
/// as is a missing or empty field.
///
/// # Examples
///
/// ```
/// use odis::repository::parse_catalog;
///
/// let catalogue = b"livingbeings_en.cxt:\n  title: Living beings and water\n  size:\n  objects: 8\n  attributes: 9\n";
/// let entries = parse_catalog(catalogue).unwrap();
///
/// assert_eq!(entries.len(), 1);
/// assert_eq!(entries[0].title, "Living beings and water");
/// assert_eq!(entries[0].objects, Some(8));
/// ```
pub fn parse_catalog(contents: &[u8]) -> Result<Vec<RepositoryEntry>, RepositoryError> {
    let text = std::str::from_utf8(contents).map_err(|_| RepositoryError::InvalidCatalog)?;

    let mut entries: Vec<RepositoryEntry> = Vec::new();
    // Which field a bare list item below it belongs to, e.g. the two `- "..."` lines
    // that follow a `source:` with no value of its own.
    let mut pending_key = String::new();

    for line in text.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }

        if !line.starts_with(' ') {
            let filename = line
                .strip_suffix(':')
                .ok_or(RepositoryError::InvalidCatalog)?;
            entries.push(RepositoryEntry {
                filename: filename.to_string(),
                ..Default::default()
            });
            pending_key.clear();
            continue;
        }

        let entry = entries.last_mut().ok_or(RepositoryError::InvalidCatalog)?;
        let body = line.trim_start();

        if let Some(item) = body.strip_prefix("- ") {
            match split_field(item) {
                // `- objects: 8`
                Some((key, value)) => entry.set(key, unquote(value)),
                // `- "Ganter, B., & Wille, R. (1996). Formale Begriffsanalyse."`
                None => entry.set(&pending_key, unquote(item)),
            }
        } else {
            let (key, value) = split_field(body).ok_or(RepositoryError::InvalidCatalog)?;
            pending_key = key.to_string();
            if !value.is_empty() {
                entry.set(key, unquote(value));
            }
        }
    }

    if entries.is_empty() {
        return Err(RepositoryError::InvalidCatalog);
    }
    Ok(entries)
}

/// Downloads and parses the repository catalogue.
///
/// ```no_run
/// # async fn run() -> Result<(), odis::repository::RepositoryError> {
/// let entries = odis::repository::fetch_catalog().await?;
/// println!("{} contexts available", entries.len());
/// # Ok(())
/// # }
/// ```
pub async fn fetch_catalog() -> Result<Vec<RepositoryEntry>, RepositoryError> {
    parse_catalog(fetch_text(CATALOG_URL).await?.as_bytes())
}

/// Downloads a context by its catalogue file name, e.g. `livingbeings_en.cxt`.
///
/// ```no_run
/// # async fn run() -> Result<(), odis::repository::RepositoryError> {
/// let ctx = odis::repository::fetch_context("livingbeings_en.cxt").await?;
/// assert_eq!(ctx.attributes.len(), 9);
/// # Ok(())
/// # }
/// ```
pub async fn fetch_context(filename: &str) -> Result<FormalContext<String>, RepositoryError> {
    let body = fetch_text(&context_url(filename)).await?;
    Ok(FormalContext::<String>::from(body.as_bytes())?)
}

/// Downloads a URL as text. On `wasm32` reqwest routes this through the browser's
/// fetch API, which is why the whole module is async rather than blocking.
async fn fetch_text(url: &str) -> Result<String, RepositoryError> {
    let to_error = |e: reqwest::Error| RepositoryError::Http(e.to_string());

    reqwest::get(url)
        .await
        .map_err(to_error)?
        .error_for_status()
        .map_err(to_error)?
        .text()
        .await
        .map_err(to_error)
}

#[cfg(test)]
mod tests {
    use super::{context_url, parse_catalog, RepositoryError};
    use std::fs;

    fn catalog() -> Vec<super::RepositoryEntry> {
        parse_catalog(&fs::read("test_data/contexts.yaml").unwrap()).unwrap()
    }

    #[test]
    fn test_parse_catalog() {
        let entries = catalog();
        assert_eq!(entries.len(), 20);
        assert_eq!(entries[0].filename, "animals_en.cxt");
        assert_eq!(entries[0].title, "Animals");
        assert_eq!(entries[0].objects, Some(35));
        assert_eq!(entries[0].attributes, Some(11));
        assert_eq!(entries[0].language.as_deref(), Some("English"));
        assert_eq!(
            entries[0].description.as_deref(),
            Some("animals and their characteristics")
        );
    }

    #[test]
    fn test_parse_catalog_accepts_both_field_shapes() {
        let entries = catalog();

        // `objects:` / `attributes:` given directly.
        let animals = entries.iter().find(|e| e.filename == "animals_en.cxt").unwrap();
        assert_eq!((animals.objects, animals.attributes), (Some(35), Some(11)));

        // The same fields given as a list below `size:`.
        let triangles = entries.iter().find(|e| e.filename == "triangles_en.cxt").unwrap();
        assert_eq!((triangles.objects, triangles.attributes), (Some(7), Some(7)));
    }

    #[test]
    fn test_parse_catalog_collects_multiple_sources() {
        let entries = catalog();

        let water = entries.iter().find(|e| e.filename == "bodiesofwater_de.cxt").unwrap();
        assert_eq!(water.source.len(), 3);
        assert!(water.source[0].starts_with("Wille, R. (1984)"));

        let drive = entries.iter().find(|e| e.filename == "driveconcepts_en.cxt").unwrap();
        assert_eq!(drive.note.len(), 2);
    }

    #[test]
    fn test_parse_catalog_keeps_colons_inside_quoted_scalars() {
        let entries = catalog();
        let spices = entries.iter().find(|e| e.filename == "seasoningplanner_de.cxt").unwrap();
        assert_eq!(
            spices.source,
            vec!["Mahn, M. (2014). Gewürze: Das Standardwerk. Christian Verlag GmbH, München"]
        );
    }

    #[test]
    fn test_parse_catalog_leaves_empty_fields_unset() {
        let entries = catalog();
        let tea = entries.iter().find(|e| e.filename == "tealady.cxt").unwrap();
        assert_eq!(tea.language, None);
        assert_eq!(tea.note, Vec::<String>::new());
    }

    #[test]
    fn test_parse_catalog_rejects_non_catalog() {
        assert!(matches!(
            parse_catalog(b"<!DOCTYPE html>"),
            Err(RepositoryError::InvalidCatalog)
        ));
        assert!(matches!(
            parse_catalog(b""),
            Err(RepositoryError::InvalidCatalog)
        ));
    }

    #[test]
    fn test_entry_url() {
        let entries = catalog();
        assert_eq!(
            entries[0].url(),
            context_url("animals_en.cxt")
        );
        assert!(entries[0].url().ends_with("/contexts/animals_en.cxt"));
    }
}
