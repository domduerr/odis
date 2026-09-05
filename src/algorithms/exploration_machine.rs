//! State machine for interactive attribute exploration.

use bit_set::BitSet;

use crate::algorithms::canonical_basis;
use crate::FormalContext;

/// Current state of the attribute exploration machine.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ExplorationState {
    /// The machine is idle and has not yet started exploration.
    #[default]
    Idle,
    /// An implication has been proposed and awaits confirmation or counterexample.
    ValidatingImplication {
        /// Premise of the current implication.
        premise: BitSet,
        /// Proposed conclusion of the current implication.
        conclusion: BitSet,
    },
    /// A counterexample was offered and is awaiting submission.
    AwaitingCounterexample {
        /// Premise of the disputed implication.
        premise: BitSet,
        /// Conclusion of the disputed implication.
        conclusion: BitSet,
    },
    /// Exploration is complete: all implications have been confirmed.
    Finished,
}

/// Input event sent to the [`ExplorationMachine`].
#[derive(Debug, Clone, PartialEq)]
pub enum ExplorationInput {
    /// Begin a new exploration session.
    Start,
    /// Confirm the current implication as valid.
    Yes,
    /// Reject the current implication (a counterexample will be requested).
    No,
    /// Submit a counterexample (a new object's attribute set) to refute the current implication.
    Submit {
        /// Index-based attribute set of the proposed counterexample object.
        counterexample: BitSet,
    },
    /// Abort the exploration session.
    Stop,
}

/// Step-wise state machine for attribute exploration over a formal context.
///
/// Drives the Ganter–Wille attribute exploration algorithm interactively.
/// Each call to [`ExplorationMachine::process_input`] advances the state by one step.
#[derive(Debug, Clone, PartialEq)]
pub struct ExplorationMachine {
    /// Current state of exploration.
    pub state: ExplorationState,
    /// Implications confirmed so far (premise → conclusion, index-based).
    pub basis: Vec<(BitSet, BitSet)>,
    /// Scratch set used during implication closure computation.
    pub temp_set: BitSet,
    /// Scratch set holding the current implication closure.
    pub temp_hull: BitSet,
}

impl ExplorationMachine {
    /// Creates a new `ExplorationMachine` in the `Idle` state.
    pub fn new() -> Self {
        ExplorationMachine {
            state: ExplorationState::Idle,
            basis: Vec::new(),
            temp_set: BitSet::new(),
            temp_hull: BitSet::new(),
        }
    }

    /// Advance the machine by one input event.
    ///
    /// `context_getter` is called lazily to obtain the current formal context
    /// (useful when the context grows as counterexamples are accepted).
    /// Returns the new [`ExplorationState`] after processing `input`.
    pub fn process_input<F>(
        &mut self,
        context_getter: F,
        input: ExplorationInput,
    ) -> ExplorationState
    where
        F: Fn() -> FormalContext<String>,
    {
        match (&self.state, input) {
            (_, ExplorationInput::Stop) => {
                self.reset();
                ExplorationState::Idle
            }
            (ExplorationState::Idle, ExplorationInput::Start) => {
                self.start_exploration(context_getter)
            }
            (ExplorationState::ValidatingImplication { .. }, ExplorationInput::Yes) => {
                self.handle_yes(context_getter)
            }
            (
                ExplorationState::ValidatingImplication {
                    premise,
                    conclusion,
                },
                ExplorationInput::No,
            ) => {
                self.state = ExplorationState::AwaitingCounterexample {
                    premise: premise.clone(),
                    conclusion: conclusion.clone(),
                };
                self.state.clone()
            }
            (
                ExplorationState::AwaitingCounterexample { .. },
                ExplorationInput::Submit { counterexample },
            ) => self.handle_submit(context_getter, counterexample),
            (ExplorationState::Finished, ExplorationInput::Start) => {
                self.reset();
                self.start_exploration(context_getter)
            }
            _ => self.state.clone(),
        }
    }

    fn start_exploration<F>(&mut self, context_getter: F) -> ExplorationState
    where
        F: Fn() -> FormalContext<String>,
    {
        let context = context_getter();
        let all_attributes: BitSet = (0..context.attributes.len()).collect();
        self.temp_set = BitSet::new();
        self.temp_hull = BitSet::new();

        self.explore_next(context_getter, all_attributes)
    }

    fn explore_next<F>(&mut self, context_getter: F, all_attributes: BitSet) -> ExplorationState
    where
        F: Fn() -> FormalContext<String>,
    {
        loop {
            let context = context_getter();

            if self.temp_set == all_attributes {
                self.state = ExplorationState::Finished;
                return ExplorationState::Finished;
            }

            self.temp_hull = context.index_attribute_hull(&self.temp_set);

            if self.temp_set != self.temp_hull {
                self.state = ExplorationState::ValidatingImplication {
                    premise: self.temp_set.clone(),
                    conclusion: self.temp_hull.clone(),
                };
                return self.state.clone();
            } else {
                self.temp_set =
                    canonical_basis::index_next_preclosure(&context, &self.basis, &self.temp_set);
            }
        }
    }

    fn handle_yes<F>(&mut self, context_getter: F) -> ExplorationState
    where
        F: Fn() -> FormalContext<String>,
    {
        if let ExplorationState::ValidatingImplication {
            premise: _,
            conclusion: _,
        } = &self.state
        {
            self.basis
                .push((self.temp_set.clone(), self.temp_hull.clone()));
            let context = context_getter();
            self.temp_set =
                canonical_basis::index_next_preclosure(&context, &self.basis, &self.temp_set);
            self.explore_next(context_getter, (0..context.attributes.len()).collect())
        } else {
            self.state.clone()
        }
    }

    fn handle_submit<F>(&mut self, context_getter: F, _counterexample: BitSet) -> ExplorationState
    where
        F: Fn() -> FormalContext<String>,
    {
        let context = context_getter();
        let all_attributes: BitSet = (0..context.attributes.len()).collect();
        self.explore_next(context_getter, all_attributes)
    }

    /// Reset the machine to the initial `Idle` state, clearing all accumulated state.
    pub fn reset(&mut self) {
        self.state = ExplorationState::Idle;
        self.basis = Vec::new();
        self.temp_set = BitSet::new();
        self.temp_hull = BitSet::new();
    }
}

impl Default for ExplorationMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{ExplorationInput, ExplorationMachine, ExplorationState};
    use crate::FormalContext;
    use std::fs;

    fn load_context() -> FormalContext<String> {
        FormalContext::<String>::from(
            &fs::read("test_data/living_beings_and_water.cxt").unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn exploration_machine_drives_to_finished() {
        let mut machine = ExplorationMachine::new();

        // Start exploration
        let state = machine.process_input(load_context, ExplorationInput::Start);

        // The context is non-trivial, so we should get an implication to validate
        assert_ne!(state, ExplorationState::Idle);

        // Confirm all implications as valid until exploration finishes
        let mut iterations = 0;
        loop {
            match machine.state.clone() {
                ExplorationState::ValidatingImplication { .. } => {
                    machine.process_input(load_context, ExplorationInput::Yes);
                }
                ExplorationState::Finished => break,
                other => panic!("Unexpected state: {:?}", other),
            }
            iterations += 1;
            assert!(iterations < 1000, "exploration did not terminate");
        }

        assert_eq!(machine.state, ExplorationState::Finished);
        assert!(!machine.basis.is_empty(), "basis should be non-empty for a non-trivial context");
    }
}
