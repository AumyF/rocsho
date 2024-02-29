use crate::rocsho_grammar_trait::{Rocsho, RocshoGrammarTrait};
#[allow(unused_imports)]
use parol_runtime::Result;
use std::fmt::{Debug, Display, Error, Formatter};

///
/// Data structure that implements the semantic actions for our Rocsho grammar
/// !Change this type as needed!
///
#[derive(Debug, Default)]
pub struct RocshoGrammar<'t> {
    pub rocsho: Option<Rocsho<'t>>,
}

impl RocshoGrammar<'_> {
    pub fn new() -> Self {
        RocshoGrammar::default()
    }
}

impl Display for Rocsho<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::result::Result<(), Error> {
        write!(f, "{:?}", self)
    }
}

impl Display for RocshoGrammar<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::result::Result<(), Error> {
        match &self.rocsho {
            Some(rocsho) => writeln!(f, "{}", rocsho),
            None => write!(f, "No parse result"),
        }
    }
}

impl<'t> RocshoGrammarTrait<'t> for RocshoGrammar<'t> {
    // !Adjust your implementation as needed!

    /// Semantic action for non-terminal 'Rocsho'
    fn rocsho(&mut self, arg: &Rocsho<'t>) -> Result<()> {
        self.rocsho = Some(arg.clone());
        Ok(())
    }
}
