#![warn(clippy::all, rust_2018_idioms, future_incompatible)]

mod rocsho_grammar;
// The output is version controlled
mod rocsho_grammar_trait;
mod rocsho_parser;

mod interpreter;

use crate::rocsho_grammar::RocshoGrammar;
use crate::rocsho_parser::parse;
use anyhow::{anyhow, Context, Result};
use parol_runtime::{log::debug, Report};
use std::{env, fs, time::Instant};

// To generate:
// parol -f ./rocsho.par -e ./rocsho-exp.par -p ./src/rocsho_parser.rs -a ./src/rocsho_grammar_trait.rs -t RocshoGrammar -m rocsho_grammar -g

struct ErrorReporter;
impl Report for ErrorReporter {}

fn main() -> Result<()> {
    env_logger::init();
    debug!("env logger started");

    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 {
        let file_name = args[1].clone();
        let input = fs::read_to_string(file_name.clone())
            .with_context(|| format!("Can't read file {}", file_name))?;
        let mut rocsho_grammar = RocshoGrammar::new();
        let now = Instant::now();
        match parse(&input, &file_name, &mut rocsho_grammar) {
            Ok(_) => {
                let elapsed_time = now.elapsed();
                println!("Parsing took {} milliseconds.", elapsed_time.as_millis());
                if args.len() > 2 && args[2] == "-q" {
                    Ok(())
                } else {
                    println!("Success!\n{}", rocsho_grammar);

                    println!(
                        "Evaluated: {:?}",
                        rocsho_grammar.rocsho.unwrap().evaluate( &interpreter::Environment::new())
                    );
                    Ok(())
                }
            }
            Err(e) => ErrorReporter::report_error(&e, file_name),
        }
    } else {
        Err(anyhow!("Please provide a file name as first parameter!"))
    }
}
