use std::process;

use parol::{build::Builder, ParolErrorReporter};
use parol_runtime::Report;

fn main() {
    // CLI equivalent is:
    // parol -f ./rocsho.par -e ./rocsho-exp.par -p ./src/rocsho_parser.rs -a ./src/rocsho_grammar_trait.rs -t RocshoGrammar -m rocsho_grammar -g
    if let Err(err) = Builder::with_explicit_output_dir("src")
        .grammar_file("rocsho.par")
        .expanded_grammar_output_file("../rocsho-exp.par")
        .parser_output_file("rocsho_parser.rs")
        .actions_output_file("rocsho_grammar_trait.rs")
        .enable_auto_generation()
        .user_type_name("RocshoGrammar")
        .user_trait_module_name("rocsho_grammar")
        .trim_parse_tree()
        .generate_parser()
    {
        ParolErrorReporter::report_error(&err, "rocsho.par").unwrap_or_default();
        process::exit(1);
    }
}
