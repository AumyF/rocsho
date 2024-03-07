mod rocsho_grammar;
mod rocsho_grammar_trait;
mod rocsho_parser;

mod interpreter;

#[cfg(test)]

mod test {
    use super::*;
    use interpreter::Properties;
    use rocsho_grammar::RocshoGrammar;
    use rocsho_parser::parse;

    #[test]
    fn 算術演算() {
        let mut grammar = RocshoGrammar::new();
        let _ = parse(r#"1 + 3"#, "test.rocsho", &mut grammar).unwrap();
        let result = grammar
            .rocsho
            .unwrap()
            .evaluate(&Properties::new())
            .unwrap()
            .wrappd_value;

        insta::assert_debug_snapshot!(result);
    }

    #[test]
    fn モジュール() {
        let mut grammar = RocshoGrammar::new();
        let _ = parse(
            r#"def powby2(x) = (
x * x
);

---

powby2 16
"#,
            "test.rocsho",
            &mut grammar,
        )
        .unwrap();
        let result = grammar
            .rocsho
            .unwrap()
            .evaluate(&Properties::new())
            .unwrap()
            .wrappd_value;

        insta::assert_debug_snapshot!(result);
    }
    #[test]
    fn パイプメソッド() {
        let mut grammar = RocshoGrammar::new();
        let _ = parse(
            r#"
def add2(n) = n + 2;
---
add2 3 * 2 |.eq 10 |. eq true
        "#,
            "test.rocsho",
            &mut grammar,
        )
        .unwrap();
        let result = grammar
            .rocsho
            .unwrap()
            .evaluate(&Properties::new())
            .unwrap()
            .wrappd_value;

        insta::assert_debug_snapshot!(result);
    }
    #[test]
    fn パイプ() {
        let mut grammar = RocshoGrammar::new();
        let _ = parse(
            r#"
def add2(n) = n + 2;
---
let a = 0 |> add2 |> add2;
a |> add2 |> add2
"#,
            "test.rocsho",
            &mut grammar,
        )
        .unwrap();
        let result = grammar
            .rocsho
            .unwrap()
            .evaluate(&Properties::new())
            .unwrap()
            .wrappd_value;

        insta::assert_debug_snapshot!(result);
    }
}
