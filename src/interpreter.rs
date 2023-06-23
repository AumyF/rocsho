use crate::rocsho_grammar_trait;
use rocsho_grammar_trait::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub enum Value {
    Int(i64),
    Unit,
}

impl Value {
    fn try_get_int(self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(i),
            _ => None,
        }
    }
}

impl std::ops::Add for Value {
    type Output = Option<Self>;
    fn add(self, rhs: Self) -> Self::Output {
        self.try_get_int()
            .zip(rhs.try_get_int())
            .map(|(lhs, rhs)| Value::Int(lhs + rhs))
    }
}
impl std::ops::Sub for Value {
    type Output = Option<Self>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.try_get_int()
            .zip(rhs.try_get_int())
            .map(|(lhs, rhs)| Value::Int(lhs - rhs))
    }
}
impl std::ops::Mul for Value {
    type Output = Option<Self>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.try_get_int()
            .zip(rhs.try_get_int())
            .map(|(lhs, rhs)| Value::Int(lhs * rhs))
    }
}
impl std::ops::Div for Value {
    type Output = Option<Self>;
    fn div(self, rhs: Self) -> Self::Output {
        self.try_get_int()
            .zip(rhs.try_get_int())
            .map(|(lhs, rhs)| Value::Int(lhs / rhs))
    }
}
pub struct Environment(HashMap<String, Value>);

pub struct Interpreter {
    environment: Environment,
}

impl Interpreter {
    pub fn new() -> Interpreter {
        Interpreter {
            environment: Environment(HashMap::new()),
        }
    }

    pub fn interpret(&mut self, source: rocsho_grammar_trait::Rocsho) -> Value {
        self.interpret_sequence(*source.script.sequence)
    }

    fn interpret_sequence(&mut self, source: Sequence) -> Value {
        source.sequence_opt.map(|s| {
            s.sequence_opt_list
                .iter()
                .fold(self.interpret_expr(&s.expr), |_, e| {
                    self.interpret_expr(&e.expr)
                })
        }).unwrap()
    }

    fn interpret_expr(&mut self, expr: &rocsho_grammar_trait::Expr) -> Value {
        self.interpret_add_sub_expr(&expr.add_sub_expr)
    }

    fn interpret_add_sub_expr(&mut self, expr: &AddSubExpr) -> Value {
        expr.add_sub_expr_list.iter().fold(
            self.interpret_mul_div_expr(&expr.add_sub_operand.mul_div_expr),
            |acc, a| match *a.add_sub_expr_list_group {
                AddSubExprListGroup::Plus(_) => {
                    let a =
                        acc + self.interpret_mul_div_expr(a.add_sub_operand.mul_div_expr.as_ref());
                    a.unwrap()
                }
                AddSubExprListGroup::Minus(_) => {
                    let a =
                        acc - self.interpret_mul_div_expr(a.add_sub_operand.mul_div_expr.as_ref());
                    a.unwrap()
                }
            },
        )
    }

    fn interpret_mul_div_expr(&mut self, expr: &MulDivExpr) -> Value {
        expr.mul_div_expr_list.iter().fold(
            self.interpreet_mul_div_operand(&expr.mul_div_operand),
            |acc, l| match *l.mul_div_expr_list_group {
                MulDivExprListGroup::Star(_) => {
                    let a = acc * self.interpreet_mul_div_operand(&l.mul_div_operand);
                    a.unwrap()
                }
                MulDivExprListGroup::Slash(_) => {
                    let a = acc / self.interpreet_mul_div_operand(&l.mul_div_operand);
                    a.unwrap()
                }
            },
        )
    }

    fn interpreet_mul_div_operand(&mut self, operand: &MulDivOperand) -> Value {
        match operand {
            MulDivOperand::DecimalIntLiteral(l) => {
                self.interpret_int_literal(&l.decimal_int_literal)
            }
            MulDivOperand::ParenExpr(p) => {
                self.interpret_add_sub_expr(&p.paren_expr.expr.add_sub_expr)
            }
        }
    }

    fn interpret_int_literal(&mut self, expr: &DecimalIntLiteral) -> Value {
        Value::Int(
            expr.decimal_int_literal
                .text()
                .parse()
                .expect("valid IntLiteral but invalid Rust i64"),
        )
    }
}

#[cfg(test)]
mod test {
    use super::Interpreter;
    use crate::rocsho_parser;
    #[test]
    fn h() {
        let mut i = Interpreter::new();
        let mut r = crate::rocsho_grammar::RocshoGrammar::new();
        rocsho_parser::parse("1 + 2", "hoge", &mut r).unwrap();
        assert_eq!(i.interpret(r.rocsho.unwrap()).try_get_int().unwrap(), 3)
    }
}
