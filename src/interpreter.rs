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

#[derive(Clone)]
pub struct Environment(HashMap<String, Value>);

impl Environment {
    pub fn new() -> Environment {
        Environment(HashMap::new())
    }
    fn get(&self, name: &str) -> Option<&Value> {
        self.0.get(name)
    }
    fn set(&self, name: &str, value: Value) -> Environment {
        let mut new = self.clone();
        new.0.insert(name.to_string(), value);
        new
    }
}

pub fn interpret(source: rocsho_grammar_trait::Rocsho, env: &Environment) -> Value {
    interpret_sequence(&source.script.block_inner, env)
}

fn interpret_block_element<'a>(
    source: &BlockElement,
    env: &'a Environment,
) -> (Value, Environment) {
    match source {
        BlockElement::Expr(e) => {
            let value = interpret_expr(&e.expr, env);
            (value, env.clone())
        }
        BlockElement::VariableDeclaration(vd) => {
            let name = vd.variable_declaration.pattern.identifier.identifier.text();
            let value = interpret_expr(&vd.variable_declaration.expr, env);
            (value, env.set(name, value))
        }
    }
}

fn interpret_sequence(source: &BlockInner, env: &Environment) -> Value {
    let ve = interpret_block_element(&source.block_element, env);
    let (value, _) = source.block_inner_list.iter().fold(ve, |(_, env), e| {
        let (value, env) = interpret_block_element(&e.block_element, &env);
        (value, env)
    });
    value
}

fn interpret_expr(expr: &rocsho_grammar_trait::Expr, env: &Environment) -> Value {
    interpret_add_sub_expr(&expr.add_sub_expr, env)
}

fn interpret_variable_expression(
    expr: &rocsho_grammar_trait::VariableDeclaration,
    env: &Environment,
) -> Value {
    let name = expr.pattern.identifier.identifier.text();
    let value = interpret_expr(&expr.expr, env);
    let env = env.set(name, value);
    interpret_expr(&expr.expr, &env)
}

fn interpret_add_sub_expr(expr: &AddSubExpr, env: &Environment) -> Value {
    expr.add_sub_expr_list.iter().fold(
        interpret_mul_div_expr(&expr.add_sub_operand.mul_div_expr, env),
        |acc, a| match *a.add_sub_expr_list_group {
            AddSubExprListGroup::Plus(_) => {
                let a = acc + interpret_mul_div_expr(a.add_sub_operand.mul_div_expr.as_ref(), env);
                a.unwrap()
            }
            AddSubExprListGroup::Minus(_) => {
                let a = acc - interpret_mul_div_expr(a.add_sub_operand.mul_div_expr.as_ref(), env);
                a.unwrap()
            }
        },
    )
}

fn interpret_mul_div_expr(expr: &MulDivExpr, env: &Environment) -> Value {
    expr.mul_div_expr_list.iter().fold(
        interpreet_mul_div_operand(&expr.mul_div_operand, env),
        |acc, l| match *l.mul_div_expr_list_group {
            MulDivExprListGroup::Star(_) => {
                let a = acc * interpreet_mul_div_operand(&l.mul_div_operand, env);
                a.unwrap()
            }
            MulDivExprListGroup::Slash(_) => {
                let a = acc / interpreet_mul_div_operand(&l.mul_div_operand, env);
                a.unwrap()
            }
        },
    )
}

fn interpret_block(block: &Block, env: &Environment) -> Value {
    block
        .block_opt
        .as_ref()
        .map_or(Value::Unit, |bi| interpret_sequence(&bi.block_inner, env))
}

fn interpreet_mul_div_operand(operand: &MulDivOperand, env: &Environment) -> Value {
    match operand {
        MulDivOperand::DecimalIntLiteral(l) => interpret_int_literal(&l.decimal_int_literal),
        MulDivOperand::Block(p) => interpret_block(&p.block, env),
        MulDivOperand::Identifier(i) => interpret_identifier(i.identifier.identifier.text(), env)
            .expect(&format!(
                "Identifier '{}' not found",
                i.identifier.identifier.text()
            )),
    }
}

fn interpret_identifier(name: &str, env: &Environment) -> Option<Value> {
    env.get(name).copied()
}

fn interpret_int_literal(expr: &DecimalIntLiteral) -> Value {
    Value::Int(
        expr.decimal_int_literal
            .text()
            .parse()
            .expect("valid IntLiteral but invalid Rust i64"),
    )
}

#[cfg(test)]
mod test {
    use super::interpret;
    use crate::rocsho_parser;
    #[test]
    fn h() {
        let mut r = crate::rocsho_grammar::RocshoGrammar::new();
        rocsho_parser::parse("1 + 2", "hoge", &mut r).unwrap();
        assert_eq!(
            interpret(r.rocsho.unwrap(), &super::Environment::new())
                .try_get_int()
                .unwrap(),
            3
        )
    }
}
