use crate::rocsho_grammar_trait;
use rocsho_grammar_trait::*;
use std::collections::HashMap;
#[derive(Debug, Clone)]

pub struct Fn<'a> {
    parameters: Vec<String>,
    body: Expr<'a>,
    // pattern: Expr<'a>,
    env: Environment<'a>,
}

impl<'a> Fn<'a> {
    /// Evaluate `self` with its environemnt and supplied arguments
    fn evaluate_with(&self, arguments: Vec<Value<'a>>) -> EvalResult<'a> {
        let env =
            self.parameters
                .iter()
                .enumerate()
                .fold(self.env.clone(), |env, (index, name)| {
                    env.set(
                        name,
                        arguments.get(index).expect("Not enough arguments").clone(),
                    )
                });
        self.body.evaluate(&env)
    }
}

trait BindTuple<T, E> {
    fn bind_tuple<U>(self, other: impl FnOnce() -> Result<U, E>) -> Result<(T, U), E>;
}

impl<T, E> BindTuple<T, E> for Result<T, E> {
    fn bind_tuple<U>(self, other: impl FnOnce() -> Result<U, E>) -> Result<(T, U), E> {
        self.and_then(|it| Ok((it, other()?)))
    }
}

#[derive(Debug, Clone)]
pub enum Value<'a> {
    Int(i64),
    Unit,
    Fn(Fn<'a>),
}

impl<'a> Value<'a> {
    fn try_get_int(self) -> Result<i64, Self> {
        match self {
            Value::Int(i) => Ok(i),
            _ => Err(self),
        }
    }

    fn try_get_fn(self) -> Result<Fn<'a>, Self> {
        match self {
            Value::Fn(f) => Ok(f),
            _ => Err(self),
        }
    }
}

impl<'a> std::ops::Add for Value<'a> {
    type Output = EvalResult<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        let lhs = self
            .try_get_int()
            .map_err(|e| format!("expected lhs to be int but got {:?}", e));
        let rhs = rhs
            .try_get_int()
            .map_err(|e| format!("expected rhs to be int but got {:?}", e));
        lhs.and_then(|lhs| rhs.map(|rhs| Value::Int(lhs + rhs)))
    }
}
impl<'a> std::ops::Sub for Value<'a> {
    type Output = EvalResult<'a>;
    fn sub(self, rhs: Self) -> Self::Output {
        let lhs = self
            .try_get_int()
            .map_err(|e| format!("expected lhs to be int but got {:?}", e));
        let rhs = rhs
            .try_get_int()
            .map_err(|e| format!("expected rhs to be int but got {:?}", e));
        lhs.and_then(|lhs| rhs.map(|rhs| Value::Int(lhs - rhs)))
    }
}
impl<'a> std::ops::Mul for Value<'a> {
    type Output = EvalResult<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        let lhs = self
            .try_get_int()
            .map_err(|e| format!("expected lhs to be int but got {:?}", e));
        let rhs = rhs
            .try_get_int()
            .map_err(|e| format!("expected rhs to be int but got {:?}", e));
        lhs.and_then(|lhs| rhs.map(|rhs| Value::Int(lhs * rhs)))
    }
}
impl<'a> std::ops::Div for Value<'a> {
    type Output = EvalResult<'a>;
    fn div(self, rhs: Self) -> Self::Output {
        let lhs = self
            .try_get_int()
            .map_err(|e| format!("expected lhs to be int but got {:?}", e));
        let rhs = rhs
            .try_get_int()
            .map_err(|e| format!("expected rhs to be int but got {:?}", e));
        lhs.and_then(|lhs| rhs.map(|rhs| Value::Int(lhs / rhs)))
    }
}

#[derive(Debug, Clone)]
pub struct Environment<'a>(HashMap<String, Value<'a>>);

impl<'a> Environment<'a> {
    pub fn new() -> Environment<'a> {
        Environment(HashMap::new())
    }
    fn get(&self, name: &str) -> Option<&Value<'a>> {
        self.0.get(name)
    }
    fn set(&self, name: &str, value: Value<'a>) -> Environment<'a> {
        let mut new = self.clone();
        new.0.insert(name.to_string(), value);
        new
    }
}

impl<'a> DecimalIntLiteral<'a> {
    fn evaluate(&self) -> Value<'a> {
        Value::Int(
            self.decimal_int_literal
                .text()
                .parse()
                .expect("BUG: valid IntLiteral but invalid Rust i64"),
        )
    }
}

impl<'a> Rocsho<'a> {
    pub fn evaluate(&self, env: &Environment<'a>) -> EvalResult<'a> {
        let module = self.script.script_opt.clone();
        let env = module.map_or(env.clone(), |module| {
            module
                .module
                .module_list
                .iter()
                .fold(env.clone(), |env, item| {
                    item.module_item.function_declaration.evaluate(&env)
                })
        });
        let script = self.script.block_inner.evaluate(&env);
        script
    }
}

type EvalResult<'a> = Result<Value<'a>, String>;

impl<'a> Identifier<'a> {
    fn evaluate(&self, env: &Environment<'a>) -> EvalResult<'a> {
        let name = self.identifier.text();
        env.get(name)
            .cloned()
            .ok_or(format!("Identifier {name} not found."))
    }
}

impl<'a> FunctionDeclaration<'a> {
    fn evaluate(&self, env: &Environment<'a>) -> Environment<'a> {
        let env = env.clone();
        let name = self.identifier.identifier.text();

        let parameters = std::iter::once(
            self.parameter_list
                .pattern
                .identifier
                .identifier
                .text()
                .to_string(),
        )
        .chain(
            self.parameter_list
                .parameter_list_list
                .iter()
                .map(|pll| pll.pattern.identifier.identifier.text().to_string()),
        );
        let parameters = parameters.collect();

        let fnc = Value::Fn(Fn {
            parameters,
            body: *self.expr.clone(),
            env: env.clone(),
        });
        env.set(name, fnc)
    }
}

impl<'a> Expr<'a> {
    fn evaluate(&self, env: &Environment<'a>) -> EvalResult<'a> {
        self.add_sub_expr.evaluate(env)
    }
}

impl<'a> AddSubExpr<'a> {
    fn evaluate(&self, env: &Environment<'a>) -> EvalResult<'a> {
        self.add_sub_expr_list.iter().fold(
            self.add_sub_operand.mul_div_expr.evaluate(env),
            |acc, a| {
                let operands =
                    acc.bind_tuple(|| a.add_sub_operand.mul_div_expr.evaluate(env));
                match *a.add_sub_expr_list_group {
                    AddSubExprListGroup::Plus(_) => operands.and_then(|(l, r)| l + r),
                    AddSubExprListGroup::Minus(_) => operands.and_then(|(l, r)| l - r),
                }
            },
        )
    }
}

impl<'a> MulDivExpr<'a> {
    fn evaluate(&self, env: &Environment<'a>) -> EvalResult<'a> {
        self.mul_div_expr_list.iter().fold(
            self.mul_div_operand.function_application.evaluate(env),
            |acc, l| {
                let operands =
                    acc.bind_tuple(|| l.mul_div_operand.function_application.evaluate(env));
                match *l.mul_div_expr_list_group {
                    MulDivExprListGroup::Star(_) => operands.and_then(|(acc, l)| acc * l),
                    MulDivExprListGroup::Slash(_) => operands.and_then(|(acc, l)| acc / l),
                }
            },
        )
    }
}

impl<'a> BlockElement<'a> {
    fn evaluate(&self, env: &Environment<'a>) -> (EvalResult<'a>, Environment<'a>) {
        match self {
            BlockElement::Expr(e) => {
                let value = e.expr.evaluate(env);
                (value, env.clone())
            }
            BlockElement::VariableDeclaration(vd) => {
                let name = vd.variable_declaration.pattern.identifier.identifier.text();
                let value = vd.variable_declaration.expr.evaluate(env);
                // FIXME
                let vreal = value.clone().unwrap();
                (value, env.set(name, vreal))
            }
        }
    }
}

impl<'a> BlockInner<'a> {
    fn evaluate(&self, env: &Environment<'a>) -> EvalResult<'a> {
        let ve = self.block_element.evaluate(env);
        let (value, _) = self.block_inner_list.iter().fold(ve, |(_, env), e| {
            let (value, env) = e.block_element.evaluate(&env);
            (value, env)
        });
        value
    }
}

impl<'a> Block<'a> {
    fn evaluate(&self, env: &Environment<'a>) -> EvalResult<'a> {
        self.block_opt
            .as_ref()
            .map_or(Ok(Value::Unit), |bi| bi.block_inner.evaluate(env))
    }
}

impl<'a> FunctionApplication<'a> {
    fn evaluate(&self, env: &Environment<'a>) -> EvalResult<'a> {
        let f = self.primary_expression.evaluate(env);
        let appls = self.function_application_list.iter();
        let res = appls.fold(f, |f, args| {
            let args = args
                .function_application_opt
                .clone()
                .map_or(Ok(Vec::new()), |args| {
                    let a = std::iter::once(args.expr.evaluate(env)).chain(
                        args.function_application_opt_list
                            .iter()
                            .map(|arg| arg.expr.evaluate(env)),
                    );
                    a.collect()
                });

            let f = f?.try_get_fn().map_err(|e| format!("{:?}", e))?;
            f.evaluate_with(args?)
        });

        res
    }
}

impl<'a> PrimaryExpression<'a> {
    fn evaluate(&self, env: &Environment<'a>) -> EvalResult<'a> {
        match self {
            PrimaryExpression::DecimalIntLiteral(l) => Ok(l.decimal_int_literal.evaluate()),
            PrimaryExpression::Block(b) => b.block.evaluate(env),
            PrimaryExpression::Identifier(i) => i.identifier.evaluate(env),
        }
    }
}

/*
pub fn interpret(source: rocsho_grammar_trait::Rocsho, env: &Environment) -> Value {
    interpret_sequence(&source.script.block_inner, env)
}

fn interpret_def(def: FunctionDeclaration, env: &Environment) -> Value {
    Value::Fn {

        body: def.expr,
        env,
        pattern: def.parameter_list,

    }
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
}*/
