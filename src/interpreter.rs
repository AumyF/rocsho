use crate::rocsho_grammar_trait;
use rocsho_grammar_trait::*;
use std::collections::BTreeMap;

trait BindTuple<T, E> {
    fn bind_tuple<U>(self, other: impl FnOnce() -> Result<U, E>) -> Result<(T, U), E>;
}

impl<T, E> BindTuple<T, E> for Result<T, E> {
    fn bind_tuple<U>(self, other: impl FnOnce() -> Result<U, E>) -> Result<(T, U), E> {
        self.and_then(|it| Ok((it, other()?)))
    }
}

#[derive(Debug, Clone)]
pub struct Properties<'a>(BTreeMap<String, Value<'a>>);

impl<'a> Properties<'a> {
    pub fn new() -> Properties<'a> {
        Properties(BTreeMap::new())
    }
    pub fn get(&self, name: &str) -> Option<&Value<'a>> {
        self.0.get(name)
    }
    pub fn set(&self, name: &str, value: Value<'a>) -> Properties<'a> {
        let mut new = self.clone();
        new.0.insert(name.to_string(), value);
        new
    }
}

#[derive(Debug, Clone)]
pub struct Func<'a> {
    parameters: Vec<String>,
    body: Expr<'a>,
    // pattern: Expr<'a>,
    env: Properties<'a>,
    name: String,
}

impl<'a> Func<'a> {
    /// Evaluate `self` with its environemnt and supplied arguments
    fn evaluate_with(&self, receiver: Value<'a>, arguments: Vec<Value<'a>>) -> EvalResult<'a> {
        let env = self.env.set("self", receiver);
        let env = self
            .parameters
            .iter()
            .enumerate()
            .try_fold(env, |env, (index, name)| {
                let argument = arguments.get(index).ok_or_else(|| {
                    format!(
                        "Expected {} arguments but got {}",
                        self.parameters.len(),
                        arguments.len()
                    )
                })?;

                let env = env.set(name, argument.clone());
                // TODO investigate this
                Ok::<_, String>(env)
            })?;
        self.body.evaluate(&env)
    }
}

type Args<'a> = Vec<Value<'a>>;

#[derive(Clone)]
pub enum Method<'a> {
    UserDefined(Func<'a>),
    Builtin {
        f: fn(Value<'a>, Args<'a>) -> Result<Value<'a>, String>,
    },
}

impl<'a> Method<'a> {
    fn evaluate_with(&self, self_value: Value<'a>, args: Vec<Value<'a>>) -> EvalResult<'a> {
        match self {
            Method::Builtin { f } => f(self_value, args),
            Method::UserDefined(f) => f.evaluate_with(self_value, args),
        }
    }
}

impl<'source> std::fmt::Debug for Method<'source> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UserDefined(fnc) => write!(f, "{:#?}", fnc),
            Self::Builtin { f:fnc } => write!(f, "<builtin function>"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Methods<'a>(BTreeMap<String, Method<'a>>);

impl<'a> Methods<'a> {
    pub fn new() -> Methods<'a> {
        Methods(BTreeMap::new())
    }
    pub fn get(&self, name: &str) -> Option<&Method<'a>> {
        self.0.get(name)
    }
    pub fn set(&self, name: &str, value: Method<'a>) -> Methods<'a> {
        let mut new = self.clone();
        new.0.insert(name.to_string(), value);
        new
    }
}

#[derive(Debug, Clone, Copy)]
pub enum WrappedValue {
    Int(i64),
    Bool(bool),
    Unit,
}

#[derive(Debug, Clone)]
pub struct Value<'a> {
    properties: Properties<'a>,
    methods: Methods<'a>,
    pub wrappd_value: Option<WrappedValue>,
}

impl<'a> Value<'a> {
    fn empty() -> Self {
        Value {
            properties: Properties::new(),
            methods: Methods::new(),
            wrappd_value: None,
        }
    }

    // TODO Identify unit values
    fn unit() -> Value<'a> {
        Value {
            properties: Properties::new(),
            methods: Methods::new(),
            wrappd_value: Some(WrappedValue::Unit),
        }
    }

    fn try_get_int(&self) -> Option<i64> {
        match self.wrappd_value? {
            WrappedValue::Int(i) => Some(i),
            _ => None,
        }
    }

    fn try_get_bool(&self) -> Result<bool, Option<Value<'a>>> {
        match self.wrappd_value.ok_or(None)? {
            WrappedValue::Bool(b) => Ok(b),
            _ => Err(Some(self.clone())),
        }
    }

    fn try_call(&self, self_value: Value<'a>, arguments: Vec<Value<'a>>) -> EvalResult<'a> {
        let f = self
            .methods
            .get("call")
            .ok_or("Cannot call an objec which doesn't have 'call' method".to_string())?;
        f.evaluate_with(self_value, arguments)
    }

    fn try_call_method(
        &self,
        self_value: Value<'a>,
        method_name: &str,
        arguments: Vec<Value<'a>>,
    ) -> EvalResult<'a> {
        let f = self
            .methods
            .get(method_name)
            .ok_or(format!("This object has no '{method_name}' method"))?;
        f.evaluate_with(self_value, arguments)
    }

    fn bool(b: bool) -> Value<'a> {
        let v = Value {
            properties: Properties::new(),
            methods: Methods::new().set(
                "eq",
                Method::Builtin {
                    f: |it, args| {
                        let other = args.get(0).ok_or("Not enough arguments")?;
                        let (it, other) = it
                            .try_get_bool()
                            .map_err(|_| format!("Expected self to be a boolean"))
                            .bind_tuple(|| {
                                other
                                    .try_get_bool()
                                    .map_err(|_| format!("Expected the argument to be a boolean"))
                            })?;
                        Ok(Value::bool(it == other))
                    },
                },
            ),
            wrappd_value: Some(WrappedValue::Bool(b)),
        };
        v
    }

    fn func(f: Func<'a>) -> Value<'a> {
        let methods = Methods::new().set("call", Method::UserDefined(f));
        Value {
            properties: Properties::new(),
            methods,
            wrappd_value: None,
        }
    }

    fn int(i: i64) -> Value<'a> {
        let properties = Properties::new();
        let methods = Methods::new();
        let methods = methods
            .set(
                "eq",
                Method::Builtin {
                    f: |it, args| {
                        let other = args.get(0).ok_or("Not enough arguments")?;
                        let (it, other) = it
                            .try_get_int()
                            .ok_or(format!("Expected self to be an int"))
                            .bind_tuple(|| {
                                other
                                    .try_get_int()
                                    .ok_or("Expected arg to be an int".to_string())
                            })?;

                        let result = it == other;
                        Ok(Value::bool(result))
                    },
                },
            )
            .set(
                "cmp",
                Method::Builtin {
                    f: |it, args| {
                        let other = args.get(0).ok_or("Not enough arguments")?;
                        let (it, other) = it
                            .try_get_int()
                            .ok_or(format!("Expected self to be an int"))
                            .bind_tuple(|| {
                                other
                                    .try_get_int()
                                    .ok_or("Expected arg to be an int".to_string())
                            })?;

                        let result = it.cmp(&other) as i64;
                        Ok(Value::int(result))
                    },
                },
            );
        Value {
            properties,
            methods,
            wrappd_value: Some(WrappedValue::Int(i)),
        }
    }
}

impl<'source> std::ops::Add for Value<'source> {
    type Output = EvalResult<'source>;
    fn add(self, rhs: Self) -> Self::Output {
        let lhs = self
            .try_get_int()
            .ok_or_else(|| format!("Expected int but got {:?}", self))?;
        let rhs = rhs
            .try_get_int()
            .ok_or_else(|| format!("Expected int but got {:?}", self))?;
        Ok(Value::int(lhs + rhs))
    }
}

impl<'source> std::ops::Sub for Value<'source> {
    type Output = EvalResult<'source>;
    fn sub(self, rhs: Self) -> Self::Output {
        let lhs = self
            .try_get_int()
            .ok_or_else(|| format!("Expected int but got {:?}", self))?;
        let rhs = rhs
            .try_get_int()
            .ok_or_else(|| format!("Expected int but got {:?}", self))?;
        Ok(Value::int(lhs - rhs))
    }
}

impl<'source> std::ops::Mul for Value<'source> {
    type Output = EvalResult<'source>;
    fn mul(self, rhs: Self) -> Self::Output {
        let lhs = self
            .try_get_int()
            .ok_or_else(|| format!("Expected int but got {:?}", self))?;
        let rhs = rhs
            .try_get_int()
            .ok_or_else(|| format!("Expected int but got {:?}", self))?;
        Ok(Value::int(lhs * rhs))
    }
}

impl<'source> std::ops::Div for Value<'source> {
    type Output = EvalResult<'source>;
    fn div(self, rhs: Self) -> Self::Output {
        let lhs = self
            .try_get_int()
            .ok_or_else(|| format!("Expected int but got {:?}", self))?;
        let rhs = rhs
            .try_get_int()
            .ok_or_else(|| format!("Expected int but got {:?}", self))?;
        Ok(Value::int(lhs / rhs))
    }
}

impl<'source> Value<'source> {
    fn eq(&self, other: &Self) -> EvalResult<'source> {
        let eq = self
            .methods
            .get("eq")
            .ok_or_else(|| "Cannot compare an object which doesn't have 'eq' method")?;
        let r = eq.evaluate_with(self.clone(), vec![other.clone()])?;
        let r = r
            .try_get_bool()
            .map_err(|_| format!("'eq' must return a boolean"))?;
        Ok(Value::bool(r))
    }
    fn ne(&self, other: &Self) -> EvalResult<'source> {
        let eq = self
            .methods
            .get("eq")
            .ok_or_else(|| "Cannot compare an object which doesn't have 'eq' method")?;
        let r = eq.evaluate_with(self.clone(), vec![other.clone()])?;
        let r = r
            .try_get_bool()
            .map_err(|_| format!("'eq' must return a boolean"))?;
        Ok(Value::bool(!r))
    }
}

impl<'a> DecimalIntLiteral<'a> {
    fn evaluate(&self) -> Value<'a> {
        Value::int(
            self.decimal_int_literal
                .text()
                .parse()
                .expect("BUG: valid IntLiteral but invalid Rust i64"),
        )
    }
}

impl<'a> Rocsho<'a> {
    pub fn evaluate(&self, env: &Properties<'a>) -> EvalResult<'a> {
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
    fn evaluate(&self, env: &Properties<'a>) -> EvalResult<'a> {
        let name = self.identifier.text();
        env.get(name)
            .cloned()
            .ok_or(format!("Identifier {name} not found."))
    }
}

impl<'a> FunctionDeclaration<'a> {
    fn evaluate(&self, env: &Properties<'a>) -> Properties<'a> {
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

        let fnc = Value::func(Func {
            parameters,
            body: *self.expr.clone(),
            env: env.clone(),
            name: name.to_string(),
        });
        env.set(name, fnc)
    }
}

impl<'a> Expr<'a> {
    fn evaluate(&self, env: &Properties<'a>) -> EvalResult<'a> {
        match self {
            Expr::If(e) => e.r#if.evaluate(env),
            Expr::PipeExpr(e) => e.pipe_expr.evaluate(env),
        }
    }
}

impl<'source> PipeOperand<'source> {
    fn evaluate(&self, env: &Properties<'source>) -> EvalResult<'source> {
        self.comparation_expr.evaluate(env)
    }
}

impl<'source> PipeExpr<'source> {
    fn evaluate(&self, env: &Properties<'source>) -> EvalResult<'source> {
        self.pipe_expr_list
            .iter()
            .fold(self.pipe_operand.evaluate(env), |acc, a| {
                match *a.pipe_expr_list_group.clone() {
                    PipeExprListGroup::OrDotIdentifierPrimaryExpressionPipeExprListGroupList(a) => {
                        acc.and_then(|acc| {
                            let args: Result<Vec<_>, String> =
                                std::iter::once(a.primary_expression.evaluate(env))
                                    .chain(
                                        a.pipe_expr_list_group_list
                                            .iter()
                                            .map(|v| v.primary_expression.evaluate(env)),
                                    )
                                    .collect();

                            acc.try_call_method(acc.clone(), a.identifier.identifier.text(), args?)
                        })
                    }
                    PipeExprListGroup::PipeExprListGroupGroupPipeOperand(a) => {
                        let operands = acc.bind_tuple(|| a.pipe_operand.evaluate(env));
                        match *a.pipe_expr_list_group_group {
                            // |>
                            PipeExprListGroupGroup::OrGT(_) => {
                                operands.and_then(|(l, r)| r.try_call(r.clone(), vec![l]))
                            }
                            // <|
                            PipeExprListGroupGroup::LTOr(_) => {
                                operands.and_then(|(l, r)| l.try_call(l.clone(), vec![r]))
                            }
                        }
                    }
                }
            })
    }
}

impl<'source> ComparationExpr<'source> {
    fn evaluate(&self, env: &Properties<'source>) -> EvalResult<'source> {
        fn cmp<'source>(l: &Value<'source>, r: &Value<'source>) -> EvalResult<'source> {
            l.try_call_method(l.clone(), "cmp", vec![r.clone()])
        }
        fn lt<'source>(l: &Value<'source>, r: &Value<'source>) -> EvalResult<'source> {
            cmp(l, r)?.eq(&Value::int(-1))
        }
        fn gt<'source>(l: &Value<'source>, r: &Value<'source>) -> EvalResult<'source> {
            cmp(l, r)?.eq(&Value::int(1))
        }
        fn not<'source>(b: &Value<'source>) -> EvalResult<'source> {
            b.eq(&Value::bool(false))
        }
        self.comparation_expr_list.iter().fold(
            self.comparation_operand.add_sub_expr.evaluate(env),
            |acc, a| {
                let operands = acc.bind_tuple(|| a.comparation_operand.add_sub_expr.evaluate(env));
                match *a.comparation_expr_list_group {
                    ComparationExprListGroup::GT(_) => operands.and_then(|(l, r)| gt(&l, &r)),
                    ComparationExprListGroup::GTEqu(_) => {
                        operands.and_then(|(l, r)| not(&lt(&l, &r)?))
                    }
                    ComparationExprListGroup::LT(_) => operands.and_then(|(l, r)| lt(&l, &r)),
                    ComparationExprListGroup::LTEqu(_) => {
                        operands.and_then(|(l, r)| not(&gt(&l, &r)?))
                    }
                    ComparationExprListGroup::EquEqu(_) => operands.and_then(|(l, r)| l.eq(&r)),
                    ComparationExprListGroup::BangEqu(_) => operands.and_then(|(l, r)| l.ne(&r)),
                }
            },
        )
    }
}

impl<'a> If<'a> {
    fn evaluate(&self, env: &Properties<'a>) -> EvalResult<'a> {
        let If {
            expr: cond,
            expr0: texpr,
            expr1: fexpr,
        } = self;
        let cond = cond
            .evaluate(env)?
            .try_get_bool()
            .map_err(|e| format!("{:?}", e))?;
        if cond {
            texpr.evaluate(env)
        } else {
            fexpr.evaluate(env)
        }
    }
}

impl<'a> AddSubExpr<'a> {
    fn evaluate(&self, env: &Properties<'a>) -> EvalResult<'a> {
        self.add_sub_expr_list.iter().fold(
            self.add_sub_operand.mul_div_expr.evaluate(env),
            |acc, a| {
                let operands = acc.bind_tuple(|| a.add_sub_operand.mul_div_expr.evaluate(env));
                match *a.add_sub_expr_list_group {
                    AddSubExprListGroup::Plus(_) => operands.and_then(|(l, r)| l + r),
                    AddSubExprListGroup::Minus(_) => operands.and_then(|(l, r)| l - r),
                }
            },
        )
    }
}

impl<'a> MulDivExpr<'a> {
    fn evaluate(&self, env: &Properties<'a>) -> EvalResult<'a> {
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
    fn evaluate(&self, env: &Properties<'a>) -> (EvalResult<'a>, Properties<'a>) {
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
    fn evaluate(&self, env: &Properties<'a>) -> EvalResult<'a> {
        let ve = self.block_element.evaluate(env);
        let (value, _) = self.block_inner_list.iter().fold(ve, |(_, env), e| {
            let (value, env) = e.block_element.evaluate(&env);
            (value, env)
        });
        value
    }
}

impl<'a> Block<'a> {
    fn evaluate(&self, env: &Properties<'a>) -> EvalResult<'a> {
        self.block_opt
            .as_ref()
            .map_or(Ok(Value::unit()), |bi| bi.block_inner.evaluate(env))
    }
}

impl<'a> FunctionApplication<'a> {
    fn evaluate(&self, env: &Properties<'a>) -> EvalResult<'a> {
        let f = self.primary_expression.evaluate(env);
        let method_name = self
            .function_application_opt
            .as_ref()
            .map_or("call", |a| a.identifier.identifier.text());
        let l = self
            .function_application_opt0
            .clone()
            .into_iter()
            .flat_map(|list| {
                std::iter::once(list.primary_expression).chain(
                    list.function_application_opt0_list
                        .into_iter()
                        .map(|o| o.primary_expression),
                )
            })
            .fold(f, |f, arg| {
                let arg = arg.evaluate(env)?;
                let f = f?;
                f.try_call_method(f.clone(), method_name, vec![arg])
            });

        l
    }
}

impl<'a> PrimaryExpression<'a> {
    fn evaluate(&self, env: &Properties<'a>) -> EvalResult<'a> {
        match self {
            PrimaryExpression::DecimalIntLiteral(l) => Ok(l.decimal_int_literal.evaluate()),
            PrimaryExpression::Block(b) => b.block.evaluate(env),
            PrimaryExpression::Identifier(i) => i.identifier.evaluate(env),
            PrimaryExpression::BoolLiteral(b) => Ok(b.bool_literal.evaluate()),
        }
    }
}

impl<'a> BoolLiteral<'a> {
    fn evaluate(&self) -> Value<'a> {
        match self {
            BoolLiteral::True(_) => Value::bool(true),
            BoolLiteral::False(_) => Value::bool(false),
        }
    }
}
