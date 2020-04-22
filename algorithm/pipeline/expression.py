import re
from itertools import chain
from numbers import Number
import numexpr
from numexpr.neconpiler import getExprNames
from numpy import full,inf

#左边
ops_to_methods = {
    '+':'__add__',
    '-':'__sub__',
    '*':'__mul__',
    '/':'__div__',
    '%':'__mod__',
    '**':'__pow__',
    '&':'__and__',
    '|':'__or__',
    '^':'__xor__',
    '<':'__lt__',
    '<=':'__le__',
    '>':'__gt__',
    '>=':'__ge__',
    '==':'__eq__',
    '!=':'__ne__'
}

#右边
ops_to_commuted_methods = {
    '+':'__radd__',
    '-':'__rsub__',
    '*':'__rmul__',
    '/':'__rdiv__',
    '%':'__rmod__',
    '**':'__rpow__',
    '&':'__rand__',
    '|':'__ror__',
    '^':'__rxor__',
    '<':'__gt__',
    '<=':'__ge__',
    '>':'__lt__',
    '>=':'__le__',
    '==':'__eq__',
    '!=':'__ne__'
}

unary_ops_to_methods = {'-':'__neg__','~':'__invert__'}

_Variable_Name_re = re.compile('^(x_)([0-9]*)$')

class NumericalExpression(ComputableTerm):
    """
        term binding to a numexpr expression
    """
    window_length = 0

    def __new__(cls,expr,binds,dtype):

        window_safe = (dtype == bool_type) or all(t.window_safe for t in binds)

        return super(NumericalExpression,cls).__new__(
            cls,
            inputs = binds,
            expr = expr,
            dtype = dtype,
            window_safe = window_safe
        )

    def _init(self,expr,*args,**kwargs):
        self._expr = expr
        return super(NumericalExpression,self)._init(*args,**kwargs)

    def _validate(self):
        variable_names ,_unused = getExprNames(self._expr,{})
        expr_indices = []
        for name in variable_names:
            if name == 'inf':
                pass
            match = _Variable_Name_re.match(name)

            expr_indices.append(int(match.group(2)))

        expr_indices.sort()
        expected_indices = list(range(len(self.inputs)))

        if expr_indices != expected_indices:
            raise ValueError('')

        super(NumericalExpression,self)._validate()

    def _compute(self,arrays,dates,assets,mask):
        """
            compute our stored expression string with numexpr
        """
        out = full(mask.shape,self.missing_value,dtype = self.dtype)
        numexpr.evaluate(self._expr,
                         local_dict = {'x_%d'% idx : array for idx,array in enumerate(arrays)},
                         global_dict = {'inf':inf},
                         out = out)
        return out

    def _rebind_variables(self,new_inputs):
        """
            根据inputs修改
        """
        expr = self._expr

        for idx ,input_ in reversed(list(enumerate(self.inputs))):
            old_varname = 'x_%d'%idx
            temp_new_varname = 'x_temp_%d'%new_inputs.index(input_)
            expr = expr.replace(old_varname,temp_new_varname)
        return expr.replace('_temp_','_')

    def _merge_expression(self,other):
        new_inputs = tuple(set(self.inputs).union(other,inputs))
        new_self_expr = self._rebind_variables(new_inputs)
        new_other_expr = other._rebind_variable(new_inputs)
        return new_self_expr,new_other_expr,new_inputs

    def bulid_binary_op(self,op,other):
        if isinstance(other,NumericalExpression):
            self._expr , other_expr,new_inputs = self._merge_expression(other)
        elif isinstance(other,Term):
            self_expr = self._expr
            new_inputs ,other_idx = _ensure_element(self.inputs,other)
            other_expr = 'x_%d'%other_idx
        elif isinstance(other,Number):
            self_expr = self._expr
            other_expr = str(other)
            new_inputs = self.inputs
        else:
            raise BadBinaryOperate(op,other)
        return self_expr,other_expr,new_inputsa

    @property
    def bindings(self):
        return {
            'x_%d'% i : input_
            for i ,input_ in enumerate(self.inputs)
        }
