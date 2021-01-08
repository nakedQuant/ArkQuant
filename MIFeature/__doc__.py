# np.fmax(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', subok=True)


# 通过scipy.interpolate.interp1d插值形成的模型，通过sco.fmin_bfgs计算min
# param
# find_min_pos: 寻找min的点位值
# param
# linear_interp: scipy.interpolate.interp1d插值形成的模型
# local_min_pos = sco.fmin_bfgs(linear_interp, find_min_pos, disp=False)[0]
# scipy.interpolate.interp1d插值形成模型

# import sympy as sy
# 符号计算（在交换式金融分析，比较有效）
# sy.Symbol('x') sy.sqrt(x)
# sy.solve(x ** 2 -1 ) 方程式右边为0
# #打印出符号积分
# sy.pretty(sy.Intergal(sy.sin(x) + 0.5 * x,x))
# #积分,求出反导数
# init_func = sy.intergrate(sy.sin(x) + 0.5 * x,x)
# #求导
# init_func.diff()
# #偏导数
# init_func.diff(,x)
# #subs代入数值，evalf求值
# init_func.subs(x,0.5).evalf()
# #nsolve与solve
# from sympy.solvers import nsolve
# solve 处理等式右边为0的表达式；而nsolve处理表达式（范围更加广）

"""
    preprocess: missing value
                numeric value
                categorical value







"""