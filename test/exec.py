namespace = dict()
with open('/Users/python/Library/Mobile Documents/com~apple~CloudDocs/simulation/test/test_driver.py','r') as f:
    exec(f.read(),namespace)

print(namespace.keys())
test = namespace['UnionEngine']
print(test)
# ins = test()
# print(ins)
# print(namespace['__builtins__'])
# print(namespace['signature'])

import glob
res = glob.glob('/Users/python/Library/Mobile Documents/com~apple~CloudDocs/simulation/pipeline/strategy/*.py')
print(list(res))

print(__file__)


# exec eval compile将字符串转化为可执行代码 , exec compile source into code or AST object ,if filename is None ,'<string>' is used
# code = compile(self.algoscript, algo_filename, 'exec')
# exec_(code, self.namespace)
#
# # dict get参数可以为方法或者默认参数
# self._initialize = self.namespace.get('initialize', noop)
# self._handle_data = self.namespace.get('handle_data', noop)
# self._before_trading_start = self.namespace.get(
#     'before_trading_start',
# )