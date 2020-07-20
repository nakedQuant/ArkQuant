from abc import ABC

class AssetConvertible(ABC):
    """
    ABC for types that are convertible to integer-representations of
    Assets.

    Includes Asset, six.string_types, and Integral
    """
    pass


AssetConvertible.register(Integral)
AssetConvertible.register(Asset)
# Use six.string_types for Python2/3 compatibility
for _type in string_types:
    AssetConvertible.register(_type)


class NotAssetConvertible(ValueError):
    pass


class PricingDataAssociable(ABC):
    """
    ABC for types that can be associated with pricing data.

    Includes Asset, Future, ContinuousFuture
    一、相关概念
    虚拟子类是将其他的不是从抽象基类派生的类”注册“到抽象基类，让Python解释器将该类作为抽象基类的子类使用，因此称为虚拟子类，
    这样第三方类不需要直接继承自抽象基类。注册的虚拟子类不论是否实现抽象基类中的抽象内容，Python都认为它是抽象基类的子类，
    调用 issubclass(子类，抽象基类),isinstance (子类对象，抽象基类)都会返回True。

    这种通过注册增加虚拟子类是抽象基类动态性的体现，也是符合Python风格的方式。它允许我们动态地，清晰地改变类的属别关系。
    当一个类继承自抽象基类时，该类必须完成抽象基类定义的语义；当一个类注册为虚拟子类时，这种限制则不再有约束力，
    可以由程序开发人员自己约束自己，因此提供了更好的灵活性与扩展性（当然也带来了一些意外的问题）。这种能力在框架程序使用第三方插件时，
    采用虚拟子类即可以明晰接口，只要第三方插件能够提供框架程序要求的接口，不管其类型是什么，都可以使用抽象基类去调用相关能力，
    又不会影响框架程序去兼容外部接口的内部实现。老猿认为，从某种程度上讲，虚拟子类这种模式，是在继承这种模式下的一种多态实现。
    """
    pass


PricingDataAssociable.register(Asset)
PricingDataAssociable.register(Future)
PricingDataAssociable.register(ContinuousFuture)
