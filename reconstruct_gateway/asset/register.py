# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from toolz import curry
from abc import ABC

from gateway.asset.assets import Asset, Equity, Convertible, Fund
# A global dictionary for storing instances of Registry:
custom_types = {}


class AssetConvertible(ABC):
    """
    ABC for types that are convertible to integer-representations of
    Assets.

    Includes Asset, six.string_types, and Integral
    """
    pass


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


AssetConvertible.register(Asset)
PricingDataAssociable.register(Asset)
PricingDataAssociable.register(Equity)
PricingDataAssociable.register(Convertible)
PricingDataAssociable.register(Fund)


class Registry(object):
    """
    Responsible for managing all instances of custom subclasses of a
    given abstract base class - only one instance needs to be created
    per abstract base class, and should be created through the
    create_registry function/decorator. All management methods
    for a given base class can be called through the global wrapper functions
    rather than through the object instance itself.

    Parameters
    ----------
    interface : type
        The abstract base class to manage.
    """
    def __init__(self, interface):
        self.interface = interface
        self._factories = {}

    def load(self, name):
        """Construct an object from a registered factory.

        Parameters
        ----------
        name : str
            Name with which the factory was registered.
        """
        try:
            return self._factories[name]()
        except KeyError:
            raise ValueError(
                "no %s factory registered under name %r, options are: %r" %
                (self.interface.__name__, name, sorted(self._factories)),
            )

    def is_registered(self, name):
        """Check whether we have a factory registered under ``name``.
        """
        return name in self._factories

    @curry
    def register(self, name, factory):
        if self.is_registered(name):
            raise ValueError(
                "%s factory with name %r is already registered" %
                (self.interface.__name__, name)
            )

        self._factories[name] = factory

        return factory

    def unregister(self, name):
        try:
            del self._factories[name]
        except KeyError:
            raise ValueError(
                "%s factory %r was not already registered" %
                (self.interface.__name__, name)
            )

    def clear(self):
        self._factories.clear()


# Public wrapper methods for Registry:

def get_registry(interface):
    """
    Getter method for retrieving the registry
    instance for a given extendable type

    Parameters
    ----------
    interface : type
        extendable type (base class)

    Returns
    -------
    manager : Registry
        The corresponding registry
    """
    try:
        return custom_types[interface]
    except KeyError:
        raise ValueError("class specified is not an extendable type")


def load(interface, name):
    """
    Retrieves a custom class whose name is given.

    Parameters
    ----------
    interface : type
        The base class for which to perform this operation
    name : str
        The name of the class to be retrieved.

    Returns
    -------
    obj : object
        An instance of the desired class.
    """
    return get_registry(interface).load(name)


@curry
def register(interface, name, custom_class):
    """
    Registers a class for retrieval by the load method

    Parameters
    ----------
    interface : type
        The base class for which to perform this operation
    name : str
        The name of the subclass
    custom_class : type
        The class to register, which must be a subclass of the
        abstract base class in self.dtype
    """

    return get_registry(interface).register(name, custom_class)


def unregister(interface, name):
    """
    If a class is registered with the given name,
    it is unregistered.

    Parameters
    ----------
    interface : type
        The base class for which to perform this operation
    name : str
        The name of the class to be unregistered.
    """
    get_registry(interface).unregister(name)


def clear(interface):
    """
    Unregisters all current registered classes

    Parameters
    ----------
    interface : type
        The base class for which to perform this operation
    """
    get_registry(interface).clear()


def create_registry(interface):
    """
    Create a new registry for an extensible interface.

    Parameters
    ----------
    interface : type
        The abstract data type for which to create a registry,
        which will manage registration of factories for this type.

    Returns
    -------
    interface : type
        The data type specified/decorated, unaltered.
    """
    if interface in custom_types:
        raise ValueError('there is already a Registry instance '
                         'for the specified type')
    custom_types[interface] = Registry(interface)
    return interface


extensible = create_registry
