class WrapperMeta(type):
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        # Create delegating methods for special operations
        cls._create_delegating_methods()
        return cls

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        cls._create_delegating_properties(instance)
        return instance

    def _create_delegating_methods(cls):
        # Define a list of special method names to delegate
        special_methods = [
            '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__mod__',
            '__pow__', '__and__', '__or__', '__xor__', '__lshift__', '__rshift__',
            '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__',
            '__round__', '__floor__', '__ceil__', '__trunc__', '__int__', '__float__', '__complex__',
            '__neg__', '__pos__', '__abs__', '__invert__', '__call__',
            '__getitem__', '__setitem__', '__delitem__', '__len__', '__iter__', '__contains__'
        ]

        for name in special_methods:
            setattr(cls, name, cls._delegate_method(name))

    def _create_delegating_properties(cls, instance):
        for name in dir(instance):
            if not name.startswith('__') and not callable(getattr(instance, name)):
                setattr(cls, name, cls._delegate_property(name))

    @staticmethod
    def _delegate_method(name):
        def method(self, *args, **kwargs):
            return getattr(self._instance, name)(*args, **kwargs)

        return method

    @staticmethod
    def _delegate_property(name):
        def property_getter(self):
            return getattr(self._instance, name)

        def property_setter(self, value):
            setattr(self._instance, name, value)

        def property_deleter(self):
            delattr(self._instance, name)
        return property(property_getter, property_setter, property_deleter)

    '''
    def __instancecheck__(cls, instance):
        subclass = instance.__class__
        import pdb
        pdb.set_trace()
        return NotImplemented

    def __subclasscheck__(cls, subclass):
        """Override for issubclass(subclass, cls)."""
        import pdb
        pdb.set_trace()
        return NotImplemented
    '''

class InstanceWrapper(metaclass=WrapperMeta):
    def __init__(self, instance, config):
        self._instance = instance
        self._config = config

    def __getattr__(self, name):
        if name in {'_instance', '_config'}:
            return self.__getattribute__(name)
        else:
            return getattr(self._instance, name)

    def __setattr__(self, name, value):
        if name in {'_instance', '_config'}:
            super().__setattr__(name, value)
        else:
            setattr(self._instance, name, value)

    def __delattr__(self, name):
        if name in {'_instance', '_config'}:
            super().__delattr__(name)
        else:
            delattr(self._instance, name)

    def __dir__(self):
        return dir(self._instance)

    def __repr__(self):
        return f'{self.__class__.__name__}[{self._instance.__class__.__name__}](' + ', '.join(f'{key!r}={val!r}' for key, val in self._config.items()) + ')'