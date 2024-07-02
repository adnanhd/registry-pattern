import weakref


class InstanceTracker(type):
    def __init__(cls, name, bases, attrs):
        cls._instances = weakref.WeakSet()
        super().__init__(name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        cls._instances.add(instance)
        return instance

    def get_instances(cls):
        # Convert weak references to strong references for returning
        cls._instances
        return list(cls._instances)
