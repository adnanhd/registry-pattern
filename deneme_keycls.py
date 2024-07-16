import torch.nn
from typing import Any, Protocol

from pydantic_pytorch.patterns.registry import PyFunctionalRegistry
from pydantic_pytorch.patterns.registry.cls_registry import ClassRegistry
from pydantic_pytorch.patterns.registry.obj_registry import ClassTracker, InstanceRegistry
from pydantic_pytorch.patterns.registry.obj_extra import Wrapped
from pydantic_pytorch.patterns.registry.api import make_class_registry


class TorchLinearModule(Protocol):
    """A torch module."""
    in_features: int
    out_features: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


FooClassRegistry = make_class_registry('Foo', protocol=TorchLinearModule, description="Class registry for Foo.")



class TorchLinearModule(Protocol):
    """A torch module."""
    in_features: int
    out_features: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""


class TorchModuleRegistry(ClassRegistry[TorchLinearModule]):
    """A registry for torch modules."""


TorchModuleRegistry.register(torch.nn.Module)
TorchModuleRegistry_ = TorchModuleRegistry()

TorchModuleRegistry_.register_class(torch.nn.Linear)


class Hello(metaclass=ClassTracker):
    """A hello class."""

    def __init__(self):
        pass

    def say_hello(self) -> str:
        """Say hello."""
        return "Hello"


class HelloLike(Protocol):
    """A hello protocol."""

    def say_hello(self) -> str:
        """Say hello."""

    def __str__(self) -> str:
        ...


class HelloTracker(InstanceRegistry[HelloLike]):
    """A tracker for hello instances."""


Tracker = HelloTracker()
HelloTracker.register(Hello)
a = Tracker.register_instance(Hello())
b = Tracker.register_instance(Hello())
c = Tracker.register_instance(Hello())


@Tracker.track_class_instances
class Hello2:
    """Another hello class."""

    def say_hello(self) -> str:
        """Say hello."""
        return "Hello2"


class MetricRegistry(PyFunctionalRegistry[[float, float], float]):
    """A registry for metrics."""


@MetricRegistry().register_function
def add(a: float, b: float) -> float:
    return a + b


class MyThirdPartyModule:
    def __init__(self, a: int = 1, b: int = 2):
        self.a = a
        self.b = b

    def __call__(self, a: int, b: int) -> int:
        return a + b
    
    def foo(self, a: int, b: int) -> int:
        return a + b

cfg = {'a': 2, 'b': 3}


class ThirdPartyModuleWraper(Wrapped[MyThirdPartyModule]):
    ...


wrap = ThirdPartyModuleWraper.from_config(cfg)
# wrap = Wrapped[MyThirdPartyModule].from_config(cfg)
wrap.foo(1,2)
wrap.foo(1,2)
print(dir(wrap))

