import collections.abc
from logging import DEBUG
import pydantic
import logging
import typing
import math
import time


class Profiler(pydantic.BaseModel):
    name: str
    _s_perf_counter: float = pydantic.PrivateAttr(math.nan)
    _e_perf_counter: float = pydantic.PrivateAttr(math.nan)
    _s_process_time: float = pydantic.PrivateAttr(math.nan)
    _e_process_time: float = pydantic.PrivateAttr(math.nan)
    _logger: pydantic.InstanceOf[logging.Logger] = pydantic.PrivateAttr()

    model_config = pydantic.ConfigDict(frozen=True)

    def model_post_init(self, __context: typing.Any) -> None:
        ctx = __context if __context is not None else {}
        lgr_name = ctx.get('name', __class__.__name__)
        self._logger = logging.getLogger(lgr_name)
        self._logger.setLevel(logging.DEBUG)

    def set(self):
        self._s_process_time = time.process_time()
        self._s_perf_counter = time.perf_counter()
        return self

    def lap(self):
        assert not math.isnan(self._s_process_time), "Call set() first!"
        self._e_process_time = time.process_time()
        self._e_perf_counter = time.perf_counter()
        return self

    def log(self):
        assert not math.isnan(self._s_process_time), "Call set() first!"
        assert not math.isnan(
            self._e_perf_counter), "Call set() and lap() first"
        self._logger.debug(f'{self.name} takes {self.perf_counter:.2e} seconds in real'
                           f' and {self.process_time:.2e} seconds in CPU.')
        return self

    def reset(self):
        self._s_perf_counter = math.nan
        self._e_perf_counter = math.nan
        self._s_process_time = math.nan
        self._e_process_time = math.nan
        return self

    @property
    def process_time(self) -> float:
        return self._e_process_time - self._s_process_time

    @property
    def perf_counter(self) -> float:
        return self._e_perf_counter - self._s_perf_counter


class ContextProfiler(Profiler):
    def __enter__(self):
        return self.set()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lap().log()


class FunctionProfiler(Profiler):
    def __call__(self, fn: typing.Callable):
        assert callable(fn), "fn must be callable"

        def wrapper(*args, **kwds):
            with ContextProfiler(name=self.name):
                return fn(*args, **kwds)
        return wrapper


def profile_call(fn: typing.Callable):
    return FunctionProfiler(name=fn.__name__)(fn)


class _IteratorProfiler(Profiler):
    iterable: typing.Iterable[typing.Any]

    @pydantic.field_validator('iterable')
    def _validate_iterable(cls, v):
        return iter(v)

    def __iter__(self):
        import pdb; pdb.set_trace()
        return self.set()

    def __next__(self):
        try:
            return next(self.lap().log().iterable)
        except StopIteration:
            self.reset()
            raise StopIteration


class IteratorProfiler(Profiler):
    iterator: typing.Any

    @pydantic.field_validator('iterator')
    @classmethod
    def _validate_iterator(cls, v):
        if not isinstance(v, collections.abc.Iterable):
            raise TypeError(f'{v} is not iterable')
        return v

    def __iter__(self):
        return iter(_IteratorProfiler(name=self.name, iterable=self.iterator))


if __name__ == '__main__':
    # have a look at https://realpython.com/python-profiling/ for more information
    logging.basicConfig(level=logging.INFO)

    @FunctionProfiler(name='simple function')
    def foo():
        return 2

    with ContextProfiler(name='simple context'):
        foo()

    x = IteratorProfiler(name='list(range(5))', iterator=list(range(5)))
    # x = list(range(5))
    list
    for j in range(3):
        for i in x:
            pass
        print('done')
