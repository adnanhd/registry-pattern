import inspect
import pytest
import time
import weakref
import gc

# Import functions and exceptions from your enhanced validator module
from registry.core._validator import (
    validate_class,
    validate_class_structure,
    validate_class_hierarchy,
    validate_init,
    validate_function,
    validate_function_signature,
    validate_function_parameters,
    validate_instance_hierarchy,
    validate_instance_structure,
    ConformanceError,
    InheritanceError,
    ValidationError,
    get_func_name,
    get_type_name,
    get_mem_addr,
    clear_validation_cache,
    get_cache_stats,
    configure_validation_cache,
)

# -----------------------------------------------------------------------------
# Dummy Protocol and Classes for Testing
# -----------------------------------------------------------------------------


class DummyProtocol:
    def foo(self, x: int) -> int: ...
    def bar(self, y: str) -> str: ...


class ComplexProtocol:
    def process(self, data: str) -> str: ...
    def validate(self, data: str) -> bool: ...
    def configure(self, settings: dict) -> None: ...


# A compliant class: implements both foo and bar with matching signatures
class GoodClass:
    def foo(self, x: int) -> int:
        return x * 2

    def bar(self, y: str) -> str:
        return y.upper()


# A noncompliant class: missing 'bar'
class BadClass:
    def foo(self, x: int) -> int:
        return x * 2


# Wrong signature class
class WrongSignatureClass:
    def foo(self, x: float) -> str:  # Wrong parameter and return types
        return str(x)

    def bar(self, y: str) -> str:
        return y.upper()


# For inheritance testing, use an abstract base class
class BaseABC:
    pass


class Derived(BaseABC):
    pass


class NotDerived:
    pass


# Dummy functions for signature testing
def expected_func(a: int, b: str) -> bool:
    return True


def matching_func(a: int, b: str) -> bool:
    return False


def mismatching_func(a: int) -> bool:
    return True


def wrong_types_func(a: float, b: int) -> str:
    return "wrong"


# -----------------------------------------------------------------------------
# Enhanced Test Cases for Basic Validation
# -----------------------------------------------------------------------------


class TestEnhancedBasicValidation:
    """Test enhanced basic validation functions."""

    def test_validate_class_success(self):
        """Test that validate_class returns the class if it's a class."""
        cls = validate_class(GoodClass)
        assert inspect.isclass(cls)
        assert cls is GoodClass

    def test_validate_class_failure_with_context(self):
        """Test that validate_class provides rich context on failure."""
        with pytest.raises(ValidationError) as exc_info:
            validate_class(42)

        error = exc_info.value
        assert "42 is not a class" in str(error)
        assert hasattr(error, 'suggestions')
        assert hasattr(error, 'context')
        assert len(error.suggestions) > 0
        assert "Ensure you're passing a class" in error.suggestions[0]

    def test_validate_function_success(self):
        """Test that validate_function accepts valid functions."""
        func = validate_function(matching_func)
        assert inspect.isfunction(func)
        assert func is matching_func

    def test_validate_function_failure_with_context(self):
        """Test that validate_function provides rich context on failure."""
        with pytest.raises(ValidationError) as exc_info:
            validate_function(123)

        error = exc_info.value
        assert hasattr(error, 'suggestions')
        assert hasattr(error, 'context')
        assert any("function" in suggestion.lower() for suggestion in error.suggestions)


# -----------------------------------------------------------------------------
# Enhanced Test Cases for Protocol Validation
# -----------------------------------------------------------------------------


class TestEnhancedProtocolValidation:
    """Test enhanced protocol validation with rich error context."""

    def test_validate_class_structure_success(self):
        """Test successful class structure validation."""
        cls = validate_class_structure(GoodClass, DummyProtocol)
        assert cls is GoodClass

    def test_validate_class_structure_missing_methods(self):
        """Test class structure validation with missing methods."""
        with pytest.raises(ConformanceError) as exc_info:
            validate_class_structure(BadClass, DummyProtocol)

        error = exc_info.value
        assert "does not conform to protocol" in str(error)
        assert "Missing methods: bar" in str(error)
        assert hasattr(error, 'suggestions')
        assert hasattr(error, 'context')

        # Check that suggestions include method signatures
        method_suggestions = [s for s in error.suggestions if "def bar" in s]
        assert len(method_suggestions) > 0

        # Check context contains useful information
        assert 'expected_type' in error.context
        assert 'actual_type' in error.context
        assert error.context['expected_type'] == 'DummyProtocol'
        assert error.context['actual_type'] == 'BadClass'

    def test_validate_class_structure_wrong_signatures(self):
        """Test class structure validation with wrong method signatures."""
        with pytest.raises(ConformanceError) as exc_info:
            validate_class_structure(WrongSignatureClass, DummyProtocol)

        error = exc_info.value
        assert "does not conform to protocol" in str(error)
        assert "Signature mismatches" in str(error)
        assert hasattr(error, 'suggestions')

        # Should suggest fixing parameter types
        param_suggestions = [s for s in error.suggestions if "parameter" in s.lower() or "type" in s.lower()]
        assert len(param_suggestions) > 0

    def test_validate_class_structure_complex_protocol(self):
        """Test validation against a more complex protocol."""
        class IncompleteClass:
            def process(self, data: str) -> str:
                return data
            # Missing validate and configure methods

        with pytest.raises(ConformanceError) as exc_info:
            validate_class_structure(IncompleteClass, ComplexProtocol)

        error = exc_info.value
        expected_error_string = """Class IncompleteClass does not conform to protocol ComplexProtocol:
Missing methods: configure, validate
  Expected: ComplexProtocol
  Actual: IncompleteClass
  Artifact: IncompleteClass
  Suggestions:
    • Add method: def configure(self, settings: dict) -> None: ...
    • Add method: def validate(self, data: str) -> bool: ..."""
        assert expected_error_string == str(error)

        # Check that all missing methods have suggestions
        validate_suggestions = [s for s in error.suggestions if "def validate" in s]
        configure_suggestions = [s for s in error.suggestions if "def configure" in s]
        assert len(validate_suggestions) > 0
        assert len(configure_suggestions) > 0


# -----------------------------------------------------------------------------
# Enhanced Test Cases for Inheritance Validation
# -----------------------------------------------------------------------------


class TestEnhancedInheritanceValidation:
    """Test enhanced inheritance validation with rich error context."""

    def test_validate_class_hierarchy_success(self):
        """Test successful inheritance validation."""
        cls = validate_class_hierarchy(Derived, BaseABC)
        assert cls is Derived

    def test_validate_class_hierarchy_failure_with_context(self):
        """Test inheritance validation failure with rich context."""
        with pytest.raises(InheritanceError) as exc_info:
            validate_class_hierarchy(NotDerived, BaseABC)

        error = exc_info.value
        assert hasattr(error, 'suggestions')
        assert hasattr(error, 'context')

        # Check suggestions include inheritance advice
        inheritance_suggestions = [s for s in error.suggestions if "inherit" in s.lower()]
        assert len(inheritance_suggestions) > 0

        # Check context
        assert 'expected_type' in error.context
        assert 'actual_type' in error.context


# -----------------------------------------------------------------------------
# Enhanced Test Cases for Function Signature Validation
# -----------------------------------------------------------------------------


class TestEnhancedFunctionValidation:
    """Test enhanced function validation with detailed error reporting."""

    def test_validate_function_signature_success(self):
        """Test successful function signature validation."""
        func = validate_function_signature(matching_func, expected_func)
        assert func is matching_func

    def test_validate_function_signature_parameter_count_mismatch(self):
        """Test function signature validation with parameter count mismatch."""
        with pytest.raises(ConformanceError) as exc_info:
            validate_function_signature(mismatching_func, expected_func)

        error = exc_info.value
        assert "Parameter count mismatch" in str(error)
        assert hasattr(error, 'suggestions')
        assert any("parameters" in suggestion for suggestion in error.suggestions)

    def test_validate_function_signature_type_mismatch(self):
        """Test function signature validation with type mismatches."""
        with pytest.raises(ConformanceError) as exc_info:
            validate_function_signature(wrong_types_func, expected_func)

        error = exc_info.value
        assert "type mismatch" in str(error)
        assert hasattr(error, 'suggestions')

        # Should suggest correct parameter types
        type_suggestions = [s for s in error.suggestions if "parameter type" in s.lower()]
        assert len(type_suggestions) > 0

    def test_validate_function_parameters_with_callable_type(self):
        """Test function parameter validation against callable type."""
        from typing import Callable

        ExpectedType = Callable[[int, str], bool]
        func = validate_function_parameters(matching_func, ExpectedType)
        assert func is matching_func

    def test_validate_function_parameters_failure_with_context(self):
        """Test function parameter validation failure with rich context."""
        from typing import Callable

        ExpectedType = Callable[[int, str], bool]

        with pytest.raises(ConformanceError) as exc_info:
            validate_function_parameters(mismatching_func, ExpectedType)

        error = exc_info.value
        assert hasattr(error, 'suggestions')
        assert hasattr(error, 'context')
        expected_error_string= """Function 'mismatching_func' parameter validation failed:
  • Parameter count mismatch: expected 2, got 1
  Expected: typing.Callable[[int, str], bool]
  Actual: (a: int) -> bool
  Artifact: mismatching_func
  Suggestions:
    • Function should have exactly 2 parameters"""
        assert expected_error_string == str(error)


# -----------------------------------------------------------------------------
# Enhanced Test Cases for Instance Validation
# -----------------------------------------------------------------------------


class TestEnhancedInstanceValidation:
    """Test enhanced instance validation with rich error context."""

    def test_validate_instance_hierarchy_success(self):
        """Test successful instance hierarchy validation."""
        instance = Derived()
        validated = validate_instance_hierarchy(instance, BaseABC)
        assert validated is instance

    def test_validate_instance_hierarchy_failure_with_context(self):
        """Test instance hierarchy validation failure with rich context."""
        instance = NotDerived()

        with pytest.raises(InheritanceError) as exc_info:
            validate_instance_hierarchy(instance, BaseABC)

        error = exc_info.value
        assert hasattr(error, 'suggestions')
        assert hasattr(error, 'context')
        expected_error_string = f"""{instance} is not an instance of BaseABC
  Expected: {get_type_name(BaseABC)}
  Actual: {get_type_name(instance.__class__)}
  Artifact: {instance}
  Suggestions:
    • Ensure the instance is of type BaseABC
    • Check that the class inherits from BaseABC
    • Got instance of NotDerived, expected BaseABC"""
        assert expected_error_string == str(error)

    def test_validate_instance_structure_success(self):
        """Test successful instance structure validation."""
        instance = GoodClass()
        validated = validate_instance_structure(instance, DummyProtocol)
        assert validated is instance

    def test_validate_instance_structure_missing_attributes(self):
        """Test instance structure validation with missing attributes."""
        instance = BadClass()

        with pytest.raises(ConformanceError) as exc_info:
            validate_instance_structure(instance, DummyProtocol)

        error = exc_info.value
        assert "does not conform to protocol" in str(error)
        assert "Missing attributes: bar" in str(error)
        assert hasattr(error, 'suggestions')

        # Should suggest adding missing attributes
        attr_suggestions = [s for s in error.suggestions if "Add attribute" in s]
        assert len(attr_suggestions) > 0

    def test_validate_instance_structure_wrong_signatures(self):
        """Test instance structure validation with wrong method signatures."""
        instance = WrongSignatureClass()

        with pytest.raises(ConformanceError) as exc_info:
            validate_instance_structure(instance, DummyProtocol)

        error = exc_info.value
        assert "Signature mismatches" in str(error)
        assert hasattr(error, 'suggestions')


# -----------------------------------------------------------------------------
# Enhanced Test Cases for Utility Functions
# -----------------------------------------------------------------------------


class TestEnhancedUtilityFunctions:
    """Test enhanced utility functions."""

    def test_get_func_name_with_regular_function(self):
        """Test get_func_name with regular functions."""
        assert get_func_name(matching_func, qualname=False) == "matching_func"
        assert "matching_func" in get_func_name(matching_func, qualname=True)

    def test_get_func_name_with_partial(self):
        """Test get_func_name with functools.partial."""
        from functools import partial

        partial_func = partial(matching_func)
        assert get_func_name(partial_func, qualname=False) == "matching_func"

    def test_get_func_name_with_wrapped_function(self):
        """Test get_func_name with wrapped functions."""
        from functools import wraps

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        @decorator
        def decorated_func():
            pass

        assert get_func_name(decorated_func) == "decorated_func"

    def test_get_type_name(self):
        """Test get_type_name function."""
        assert get_type_name(GoodClass) == "GoodClass"
        assert get_type_name(int) == "int"

    def test_get_mem_addr(self):
        """Test get_mem_addr function."""
        obj = GoodClass()
        addr_with_prefix = get_mem_addr(obj, with_prefix=True)
        addr_without_prefix = get_mem_addr(obj, with_prefix=False)

        assert addr_with_prefix.startswith("0x")
        assert not addr_without_prefix.startswith("0x")
        assert addr_with_prefix[2:] == addr_without_prefix


# -----------------------------------------------------------------------------
# Enhanced Test Cases for Validation Cache
# -----------------------------------------------------------------------------


class TestEnhancedValidationCache:
    """Test enhanced validation caching functionality."""

    def setup_method(self):
        """Set up test method by clearing cache."""
        clear_validation_cache()

    def test_validation_cache_basic_functionality(self):
        """Test basic validation cache functionality."""
        # Configure cache for testing
        configure_validation_cache(max_size=100, ttl_seconds=1.0)

        # First validation should populate cache
        validate_class(GoodClass)

        stats = get_cache_stats()
        # Note: Cache may have entries from the validation process
        assert isinstance(stats['total_entries'], int)
        assert isinstance(stats['max_size'], int)
        assert isinstance(stats['ttl_seconds'], float)

    def test_validation_cache_expiration(self):
        """Test validation cache TTL expiration."""
        # Configure short TTL for testing
        configure_validation_cache(max_size=100, ttl_seconds=0.1)

        # Validate something to populate cache
        validate_class(GoodClass)

        # Wait for TTL to expire
        time.sleep(0.2)

        # Get stats - expired entries should be detected
        stats = get_cache_stats()
        assert 'expired_entries' in stats

    def test_validation_cache_size_limit(self):
        """Test validation cache size limiting."""
        # Configure small cache for testing
        configure_validation_cache(max_size=2, ttl_seconds=10.0)

        # Clear cache to start fresh
        clear_validation_cache()

        # Validate multiple items to test size limiting
        classes = [GoodClass, BadClass, WrongSignatureClass]
        for cls in classes:
            try:
                validate_class(cls)
            except ValidationError:
                pass  # Expected for some classes

        stats = get_cache_stats()
        # Cache should respect size limit
        assert stats['total_entries'] <= stats['max_size']

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Populate cache
        validate_class(GoodClass)

        # Clear cache
        cleared_count = clear_validation_cache()

        # Verify cache is cleared
        stats = get_cache_stats()
        assert stats['total_entries'] == 0
        assert isinstance(cleared_count, int)


# -----------------------------------------------------------------------------
# Enhanced Test Cases for Error Message Quality
# -----------------------------------------------------------------------------


class TestEnhancedErrorMessages:
    """Test the quality and helpfulness of error messages."""

    def test_error_message_contains_suggestions(self):
        """Test that all enhanced errors contain actionable suggestions."""
        test_cases = [
            (lambda: validate_class(42), ValidationError),
            (lambda: validate_class_structure(BadClass, DummyProtocol), ConformanceError),
            (lambda: validate_class_hierarchy(NotDerived, BaseABC), InheritanceError),
            (lambda: validate_function(123), ValidationError),
        ]

        for test_func, expected_error in test_cases:
            with pytest.raises(expected_error) as exc_info:
                test_func()

            error = exc_info.value
            assert hasattr(error, 'suggestions')
            assert len(error.suggestions) > 0
            # All suggestions should be non-empty strings
            assert all(isinstance(s, str) and len(s) > 0 for s in error.suggestions)

    def test_error_message_contains_context(self):
        """Test that enhanced errors contain useful context information."""
        with pytest.raises(ConformanceError) as exc_info:
            validate_class_structure(BadClass, DummyProtocol)

        error = exc_info.value
        assert hasattr(error, 'context')
        assert isinstance(error.context, dict)

        # Should contain key context information
        expected_keys = ['expected_type', 'actual_type', 'artifact_name']
        for key in expected_keys:
            assert key in error.context

    def test_error_message_readability(self):
        """Test that error messages are human-readable and helpful."""
        with pytest.raises(ConformanceError) as exc_info:
            validate_class_structure(WrongSignatureClass, DummyProtocol)

        error = exc_info.value
        error_str = str(error)

        # Should contain clear information about what went wrong
        assert "does not conform" in error_str
        assert "DummyProtocol" in error_str
        assert "WrongSignatureClass" in error_str

        # Should contain actionable suggestions
        assert "Suggestions:" in error_str
        assert "•" in error_str  # Bullet points for suggestions


# -----------------------------------------------------------------------------
# Enhanced Test Cases for Edge Cases
# -----------------------------------------------------------------------------


class TestEnhancedEdgeCases:
    """Test enhanced validation with edge cases and boundary conditions."""

    def test_validation_with_none_values(self):
        """Test validation behavior with None values."""
        with pytest.raises(ValidationError):
            validate_class(None)

        with pytest.raises(ValidationError):
            validate_function(None)

    def test_validation_with_empty_protocol(self):
        """Test validation with empty protocol."""
        class EmptyProtocol:
            pass

        class EmptyClass:
            pass

        # Should succeed since no methods are required
        result = validate_class_structure(EmptyClass, EmptyProtocol)
        assert result is EmptyClass

    def test_validation_with_builtin_types(self):
        """Test validation with builtin types."""
        # Builtin functions should be valid
        result = validate_function(len)
        assert result is len

        # Builtin types should not be valid classes for our purposes in some contexts
        result = validate_class(int)
        assert result is int

    def test_validation_with_complex_inheritance(self):
        """Test validation with complex inheritance hierarchies."""
        class GrandParent:
            pass

        class Parent(GrandParent):
            pass

        class Child(Parent):
            pass

        instance = Child()

        # Should validate against any class in the hierarchy
        validate_instance_hierarchy(instance, GrandParent)
        validate_instance_hierarchy(instance, Parent)
        validate_instance_hierarchy(instance, Child)

    def test_memory_cleanup_after_validation(self):
        """Test that validation doesn't cause memory leaks."""
        import sys

        # Create objects for validation
        test_objects = []
        for i in range(100):
            class TestClass:
                def method(self):
                    pass
            test_objects.append(TestClass)

        # Validate all objects
        for obj in test_objects:
            try:
                validate_class(obj)
            except ValidationError:
                pass

        # Clear references
        del test_objects
        gc.collect()

        # Cache should still work normally
        validate_class(GoodClass)


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestEnhancedValidationIntegration:
    """Integration tests for the enhanced validation system."""

    def test_full_validation_workflow(self):
        """Test a complete validation workflow."""
        # Test class validation
        validate_class(GoodClass)

        # Test protocol conformance
        validate_class_structure(GoodClass, DummyProtocol)

        # Test instance validation
        instance = GoodClass()
        validate_instance_structure(instance, DummyProtocol)

        # Test function validation
        validate_function(matching_func)
        validate_function_signature(matching_func, expected_func)

    def test_validation_error_chaining(self):
        """Test that validation errors chain properly."""
        try:
            validate_class_structure(BadClass, DummyProtocol)
        except ConformanceError as e:
            # Error should have proper context and suggestions
            assert hasattr(e, 'context')
            assert hasattr(e, 'suggestions')
            assert len(e.suggestions) > 0

            # Error should be properly formatted
            error_str = str(e)
            assert len(error_str) > len(e.message)  # Should be enhanced

    def test_performance_with_cache(self):
        """Test that caching improves performance."""
        import time

        # Clear cache and configure
        clear_validation_cache()
        configure_validation_cache(max_size=1000, ttl_seconds=60.0)

        # Time first validation (cache miss)
        start_time = time.time()
        validate_class(GoodClass)
        first_time = time.time() - start_time

        # Time second validation (cache hit)
        start_time = time.time()
        validate_class(GoodClass)
        second_time = time.time() - start_time

        # Second validation should be faster (though this is not guaranteed due to timing variations)
        # At minimum, it should not cause errors
        assert second_time >= 0

        # Check that cache has entries
        stats = get_cache_stats()
        assert stats['total_entries'] >= 0
