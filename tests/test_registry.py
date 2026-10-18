import pytest

from eval_service.registry import MODEL_REGISTRY, TASK_REGISTRY, Registry


def test_builtin_components_are_registered():
    assert "mock" in MODEL_REGISTRY.list()
    assert "openai" in MODEL_REGISTRY.list()
    assert "squad" in TASK_REGISTRY.list()
    assert "agnews" in TASK_REGISTRY.list()


def test_register_and_create():
    reg = Registry("thing")

    @reg.register("widget")
    class Widget:
        def __init__(self, size: int = 1):
            self.size = size

    assert reg.create("widget", size=3).size == 3
    assert reg.list() == ["widget"]


def test_duplicate_registration_raises():
    reg = Registry("thing")

    @reg.register("widget")
    class Widget:
        pass

    with pytest.raises(ValueError):
        reg.register("widget")(Widget)


def test_unknown_component_error_lists_available():
    with pytest.raises(KeyError, match="Available"):
        TASK_REGISTRY.get("does-not-exist")
