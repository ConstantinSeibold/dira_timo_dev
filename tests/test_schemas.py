import pytest
from pydantic import ValidationError

from schemas import create_dynamic_schema


class TestCreateDynamicSchema:
    def test_single_field(self):
        Schema = create_dynamic_schema(["output"])
        obj = Schema(output="hello")
        assert obj.output == "hello"

    def test_multiple_fields(self):
        Schema = create_dynamic_schema(["foo", "bar", "baz"])
        obj = Schema(foo="a", bar="b", baz="c")
        assert obj.foo == "a"
        assert obj.bar == "b"
        assert obj.baz == "c"

    def test_missing_field_raises(self):
        Schema = create_dynamic_schema(["foo", "bar"])
        with pytest.raises(ValidationError):
            Schema(foo="a")

    def test_fields_match(self):
        fields = ["x", "y", "z"]
        Schema = create_dynamic_schema(fields)
        assert list(Schema.model_fields.keys()) == fields

    def test_validation_pass(self):
        Schema = create_dynamic_schema(["output"])
        obj = Schema.model_validate({"output": "value"})
        assert obj.output == "value"

    def test_validation_fail_extra_only(self):
        Schema = create_dynamic_schema(["output"])
        with pytest.raises(ValidationError):
            Schema.model_validate({"wrong_key": "value"})
