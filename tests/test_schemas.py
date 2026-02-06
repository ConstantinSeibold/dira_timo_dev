import pytest
from pydantic import ValidationError

from schemas import SingleOutput, TableOutput, get_schema, get_target_fields


class TestSingleOutput:
    def test_valid(self):
        obj = SingleOutput(output="hello")
        assert obj.output == "hello"

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            SingleOutput()


class TestTableOutput:
    def test_valid(self):
        obj = TableOutput(
            refined_clinical_question_and_info="q",
            organizational_information="o",
            deleted_input="d",
        )
        assert obj.refined_clinical_question_and_info == "q"
        assert obj.organizational_information == "o"
        assert obj.deleted_input == "d"

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            TableOutput(refined_clinical_question_and_info="q")


class TestGetSchema:
    def test_single(self):
        assert get_schema("single") is SingleOutput

    def test_table(self):
        assert get_schema("table") is TableOutput

    def test_invalid(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            get_schema("bad")


class TestGetTargetFields:
    def test_single(self):
        assert get_target_fields("single") == ["output"]

    def test_table(self):
        fields = get_target_fields("table")
        assert fields == [
            "refined_clinical_question_and_info",
            "organizational_information",
            "deleted_input",
        ]
