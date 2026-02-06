from pydantic import BaseModel


class SingleOutput(BaseModel):
    output: str


class TableOutput(BaseModel):
    refined_clinical_question_and_info: str
    organizational_information: str
    deleted_input: str


def get_schema(mode: str) -> type[BaseModel]:
    """Return the Pydantic model class for the given mode."""
    if mode == "single":
        return SingleOutput
    elif mode == "table":
        return TableOutput
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Must be 'single' or 'table'.")


def get_target_fields(mode: str) -> list[str]:
    """Return field names matching CSV target columns for the given mode."""
    schema = get_schema(mode)
    return list(schema.model_fields.keys())
