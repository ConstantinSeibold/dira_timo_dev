from pydantic import BaseModel, create_model


def create_dynamic_schema(fields: list[str]) -> type[BaseModel]:
    """Create a Pydantic model dynamically from a list of field names.

    All fields are required strings.
    """
    field_definitions = {f: (str, ...) for f in fields}
    return create_model("DynamicOutput", **field_definitions)
