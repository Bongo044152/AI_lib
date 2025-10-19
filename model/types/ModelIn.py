import pydantic
from typing import *


class ModelIn(pydantic.BaseModel):
    content: Union[str, List[Dict[str, str]]]
    system_prompt: Optional[str] = None
    thinking: bool = False

    @pydantic.field_validator("content")
    def _check_content(cls, content):
        if isinstance(content, str):
            return content
        else:
            assert len(content)

            for item in content:
                role = item.get("role")
                context = item.get("content")

                if not role:
                    raise ValueError("field role is required")
                elif not context:
                    raise ValueError("field content is required")
                elif role not in ["user", "assistant"]:
                    raise ValueError("role should be user, assistant")
            assert content[-1]["role"] == "user"
            return content


# # example of ModelIn -> if array type
# [
#     {
#         "role": "user",
#         "content": "message"
#     }
# ]

# # example of ModelIn -> if string only
# "some messagee here ... "
