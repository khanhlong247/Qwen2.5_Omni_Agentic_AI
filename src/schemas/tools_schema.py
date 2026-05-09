import json

class InsuranceToolSchema:
    def __init__(self):
        self.functions_list = [
            {
                "name": "verify_policy",
                "description": "Verify the validity of an insurance policy. Requires both the customer's full name and the policy ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_name": {"type": "string", "description": "The full name of the policyholder."},
                        "policy_id": {"type": "string", "description": "The unique insurance policy identifier (e.g., HD277844)."}
                    },
                    "required": ["customer_name", "policy_id"]
                }
            },
            {
                "name": "process_medical_claim",
                "description": "Initialize a claim for medical/health issues such as sickness or hospitalization.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "policy_id": {"type": "string", "description": "The insurance policy ID."},
                        "hospital_name": {"type": "string", "description": "Name of the hospital where the treatment occurred."},
                        "admission_date": {"type": "string", "description": "Date of admission (Format: YYYY-MM-DD)."},
                        "diagnosis": {"type": "string", "description": "The medical diagnosis or condition description."}
                    },
                    "required": ["policy_id", "hospital_name", "admission_date", "diagnosis"]
                }
            },
            {
                "name": "process_accident_claim",
                "description": "Initialize a claim related to accidents, injuries, or physical trauma.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "policy_id": {"type": "string", "description": "The insurance policy ID."},
                        "incident_date": {"type": "string", "description": "The date the accident occurred (Format: YYYY-MM-DD)."},
                        "incident_description": {"type": "string", "description": "Brief description of how the accident happened."},
                        "injury_type": {"type": "string", "description": "Type of injury sustained (e.g., Fractured wrist)."}
                    },
                    "required": ["policy_id", "incident_date", "incident_description", "injury_type"]
                }
            }
        ]

    def get_qwen_xml_prompt(self) -> str:
        """Chuyển đổi danh sách hàm sang định dạng XML"""
        tools_xml = "<tools>\n"
        for tool in self.functions_list:
            tool_json = json.dumps({"type": "function", "function": tool}, ensure_ascii=False)
            tools_xml += tool_json + "\n"
        tools_xml += "</tools>"
        return tools_xml

    def get_available_functions_names(self) -> list:
        """Trả về danh sách tên các hàm đang có."""
        return [f["name"] for f in self.functions_list]