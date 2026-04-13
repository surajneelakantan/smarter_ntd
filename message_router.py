import json
import logging

log = logging.getLogger("message-router")


def _build_module_list(selected_modules):
    lines = []
    for i, mod in enumerate(selected_modules, 1):
        pdf_name = mod.get("pdf_name", "Unknown")
        course = mod.get("course", "Unknown Course")
        lines.append(f"  {i}. {pdf_name} ({course})")
    return "\n".join(lines)


def classify_message(user_message, selected_modules, llm_model):
    """Classify a user message into one of four categories."""
    from course_planner_updated import ollama_chat_json

    module_list = _build_module_list(selected_modules)

    prompt = f"""The user has these modules in their study plan:
{module_list}

The user said: "{user_message}"

Classify this message into exactly one of:
- module_qa: a question about course content or a specific module
- professional_problem: a real-world work problem the user wants help with
- new_plan_request: the user wants a different topic or a new plan
- general: greeting, thanks, meta-question, or off-topic

If module_qa, identify which module number (1-{len(selected_modules)}) is most relevant, or 0 if the question relates to the general topic but not a specific module. Also extract the main concept being asked about.

Respond with JSON only, no other text:
{{"type": "module_qa", "module_number": 0, "concept": "the concept"}}"""
    try:
        result = ollama_chat_json(llm_model, prompt, temp=0.1)
        
        msg_type = result.get("type", "general")
        if msg_type not in ("module_qa", "professional_problem", "new_plan_request", "general"):
            msg_type = "general"
        
        module_number = result.get("module_number", 0)
        concept = result.get("concept", "")
        
        log.info(f"Classified as: {msg_type}, module: {module_number}, concept: {concept}")
        
        return {
            "type": msg_type,
            "module_number": module_number,
            "concept": concept
        }
    except Exception as e:
        log.warning(f"Classification failed: {e}, defaulting to general")
        return {
            "type": "general",
            "module_number": 0,
            "concept": ""
        }