import math
import pandas as pd


def _clean_summary(value): 
    if value is None:
        return "No summary available."
    if isinstance(value, float) and math.isnan(value):
        return "No summary available"
    text = str(value).strip()
    if len(text) == 0:
        return "No summary available"

    return text


def build_single_module_context(module_idx, df, user_format="both"):
    row = df.iloc[module_idx]
    pdf_name = row.get("pdf_name", "")
    video_name = row.get("video_related_to_pdf", "")
    course = row.get("course_name", "Unknown Course")
    pdf_summary = _clean_summary(row.get("pdf_summary", ""))
    video_summary = _clean_summary(row.get("video_transcription_summary", ""))

    if user_format == "video" and video_name:
        title = video_name
    elif pdf_name:
        title = pdf_name
    elif video_name:
        title = video_name
    else:
        title = "Unknown Module"
    block = f"""=== MODULE: {title} === 
Course: {course}
PDF Summary: {pdf_summary}
Video Summary: {video_summary}
=== END MODULE ==="""

    return block

def build_knowledge_window(selected_modules, df, user_format="both"):

    if not selected_modules:
        return ""

    blocks = []
    for i, mod in enumerate(selected_modules, 1):
        module_idx = mod["module_idx"]
        block = build_single_module_context(module_idx, df, user_format)
        numbered_block = block.replace("=== MODULE:", f"=== MODULE {i}:")
        blocks.append(numbered_block)

    return "\n\n".join(blocks)