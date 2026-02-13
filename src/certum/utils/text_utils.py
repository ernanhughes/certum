import re

def clean_wiki_markup(text: str) -> str:
    # [[Page|Display]] → Display
    text = re.sub(r"\[\[[^\|\]]+\|([^\]]+)\]\]", r"\1", text)

    # [[Page]] → Page
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # Replace underscores with spaces
    text = text.replace("_", " ")

    return text
