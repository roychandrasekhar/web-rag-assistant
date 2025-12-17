import re

URL_REGEX = r"(https?://[^\s]+)"

def extract_url(text: str):
    match = re.search(URL_REGEX, text)
    return match.group(1) if match else None
