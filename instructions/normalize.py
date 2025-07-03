import re
import os
from collections import defaultdict
from project.helpers.helpers_fn import get_all_files

directories = 'data/text'
save_directory = 'data/structured'
os.makedirs(save_directory, exist_ok=True)
def parse_guide_text(text: str):
    structured = defaultdict(list)

    current_section = None
    current_subsection = None

    lines = text.split('\n')
    buffer = []

    def flush_buffer():
        nonlocal buffer
        content = '\n'.join(buffer).strip()
        if content:
            structured[(current_section, current_subsection)].append(content)
        buffer = []

    for line in lines:
        line = line.strip()

        # Section headers like: "1. Coaching Skills"
        section_match = re.match(r'^(\d+)\.\s+(.+)', line)
        subsection_match = re.match(r'^(\d+\.\d+)\.\s+(.+)', line)

        if subsection_match:
            flush_buffer()
            current_subsection = subsection_match.group(2)
        elif section_match:
            flush_buffer()
            current_section = section_match.group(2)
            current_subsection = None
        elif line == "ACTION" or line == "TIP":
            flush_buffer()
            current_subsection = line  # Use ACTION/TIP as pseudo-subsection
        else:
            buffer.append(line)

    flush_buffer()
    return structured
files = get_all_files(directories)
for file in files:
    if not file.endswith('.txt'):
        continue
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    structured = parse_guide_text(text)
    
    # Save structured content to a markdown file
    base_name = os.path.basename(file).replace('.txt', '.md')
    save_path = os.path.join(save_directory, base_name)
    
    with open(save_path, 'w', encoding='utf-8') as md_file:
        for (section, subsection), contents in structured.items():
            section_header = f"## {section}" if section else ""
            subsection_header = f"### {subsection}" if subsection else ""

            md_file.write(f"{section_header}\n{subsection_header}\n\n")
            for content in contents:
                md_file.write(f"{content}\n\n")