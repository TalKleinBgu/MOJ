import stanza
import re
import sys
from docx.text.paragraph import Paragraph
from docx.document import Document
from docx.table import _Cell, Table
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
sys.path.append('../../')
stanza.download('he')
nlp = stanza.Pipeline('he')

# Modify property of Paragraph.text to include hyperlink text
Paragraph.text = property(lambda self: get_paragraph_text(self))

def get_paragraph_text(paragraph) -> str:
    """
    Extract text from paragraph, including hyperlink text.
    """
    def get_xml_tag(element):
        return "%s:%s" % (element.prefix, re.match("{.*}(.*)", element.tag).group(1))

    text_content = ''
    run_count = 0
    for child in paragraph._p:
        tag = get_xml_tag(child)
        if tag == "w:r":
            text_content += paragraph.runs[run_count].text
            run_count += 1
        if tag == "w:hyperlink":
            for sub_child in child:
                if get_xml_tag(sub_child) == "w:r":
                    text_content += sub_child.text
    return text_content

def is_block_bold(block) -> bool:
    """
    Check if the entire block/paragraph text is bold.
    """
    if block.runs:
        for run in block.runs:
            if run.bold:
                return True
    return False

def iterate_block_items(parent):
    """
    considering both top-level paragraphs/tables and those nested within cells.
    """
    if isinstance(parent, Document):
        parent_element = parent.element.body
    elif isinstance(parent, _Cell):
        parent_element = parent._tc
    else:
        raise ValueError("Unsupported parent type.")

    for child in parent_element.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            table = Table(child, parent)
            for row in table.rows:
                for cell in row.cells:
                    yield from iterate_block_items(cell)

def extract_part_after_number_or_hebrew_letter(sentence: str) -> str:
    """
    Extract text following a pattern of number or Hebrew letter.
    """
    pattern = r'^(?:[0-9\u05D0-\u05EA]+)\.\s*(.*)'
    match = re.search(pattern, sentence)
    return match.group(1).strip() if match else sentence

def count_patterns_in_block(block) -> int:
    """
    Count the number-dot or dot-number patterns in a block.
    """
    pattern = r'\s*(?:\.\d+|\d+\.)'
    return len(re.findall(pattern, block.text))

def count_consecutive_blocks_starting_with_number(blocks) -> int:
    """
    Count consecutive blocks starting with a number or Hebrew letter.
    """
    count = 0
    for block in blocks:
        if 'הנאשם' in block.text:
            return 1
        count += count_patterns_in_block(block)
        if 'חקיקה שאוזכרה' in block.text:
            break
    return count

def extract_name_after_word(text: str, word: str) -> str:
    """
    Extract the words following a given word up to the end of the sentence.
    """
    pattern = re.compile(fr'{word}(?:,)?\s*([\u0590-\u05FF\s\'\(\)-]+)')
    match = pattern.search(text)
    return match.group(1) if match else ''

def extract_violations(text: str) -> list:
    """
    Extract violations from the text based on a pre-defined pattern.
    """

    matches = re.findall(r"(?:סעיף|סעיפים|ס'|סע')\s*\d+\s*(?:\([\s\S]*?\))?.*?(?=\s*(?:ב|ל)(?:חוק|פקודת))\s*(?:ב|ל)(?:חוק|פקודת)\s*ה?(?:עונשין|כניסה לישראל|סמים\s+המסוכנים|\w+)?", text)
    # matches = re.findall(r"(?:סעיף|סעיפים|ס'|סע')\s*\d+\s*(?:\([\s\S]*?\))?.*?(?=\s*(?:ב|ל)(?:חוק|פקודת))\s*(?:ב|ל)(?:חוק|פקודת)\s*ה?(?:עונשין|כניסה לישראל|סמים\s+המסוכנים|[^\[]+)?", text)

    matches = [match.strip() for match in matches]
    return matches
