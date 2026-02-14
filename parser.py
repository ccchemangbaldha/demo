# parser.py
import pdfplumber, re, json
from typing import List, Dict
from collections import namedtuple

Block = namedtuple("Block", ["id", "subject","class_", "chapter", "section", "type", "seq", "text", "page"])

CHAPTER_RE = re.compile(r'^\s*chapter\s+(\d+)', re.I)
EXAMPLE_RE = re.compile(r'\bExample\s+(\d+)', re.I)
EXERCISE_RE = re.compile(r'\bEXERCISE\b|\bExercise\b', re.I)
SECTION_RE = re.compile(r'^\s*\d+(\.\d+)+')  # lines starting with 1.1 1.2 etc

def parse_pdf(path: str, subject="maths", class_=12):
    blocks = []
    cur_chapter = None
    cur_section = None
    buf = []
    block_type = "paragraph"
    seq = 0

    with pdfplumber.open(path) as pdf:
        for i,page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            lines = text.splitlines()
            for line in lines:
                # detect chapter
                ch = CHAPTER_RE.search(line)
                if ch:
                    # flush previous
                    if buf:
                        seq += 1
                        blocks.append(
                            Block(id=f"b{len(blocks)+1}",
                                  subject=subject,class_=class_,
                                  chapter=cur_chapter,section=cur_section,
                                  type=block_type,seq=seq,text="\n".join(buf),page=i)
                        )
                        buf=[]
                    cur_chapter = int(ch.group(1))
                    cur_section = None
                    block_type="chapter_title"
                    seq += 1
                    blocks.append(Block(id=f"b{len(blocks)+1}", subject=subject,class_=class_,
                                        chapter=cur_chapter,section=None,type=block_type,seq=seq,text=line,page=i))
                    block_type="paragraph"
                    continue

                # detect section
                sec = SECTION_RE.match(line.strip())
                if sec:
                    if buf:
                        seq += 1
                        blocks.append(Block(id=f"b{len(blocks)+1}",subject=subject,class_=class_,
                                            chapter=cur_chapter,section=cur_section,type=block_type,seq=seq,
                                            text="\n".join(buf),page=i))
                        buf=[]
                    cur_section = line.strip().split()[0]
                    seq += 1
                    blocks.append(Block(id=f"b{len(blocks)+1}",subject=subject,class_=class_,
                                        chapter=cur_chapter,section=cur_section,type="section_title",seq=seq,text=line,page=i))
                    block_type="paragraph"
                    continue

                # detect Example
                ex = EXAMPLE_RE.search(line)
                if ex:
                    # flush previous
                    if buf:
                        seq += 1
                        blocks.append(Block(id=f"b{len(blocks)+1}",subject=subject,class_=class_,
                                            chapter=cur_chapter,section=cur_section,type=block_type,seq=seq,
                                            text="\n".join(buf),page=i))
                        buf=[]
                    block_type="example"
                    seq += 1
                    blocks.append(Block(id=f"b{len(blocks)+1}",subject=subject,class_=class_,
                                        chapter=cur_chapter,section=cur_section,type="example_title",seq=seq,text=line,page=i))
                    block_type="example_body"
                    continue

                # detect exercise
                if EXERCISE_RE.search(line):
                    if buf:
                        seq += 1
                        blocks.append(Block(id=f"b{len(blocks)+1}",subject=subject,class_=class_,
                                            chapter=cur_chapter,section=cur_section,type=block_type,seq=seq,
                                            text="\n".join(buf),page=i))
                        buf=[]
                    seq += 1
                    blocks.append(Block(id=f"b{len(blocks)+1}",subject=subject,class_=class_,
                                        chapter=cur_chapter,section=cur_section,type="exercise_title",seq=seq,text=line,page=i))
                    block_type="exercise"
                    continue

                # otherwise accumulate
                buf.append(line)

            # end page
    # final flush
    if buf:
        seq += 1
        blocks.append(Block(id=f"b{len(blocks)+1}",subject=subject,class_=class_,
                            chapter=cur_chapter,section=cur_section,type=block_type,seq=seq,text="\n".join(buf),page=i))
    # convert to dicts
    return [b._asdict() for b in blocks]
