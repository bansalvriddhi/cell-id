#!/usr/bin/env python3
"""Final, read-only submission checks on both rendered IRIS PDFs."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pymupdf
from PIL import Image
from render_iris import SYNOPSIS_REPLACEMENTS

ROOT=Path(__file__).resolve().parents[1]
LIMITS=[('Abstract',250,250),('Introduction & Objective',100,150),('Innovation',50,100),
        ('Methodology',150,250),('Results & Conclusions',100,150),('Acknowledgement & References',50,100)]


def norm(text):return re.sub(r'\s+','',text)


def main():
    source=json.loads((ROOT/'report/content.json').read_text())
    pdfs={}
    counts={}
    synopsis_text=''
    paper_text=''
    for name,range_ in [('Synopsis',(1,5)),('Project_Paper',(5,40))]:
        path=ROOT/'report'/f'{name}.pdf'
        assert path.is_file() and path.stat().st_size>20000
        with pymupdf.open(path) as pdf:
            assert range_[0]<=len(pdf)<=range_[1]
            assert not pdf.metadata.get('author')
            text=' '.join(page.get_text(sort=True) for page in pdf)
            assert not re.search(r'Calcutta International School|bansalvriddhi|Vriddhi Bansal',text,re.I),name
            assert not re.search(r'\bCalcutta\b|\bKolkata\b|\bWest Bengal\b',text,re.I),name
            assert '\ufffd' not in text and '\u25a1' not in text and '\u200b' not in text
            pages=[]
            for i,page in enumerate(pdf):
                assert page.rect.width>590 and page.rect.height>830
                for block in page.get_text('dict')['blocks']:
                    x0,y0,x1,y1=block['bbox']
                    assert x0>=-0.5 and y0>=-0.5 and x1<=page.rect.width+0.5 and y1<=page.rect.height+0.5,(name,i+1,block['bbox'])
                pages.append({'number':i+1,'text_words':len(page.get_text().split()),'images':len(page.get_images(full=True))})
            if name=='Project_Paper':
                paper_text=text
                assert sum(p['images'] for p in pages)==4
                toc=pdf.get_toc()
                toc_text=' '.join(x[1] for x in toc)
                for heading in ['Introduction and Objectives','Materials and Methods','Results','Discussion','Conclusion','Acknowledgements','References']:
                    assert heading in toc_text,heading
                for snippet in ['Λ','∑','μ','ε','τ','N(0, I)','0.994967','0.994055','0.967533','0.844995']:
                    assert snippet in text,snippet
                assert 'Lambda_S' not in text and 'sum[p in S]' not in text
            else:
                synopsis_text=text
                assert all(p['images']==0 for p in pages)
            pdfs[name]={'path':str(path.relative_to(ROOT)),'pages':len(pdf),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'page_summary':pages,'anonymous':True}
    for sec,(heading,lo,hi) in zip(source['sections'],LIMITS):
        assert sec['title']==heading
        paragraphs=[]
        for p in sec['paragraphs']:
            for before,after in SYNOPSIS_REPLACEMENTS.items():p=p.replace(before,after)
            paragraphs.append(p)
            assert norm(p) in norm(synopsis_text),heading
        count=sum(any(c.isalnum() for c in word) for word in ' '.join(paragraphs).split())
        assert lo<=count<=hi,(heading,count)
        counts[heading]=count
    for before in SYNOPSIS_REPLACEMENTS:
        assert before not in synopsis_text
    paper=(ROOT/'report/Project_Paper.md').read_text()
    for name in ['figures/concept.png','figures/method_corrected.png',
                 'figures/paper/budget_curves.png','figures/paper/simple_baseline.png']:
        assert name in paper
        with Image.open(ROOT/name) as image:assert image.width>=1500 and image.height>=800
    assert sum(p['images'] for p in pdfs['Project_Paper']['page_summary'])==4
    assert 'normalized error' in paper
    result={'passed':True,'synopsis_section_words':counts,'pdfs':pdfs,
            'math':'typeset subscripts and superscripts checked in final PDF text',
            'figure_usage':'four colored conceptual/measured figures in project paper; none in synopsis',
            'limits':'layout, text, word counts, anonymity and recorded evidence; independent study eligibility and guide/video require applicant confirmation'}
    (ROOT/'report/submission_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS anonymous IRIS synopsis + project paper; lengths, source prose, four figures, equations and all word limits')
    print('WORD COUNTS',counts)
    print('PAGE COUNTS',{k:v['pages'] for k,v in pdfs.items()})


if __name__=='__main__':main()
