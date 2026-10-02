#!/usr/bin/env python3
"""Build the two anonymous IRIS submission PDFs from saved scientific sources."""
import hashlib
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

import pymupdf
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    PageBreak, KeepTogether, Image, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT=Path(__file__).resolve().parents[1]
W,H=A4
MARGIN=51
WIDTH=W-2*MARGIN


def setup():
    fonts=Path('/System/Library/Fonts/Supplemental')
    for name,file in [('Roman','Times New Roman.ttf'),('Bold','Times New Roman Bold.ttf'),('Italic','Times New Roman Italic.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('Roman',normal='Roman',bold='Bold',italic='Italic',boldItalic='Bold')


def style(name,size=11,leading=14,font='Roman',align=TA_JUSTIFY,after=7):
    return ParagraphStyle(name,fontName=font,fontSize=size,leading=leading,alignment=align,
        textColor=colors.black,spaceAfter=after,spaceBefore=0,spaceShrinkage=0,
        allowWidows=0,allowOrphans=0,splitLongWords=1)


ST={}
SYNOPSIS_REPLACEMENTS={}
def markup(text):
    text=escape(text)
    text=re.sub(r'`([^`]+)`',r'\1',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'(?<!\*)\*([^*]+)\*',r'<i>\1</i>',text)
    return text


class Document(BaseDocTemplate):
    def __init__(self,path,title,**kw):
        super().__init__(str(path),pagesize=A4,leftMargin=MARGIN,rightMargin=MARGIN,
            topMargin=47,bottomMargin=43,title=title,author='',subject='Computational Biology and Bioinformatics',**kw)
        frame=Frame(MARGIN,43,WIDTH,H-90,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates(PageTemplate(id='main',frames=[frame],onPage=self.page))
        self.heading_counter=0
    def beforeDocument(self):
        self.heading_counter=0
    def page(self,canvas,doc):
        canvas.setFillGray(0);canvas.setFont('Roman',9)
        canvas.drawCentredString(W/2,22,str(doc.page))
    def afterFlowable(self,item):
        if isinstance(item,Paragraph) and item.style.name in ('H1','H2'):
            self.heading_counter+=1
            level=0 if item.style.name=='H1' else 1
            key=f'section-{self.heading_counter}'
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(item.getPlainText(),key,level=level,closed=False)
            self.notify('TOCEntry',(level,item.getPlainText(),self.page_number(),key))
    def page_number(self):return self.page


def figure(path,caption):
    with PILImage.open(path) as im:iw,ih=im.size
    maxheight=200 if 'concept' in path.name else 246
    scale=min(WIDTH/iw,maxheight/ih)
    return KeepTogether([Spacer(1,5),Image(str(path),iw*scale,ih*scale),Spacer(1,5),Paragraph(markup(caption),ST['caption']),Spacer(1,7)])


# Each display equation in Project_Paper.md is typeset as vector PDF text using
# ReportLab's subscript/superscript markup; unknown TeX expressions fail closed.
EQUATIONS={
    r'y_p = \mu_p + A_p z + \varepsilon_p,\quad z\sim\mathcal{N}(0,I),\quad \varepsilon_p\sim\mathcal{N}(0,\tau^2 I).':(
        'y<sub>p</sub> = μ<sub>p</sub> + A<sub>p</sub> z + ε<sub>p</sub>,   '
        'z ~ N(0, I),   ε<sub>p</sub> ~ N(0, τ<super>2</super> I).',
    ),
    r'\Lambda_S=I+\frac{1}{\tau^2}\sum_{p\in S}A_{p,\mathrm{obs}}^\top A_{p,\mathrm{obs}},':(
        'Λ<sub>S</sub> = I + (1/τ<super>2</super>) '
        '∑<sub>p in S</sub> A<sub>p,obs</sub><super>T</super> A<sub>p,obs</sub>,',
    ),
    r'b_S=\frac{1}{\tau^2}\sum_{p\in S} A_{p,\mathrm{obs}}^\top\left(y_{p,\mathrm{obs}}-\mu_{p,\mathrm{obs}}\right), \qquad m_S=\Lambda_S^{-1}b_S,\qquad \mathbb{E}[y_q\mid S]=\mu_q+A_qm_S.':(
        'b<sub>S</sub> = (1/τ<super>2</super>) ∑<sub>p in S</sub> '
        'A<sub>p,obs</sub><super>T</super>(y<sub>p,obs</sub> − μ<sub>p,obs</sub>),',
        'm<sub>S</sub> = Λ<sub>S</sub><super>−1</super>b<sub>S</sub>,    '
        'E[y<sub>q</sub> | S] = μ<sub>q</sub> + A<sub>q</sub>m<sub>S</sub>.',
    ),
}


def inline_math(text):
    """Readable, precise inline variables rather than literal TeX commands."""
    convert={r'\(y_p\)':'y<sub>p</sub>',r'\(p\)':'p',r'\(\mu_p\)':'μ<sub>p</sub>',
             r'\(A_p\)':'A<sub>p</sub>',r'\(z\)':'z',r'\(\tau\)':'τ',r'\(S\)':'S'}
    text=markup(text)
    for original,replacement in convert.items():text=text.replace(original,replacement)
    if '\\(' in text or '\\)' in text:raise ValueError(f'Unhandled inline mathematical expression: {text}')
    return text


def synopsis(content):
    nodes=[Paragraph(escape(content['title']),ST['title']),Paragraph(escape(content['subtitle']),ST['subtitle']),
        Paragraph('Project Synopsis · Computational Biology &amp; Bioinformatics',ST['meta']),Spacer(1,8)]
    # Two concise text pages, retaining the requested six exact section word limits.
    for i,s in enumerate(content['sections']):
        if i==3:nodes.append(PageBreak())
        nodes.append(Paragraph(f'{i+1}. {escape(s["title"])}',ST['h1']))
        paragraphs=[]
        for p in s['paragraphs']:
            for before,after in SYNOPSIS_REPLACEMENTS.items():p=p.replace(before,after)
            paragraphs.append(p)
            nodes.append(Paragraph(markup(p),ST['body']))
        count=sum(any(ch.isalnum() for ch in token) for token in ' '.join(paragraphs).split())
        lo,hi=[(250,250),(100,150),(50,100),(150,250),(100,150),(50,100)][i]
        if not lo<=count<=hi:raise ValueError(f'Synopsis section {i+1} has {count} words, expected {lo}–{hi}')
    Document(ROOT/'report/Synopsis.pdf',content['title']).build(nodes)


def parse_paper(path):
    lines=path.read_text().splitlines()
    nodes=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line or line=='---':i+=1;continue
        if line=='$$':
            i+=1;math=[]
            while i<len(lines) and lines[i].strip()!='$$':
                math.append(lines[i].strip());i+=1
            if i==len(lines):raise ValueError('Unterminated display equation')
            expression=' '.join(math)
            if expression not in EQUATIONS:raise ValueError('Unknown display equation: '+expression)
            for rendered in EQUATIONS[expression]:
                nodes.append(Paragraph(rendered,ST['equation']))
            i+=1;continue
        if line.startswith('# '):i+=1;continue
        if line.startswith('### '):nodes.append(Paragraph(markup(line[4:]),ST['h2']));i+=1;continue
        if line.startswith('## '):nodes.append(Paragraph(markup(line[3:]),ST['h1']));i+=1;continue
        match=re.match(r'!\[(.*)\]\((.*)\)',line)
        if match:
            caption,relative=match.groups();image=(path.parent/relative).resolve()
            nodes.append(figure(image,caption));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\-\s]+',v or '-') for v in row):rows.append(row)
                i+=1
            n=len(rows[0]); widths=[WIDTH/n]*n
            if n==5:widths=[WIDTH*.25]+[WIDTH*.75/4]*4
            cells=[[Paragraph(markup(v),ST['table']) for v in row] for row in rows]
            table=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),.7,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            nodes.extend([Spacer(1,4),table,Spacer(1,9)]);continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','![','|')):
            para.append(lines[i].strip());i+=1
        body=' '.join(para)
        if body.startswith('- '):body='• '+body[2:]
        nodes.append(Paragraph(inline_math(body),ST['body']))
    return nodes


def project(content):
    cover=[Spacer(1,108),Paragraph(escape(content['title']),ST['cover']),Spacer(1,12),
        Paragraph(escape(content['subtitle']),ST['subtitle']),Spacer(1,28),
        Paragraph('Project Paper',ST['subtitle']),
        Paragraph('Computational Biology &amp; Bioinformatics',ST['meta']),Spacer(1,26),
        Paragraph('A retrospective experimental-design study of published cellular perturbation signatures',ST['meta']),PageBreak()]
    toc=TableOfContents();toc.levelStyles=[style('toc0',11,17,align=TA_LEFT,after=4),style('toc1',10,14,align=TA_LEFT,after=2)]
    toc.levelStyles[1].leftIndent=14
    nodes=cover+[Paragraph('Contents',ST['toc_heading']),Spacer(1,12),toc,PageBreak()]
    nodes+=parse_paper(ROOT/'report/Project_Paper.md')
    Document(ROOT/'report/Project_Paper.pdf',content['title']).multiBuild(nodes)


def main():
    setup()
    ST.update({'body':style('Body'),'title':style('Title',18,21,'Bold',TA_CENTER,7),
        'cover':style('Cover',24,29,'Bold',TA_CENTER,10),'subtitle':style('Subtitle',12,16,align=TA_CENTER),
        'meta':style('Meta',10.5,14,align=TA_CENTER),'h1':style('H1',12.5,17,'Bold',TA_LEFT,7),
        'h2':style('H2',11.5,15,'Bold',TA_LEFT,6),'caption':style('Caption',9,11,align=TA_LEFT,after=7),
        'table':style('Table',9,11,align=TA_LEFT,after=0),'toc_heading':style('TOC',17,21,'Bold',TA_LEFT)})
    ST['equation']=style('Equation',10.5,17,'Roman',TA_CENTER,after=7)
    ST['h1'].spaceBefore=10;ST['h1'].keepWithNext=True
    ST['h2'].spaceBefore=8;ST['h2'].keepWithNext=True
    content=json.loads((ROOT/'report/content.json').read_text())
    synopsis(content)
    project(content)
    for name in ['Synopsis','Project_Paper']:
        p=ROOT/f'report/{name}.pdf'
        with pymupdf.open(p) as doc:print(name,'pages',len(doc),'bytes',p.stat().st_size)


if __name__=='__main__':main()
