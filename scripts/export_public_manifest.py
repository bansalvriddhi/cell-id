#!/usr/bin/env python3
"""Export pinned public data URLs and SHA-256 hashes without local machine paths."""
import hashlib
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'data/source_manifest.tsv'
    original=pd.read_csv(source,sep='\t',keep_default_na=False)
    required=['filename','url','bytes','sha256','revision']
    assert all(key in original for key in required)
    assert original.filename.is_unique
    assert original.url.str.startswith('https://').all()
    for _,row in original.iterrows():
        path=ROOT/'data/source'/row.filename
        assert path.is_file() and path.stat().st_size==row.bytes
        h=hashlib.sha256(path.read_bytes()).hexdigest()
        assert h==row.sha256,(row.filename,h)
    public=original[required].copy()
    public['relative_path']='data/source/'+public['filename']
    cols=['relative_path',*required]
    dest=ROOT/'data/source_manifest_public.tsv'
    public[cols].to_csv(dest,sep='\t',index=False)
    assert not public[cols].astype(str).apply(lambda c:c.str.contains(str(ROOT),regex=False)).any().any()
    print(f'Exported {len(public)} verified upstream files, {int(public.bytes.sum())} source bytes: {dest.relative_to(ROOT)}')


if __name__=='__main__':main()
