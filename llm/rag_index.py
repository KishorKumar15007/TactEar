from __future__ import annotations
from pathlib import Path
import re, json
from tactear.types import SourceChunk
class LocalRAG:
    def __init__(self, chunks=None): self.chunks=chunks or []
    @classmethod
    def from_directory(cls,path):
        chunks=[]
        for p in Path(path).glob("**/*"):
            if p.suffix.lower() not in (".txt",".md",".json"): continue
            text=json.dumps(json.loads(p.read_text(encoding="utf8"))) if p.suffix==".json" else p.read_text(encoding="utf8")
            for i,part in enumerate(re.split(r"\n\s*\n",text)):
                if part.strip(): chunks.append(SourceChunk(p.name,str(i+1),part.strip(),0))
        return cls(chunks)
    def query(self,text,top_k=3,min_score=0.05):
        terms=set(re.findall(r"\w+",text.lower())); scored=[]
        for c in self.chunks:
            words=set(re.findall(r"\w+",c.text.lower())); score=len(terms&words)/max(1,len(terms))
            if score>=min_score: scored.append(SourceChunk(c.source,c.section,c.text,score))
        return sorted(scored,key=lambda c:c.score,reverse=True)[:top_k]
