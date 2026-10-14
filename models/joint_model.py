from __future__ import annotations
import numpy as np
CLASSES=["gunshot","vehicle","helicopter","siren","human_voice"]
SUPPRESSION_POLICY={"gunshot":"preserve","vehicle":"preserve","helicopter":"suppress","siren":"preserve","human_voice":"preserve"}
class SharedEncoder:
    def __call__(self, features): return np.asarray(features).mean(axis=0,keepdims=True)
class ClassificationHead:
    def probabilities(self, features):
        z=np.asarray(features).mean(axis=tuple(range(np.asarray(features).ndim-1))); z=np.resize(z,len(CLASSES)); e=np.exp(z-z.max()); return e/e.sum()
class EnhancementHead:
    def apply(self, waveform, label): return np.zeros_like(waveform) if SUPPRESSION_POLICY[label]=="suppress" else np.asarray(waveform)
class TactEarJointModel:
    def __init__(self): self.encoder,self.classifier,self.enhancer=SharedEncoder(),ClassificationHead(),EnhancementHead()
    def infer(self,waveform):
        p=self.classifier.probabilities(self.encoder(waveform)); i=int(np.argmax(p)); return CLASSES[i],float(p[i]),self.enhancer.apply(waveform,CLASSES[i])
class MockAudioModel:
    """Explicit development backend; scenarios may supply the deterministic answer."""
    backend="MOCK"
    def infer(self, waveform, hint=None):
        label=(hint or "human_voice").lower(); return label, 0.92, np.asarray(waveform)
