"""Two-channel physical delay estimation with explicit observability limits."""
import numpy as np

def spatial_estimate(left, right, sample_rate, channels=1, spacing_m=None):
    result = {"status":"UNAVAILABLE", "reason":"Mono input cannot reveal direction",
              "azimuth_deg":None, "delay_us":None, "quality":None, "balance_db":None}
    if channels != 2: return result
    x,y=np.asarray(left,float),np.asarray(right,float)
    if x.shape!=y.shape or len(x)<64 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return {**result,"reason":"Invalid stereo data"}
    x,y=x-x.mean(),y-y.mean()
    ex,ey=float(np.mean(x*x)),float(np.mean(y*y))
    if min(ex,ey)<1e-10: return {**result,"reason":"One or both channels are silent"}
    result["balance_db"]=float(10*np.log10(ex/ey))
    correlation=float(np.dot(x,y)/np.sqrt(np.dot(x,x)*np.dot(y,y)))
    if correlation > .9999:
        return {**result,"reason":"Channels are indistinguishable: duplicated mono or unresolved centre"}
    if spacing_m is None:
        return {**result,"reason":"Stereo detected; enter measured microphone spacing to enable bearing"}
    if not .01<=spacing_m<=2: raise ValueError("Spacing must be between 0.01 and 2 metres")
    n=1 << (2*len(x)-1).bit_length()
    spectrum=np.fft.rfft(y,n)*np.conj(np.fft.rfft(x,n))
    cc=np.fft.irfft(spectrum/np.maximum(abs(spectrum),1e-12),n)
    limit=max(1,int(np.ceil(sample_rate*spacing_m/343)))
    region=np.abs(np.concatenate((cc[-limit:],cc[:limit+1])))
    index=int(np.argmax(region))
    shift=index-limit
    # Isolated broadband peak requirement rejects periodic/ambiguous correlations.
    remaining=region.copy()
    remaining[max(0,index-1):index+2]=0
    ratio=float(region[index]/max(float(remaining.max()),1e-12))
    peak=float(region[index])
    result.update(delay_us=shift/sample_rate*1e6, quality=min(1.,max(0.,1-1/max(ratio,1))))
    if ratio<2 or peak<.05:
        return {**result,"status":"AMBIGUOUS","reason":"No isolated delay peak; reverberation, periodic sound or multiple sources"}
    angle=float(np.degrees(np.arcsin(np.clip(-shift/sample_rate*343/spacing_m,-1,1))))
    return {**result,"status":"ESTIMATE","azimuth_deg":angle,
            "reason":"Microphone-relative only; front/back ambiguous. Assumes channel 1 left, channel 2 right. Not class-specific."}
