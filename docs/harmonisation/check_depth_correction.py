"""Do the published wave speeds and firn corrections explain the depth offsets between datasets?
(plan.md, 3.2). Run explore.py first, which writes the pairs of layers at the crossovers.
"""
import json, numpy as np
rows=json.load(open("/tmp/anta_pairs.json"))
# as stored in the published files: (wave speed, firn)
P={"Bodart_2021":(168.5,10),"Bodart_2025a":(169.5,13),"Bodart_2025b":(168.5,14.6),"Cavitte_2020":(168.5,14.6),"Chung_2023":(168.5,10),"Franke_2025":(169,13),"Mulvaney_2023":(168.5,10),"Napoleoni_2026":(168.5,10),"Sanderson_2024":(168.5,12),"Wang_2023":(168.5,15.5),"Winter_2018":(168.5,13),"Yan_2025":(168.5,0),"Beem_2021":(167,0)}
best={}
for r in rows:
    if r["A"] not in P or r["B"] not in P or r["n"]<5: continue
    if abs(r["ageB"]-r["ageA"])>0.03*max(r["ageA"],r["ageB"]): continue
    if abs(r["dz_med"])>0.05*r["depth"]: continue
    k=(r["A"],r["irhA"],r["B"])
    if k not in best or abs(r["dz_med"])<abs(best[k]["dz_med"]): best[k]=r
o=[]
print(f"{'A':15s} {'B':15s} {'age ka':>7s} {'depth':>6s} {'n':>5s} {'measured B-A':>12s} {'predicted':>9s} {'residual':>8s}")
for k,r in sorted(best.items()):
    va,fa=P[r["A"]]; vb,fb=P[r["B"]]; z=r["depth"]; t=(z-fa)/va
    pred=(vb*t+fb)-z
    o.append((r["A"],r["B"],r["dz_med"],pred,r["n"]))
    print(f"{r['A']:15s} {r['B']:15s} {r['ageA']/1e3:7.1f} {z:6.0f} {r['n']:5d} {r['dz_med']:+12.1f} {pred:+9.1f} {r['dz_med']-pred:+8.1f}")
d=np.array([x[2] for x in o]); p=np.array([x[3] for x in o])
print("\nlayer pairs",len(o),"| median |measured|",np.median(abs(d)).round(1),"| median |residual|",np.median(abs(d-p)).round(1),"| RMS measured",np.sqrt((d**2).mean()).round(1),"| RMS residual",np.sqrt(((d-p)**2).mean()).round(1),"| correlation",np.corrcoef(d,p)[0,1].round(2))
