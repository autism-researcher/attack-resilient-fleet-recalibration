import json,os,numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11,"hatch.linewidth":0.6,"axes.axisbelow":True})
BASE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(BASE,"figures_bw"); os.makedirs(OUT,exist_ok=True)
def J(p): return json.load(open(os.path.join(BASE,p)))
def ci(t): m,lo,hi=t[:3]; return m,[[m-lo],[hi-m]]
# B&W-safe styles
BAR4={  # B0/B1/B2/P5
 0:dict(facecolor="#eaeaea",hatch="...."),
 1:dict(facecolor="white",  hatch="////"),
 2:dict(facecolor="#bdbdbd",hatch="\\\\\\\\"),
 3:dict(facecolor="#7a7a7a",hatch="xxxx"),
}
def bar(ax,xs,vals,labels,styleidx,title,ylabel=None):
    for i,(x,v) in enumerate(zip(xs,vals)):
        m,ye=ci(v)
        ax.bar(x,m,width=0.7,yerr=ye,capsize=4,edgecolor="black",linewidth=0.8,
               error_kw=dict(ecolor="black",lw=1),**BAR4[styleidx[i]])
    ax.set_xticks(xs); ax.set_xticklabels(labels); ax.set_title(title); ax.grid(axis="y",alpha=0.35)
    if ylabel: ax.set_ylabel(ylabel)

def figA(src,out,suptitle):
    d=J(src); conds=["B0","B1","B2","P5"]; labs=["B0 frozen","B1 local","B2 naive","P5 fleet"]
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    bar(ax[0],range(4),[d[c]["rate_err"] for c in conds],labs,[0,1,2,3],"Intervention-rate error $|$rate$-\\tau|$")
    bar(ax[1],range(4),[d[c]["missed"] for c in conds],labs,[0,1,2,3],"Missed high-risk events (rate)")
    for a in ax: a.tick_params(axis="x",rotation=15)
    fig.suptitle(suptitle); fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(os.path.join(OUT,out),dpi=150); plt.close(fig)

def figC(src,out,title):
    d=J(src); groups=["B1","B2","P5"]; glab=["B1 local","B2 naive","P5 fleet"]
    x=np.arange(3); w=0.38
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    for off,key,lab,st in [(-w/2,"honest","honest fleet",dict(facecolor="white",hatch="////")),
                            (w/2,"adv","30% adversarial uploads",dict(facecolor="#9e9e9e",hatch="xxxx"))]:
        for i,g in enumerate(groups):
            m,ye=ci(d[g][key]); ax.bar(x[i]+off,m,w,yerr=ye,capsize=3,edgecolor="black",linewidth=0.8,
                error_kw=dict(ecolor="black",lw=1),label=lab if i==0 else None,**st)
    b=d["theoretical_bound"][0]
    ax.axhline(b,ls=(0,(6,4)),color="#333333",lw=2,label=f"proven bound = {b:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(glab); ax.set_ylabel(r"max under-protection $(B^{eff}-B^\star)_+$")
    ax.set_title(title); ax.grid(axis="y",alpha=0.35); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,out),dpi=150); plt.close(fig)

def figReal(out):
    styles=[("highd","highD","-","o"),("waymo","waymo","--","s"),("ngsim","ngsim","-.","^")]
    fig,ax=plt.subplots(figsize=(8,5))
    for name,lab,ls,mk in styles:
        x=np.sort(np.load(os.path.join(BASE,"real",name+".npy")))
        y=np.arange(1,len(x)+1)/len(x); Bstar=np.quantile(x,0.90)
        ax.plot(x,y,ls=ls,color="black",lw=1.7,marker=mk,markevery=600,markersize=6,
                markerfacecolor="white",label=f"{lab} ($B^\\star$={Bstar:.2f})")
        ax.axvline(Bstar,ls=ls,color="#666666",lw=1.1)
    ax.axhline(0.90,ls=":",color="#999999",lw=1); ax.text(0.02,0.915,r"$1-\tau=0.90$",fontsize=9,color="#555")
    ax.set_xlabel(r"per-trajectory max risk $R_{max}$"); ax.set_ylabel("empirical CDF")
    ax.set_title("Real per-corpus risk distributions and matching boundaries")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,out),dpi=150); plt.close(fig)

def figClosed(out):
    d=J("results_real/closed_loop.json"); C=d["conditions"]
    order=["AEB","B0","B1","P5"]; labs=["AEB\n(fixed TTC)","B0\nfrozen","B1\nlocal","P5\nfleet"]
    fig,ax=plt.subplots(1,2,figsize=(13,4.6))
    bar(ax[0],range(4),[C[c]["coll"] for c in order],labs,[0,1,2,3],"Closed-loop safety vs baseline",
        ylabel="rear-end collisions / 1000 veh-steps")
    v=d["v2x"]; vk=["lat0ms_loss0","lat300ms_loss20","lat1000ms_loss50"]
    xt=["ideal\n(0 ms, 0%)","300 ms\n20% loss","1000 ms\n50% loss"]; x=range(3)
    up=[v[k]["under_max"][0] for k in vk]; co=[v[k]["coll"][0] for k in vk]
    a=ax[1]; a.plot(x,up,"-o",color="black",lw=1.8,markersize=7,markerfacecolor="white",label="max under-protection")
    a.axhline(0.12,ls=":",color="#888888",lw=1.4,label=r"proven bound $\approx0.12$")
    a.set_ylim(0,0.16); a.set_ylabel("worst-case under-protection"); a.set_xticks(list(x)); a.set_xticklabels(xt)
    a.set_title("P5 under V2X latency / packet loss"); a.grid(alpha=0.3)
    a2=a.twinx(); a2.plot(x,co,"--s",color="#555555",lw=1.8,markersize=7,markerfacecolor="white",label="collisions / 1000")
    a2.set_ylim(0,0.6); a2.set_ylabel("collisions / 1000 veh-steps")
    l1,lab1=a.get_legend_handles_labels(); l2,lab2=a2.get_legend_handles_labels()
    a.legend(l1+l2,lab1+lab2,fontsize=8,loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,out),dpi=150); plt.close(fig)

figA("results/A.json","expA_accuracy_safety.png","Experiment A - fleet=20, heterogeneous shift")
figA("results_real/A.json","expA_real.png","Experiment A (real cross-dataset) - fleet=20")
figC("results/C.json","expC_underprotection.png","Experiment C - safety under adversarial fleet uploads")
figC("results_real/C.json","expC_real.png","Experiment C (real cross-dataset)")
figReal("real_dist.png")
figClosed("closed_loop.png")
print("generated:",sorted(os.listdir(OUT)))
# grayscale contact sheet for verification
from PIL import Image
ims=["expA_accuracy_safety.png","expC_underprotection.png","real_dist.png","closed_loop.png"]
tiles=[Image.open(os.path.join(OUT,i)).convert("L") for i in ims]
w=max(t.width for t in tiles); H=sum(int(t.height*w/t.width) for t in tiles)
sheet=Image.new("L",(w,H),255); y=0
for t in tiles:
    t2=t.resize((w,int(t.height*w/t.width))); sheet.paste(t2,(0,y)); y+=t2.height
sheet.save(os.path.join(OUT,"_GRAYSCALE_CHECK.png")); print("wrote grayscale check")

def figLatency(out="adaptation_latency.png"):
    import numpy as np
    B=J("results_real/B.json"); P=J("results_real/penetration.json")
    Ks=[1,5,20,100]
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    # (a) latency vs fleet size: local-only vs collab
    for c,ls,mk,lab in [("B1","--","s","B1 local-only"),("P5","-","o","P5 collab+isolation")]:
        m=[B[c][str(k)][0] for k in Ks]; lo=[B[c][str(k)][1] for k in Ks]; hi=[B[c][str(k)][2] for k in Ks]
        ax[0].plot(Ks,m,ls=ls,marker=mk,color="black",lw=1.8,markersize=7,markerfacecolor="white",label=lab)
        ax[0].fill_between(Ks,lo,hi,color="0.6",alpha=0.25)
    ax[0].set_xscale("log"); ax[0].set_xticks(Ks); ax[0].set_xticklabels(Ks)
    ax[0].set_xlabel("fleet size $K$"); ax[0].set_ylabel("adaptation latency (steps to target rate)")
    ax[0].set_title("(a) Collaboration shortens adaptation"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=9)
    # (b) latency vs penetration (collab), flat
    ps=[0.10,0.25,0.50,0.75,1.00]; lat=[P[f"{p:.2f}"]["latency_P5"] for p in ps]
    ax[1].plot([p*100 for p in ps],lat,"-o",color="black",lw=1.8,markersize=7,markerfacecolor="white",label="P5 collab+isolation")
    ax[1].set_ylim(0,max(lat)*1.35); ax[1].set_xlabel("equipped market penetration (%)")
    ax[1].set_ylabel("adaptation latency (steps)")
    ax[1].set_title("(b) Benefit is penetration-independent"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,out),dpi=150); plt.close(fig)
    # grayscale check
    from PIL import Image
    Image.open(os.path.join(OUT,out)).convert("L").save(os.path.join(OUT,"_LAT_GRAY.png"))
    print("wrote",out)

figLatency()

def figLDP(out="privacy_utility.png"):
    d=J("results_real/expI_ldp.json")
    eps=[0.5,1.0,2.0]; ekeys={0.5:"eps0.5",1.0:"eps1.0",2.0:"eps2.0"}
    styles=[("K5","$K=5$","-","o"),("K20","$K=20$","--","s"),("K100","$K=100$","-.","^")]
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for Kk,lab,ls,mk in styles:
        dev=[d[Kk][ekeys[e]]["rate_err_dev_vs_noisefree"] for e in eps]
        ax[0].plot(eps,dev,ls=ls,marker=mk,color="black",lw=1.7,markersize=7,markerfacecolor="white",label=lab)
    ax[0].set_xlabel(r"LDP budget $\varepsilon_0$ (smaller = more private)"); ax[0].set_ylabel("rate-error deviation vs. noise-free")
    ax[0].set_title("(a) Privacy cost to calibration accuracy"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=9); ax[0].invert_xaxis()
    for Kk,lab,ls,mk in styles:
        um=[d[Kk][ekeys[e]]["under_max"] for e in eps]
        ax[1].plot(eps,um,ls=ls,marker=mk,color="black",lw=1.7,markersize=7,markerfacecolor="white",label=lab)
    bnds=[d[Kk][ekeys[e]]["bound"] for Kk,_,_,_ in styles for e in eps]
    ax[1].axhline(max(bnds),ls=":",color="#888888",lw=1.4,label=f"proven bound $\\approx${max(bnds):.3f}")
    ax[1].set_ylim(0,max(bnds)*1.25); ax[1].set_xlabel(r"LDP budget $\varepsilon_0$"); ax[1].set_ylabel(r"worst-case under-protection $(B^{eff}-B^\star)_+$")
    ax[1].set_title("(b) Safety holds under privacy noise"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8); ax[1].invert_xaxis()
    fig.tight_layout(); fig.savefig(os.path.join(OUT,out),dpi=150); plt.close(fig)
    from PIL import Image; Image.open(os.path.join(OUT,out)).convert("L").save(os.path.join(OUT,"_LDP_GRAY.png")); print("wrote",out)

def figSens(out="sensitivity.png"):
    d=J("results_real/sensitivity.json")
    params=[("gamma",r"step $\gamma$",["0.02","0.05","0.1"]),
            ("N_s",r"window $N_s$",["150","300","600"]),
            ("m",r"margin $m$",["0.01","0.02","0.05"])]
    fig,ax=plt.subplots(1,3,figsize=(13,3.9))
    for j,(key,lab,vals) in enumerate(params):
        x=range(len(vals))
        re_=[d[key][v]["rate_err_P5"] for v in vals]; um=[d[key][v]["under_max_P5"] for v in vals]
        a=ax[j]; a.plot(x,re_,"-o",color="black",lw=1.7,markersize=7,markerfacecolor="white",label="rate error")
        a.plot(x,um,"--s",color="black",lw=1.7,markersize=7,markerfacecolor="0.6",label="worst-case under-prot.")
        a.set_xticks(list(x)); a.set_xticklabels(vals); a.set_xlabel(lab); a.grid(alpha=0.3); a.set_ylim(0,0.065)
        if j==0: a.set_ylabel("value"); a.legend(fontsize=8)
    fig.suptitle("Hyperparameter sensitivity (P5)"); fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(os.path.join(OUT,out),dpi=150); plt.close(fig)
    from PIL import Image; Image.open(os.path.join(OUT,out)).convert("L").save(os.path.join(OUT,"_SENS_GRAY.png")); print("wrote",out)

figLDP(); figSens()
