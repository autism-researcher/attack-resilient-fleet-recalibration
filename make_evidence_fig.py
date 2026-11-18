import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
plt.rcParams.update({"font.size":10.5})
fig,ax=plt.subplots(figsize=(12.5,8.8)); ax.set_xlim(0,12.5); ax.set_ylim(0,9.4); ax.axis("off")

def box(cx,cy,w,h,text,serial=None,fc="white",fs=10.5,bold=False):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.02,rounding_size=0.10",
                 linewidth=1.3,edgecolor="black",facecolor=fc))
    ax.text(cx,cy,text,ha="center",va="center",fontsize=fs,fontweight="bold" if bold else "normal")
    if serial is not None:
        bx,by=cx-w/2,cy+h/2
        ax.add_patch(Circle((bx,by),0.23,facecolor="black",edgecolor="black",zorder=5))
        ax.text(bx,by,str(serial),ha="center",va="center",color="white",fontsize=10.5,fontweight="bold",zorder=6)

def arrow(x1,y1,x2,y2,lw=1.3):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=15,lw=lw,color="black",shrinkA=2,shrinkB=2))

# Tier 1: foundations
box(3.25,8.0,5.6,1.15,"Controlled NGSIM-range-informed generator\nknown ground-truth $B^\\star$  $\\Rightarrow$  measures\nTRUE worst-case under-protection",serial=1,fc="0.90")
box(9.25,8.0,5.6,1.15,"Real cross-dataset shift\nhighD / NGSIM / Waymo  ($B^\\star=0.31/0.38/0.85$)\nsevere, naturalistic distribution shift",serial=2,fc="0.90")
# Tier 2: study battery
y2=5.35
box(2.35,y2,4.0,1.7,"Accuracy & adaptation\n(Exp A, B, D)\n\nP5 recalibrates\n$\\approx$30% faster than\nlocal-only",serial=3)
box(6.25,y2,3.7,1.7,"Safety under corruption\n(Exp C, E, F, H, I)\n\nadv. $\\leq$50%, 5 Byzantine\naggregators, delay,\nloss, LDP noise",serial=4)
box(10.15,y2,4.0,1.7,"Closed-loop V2X\nmicrosimulation\n\ncollisions/1k under\nlatency $\\leq$1 s,\nloss $\\leq$50%",serial=5)
# Ablation callout
box(6.25,3.15,6.4,0.85,"Ablations — naive / clamp-free designs BREACH the bound (0.079 / 0.160 / 0.242)",serial=6,fc="0.80",fs=10)
# Tier 3: findings
box(3.1,1.0,5.5,1.25,"ACCURACY CHANNEL\ncollaboration accelerates recalibration;\nbenefit is penetration-independent ($\\geq$10%)",serial=7,fc="0.92")
box(9.4,1.0,5.5,1.25,"SAFETY CHANNEL\nper-vehicle under-protection within the\ncertified bound; fleet-reading designs 2--3x larger",serial=8,fc="0.92")

# arrows
for xs in (3.25,9.25):
    for xt in (2.35,6.25,10.15):
        arrow(xs,7.42,xt,6.22,lw=1.0)
arrow(2.35,4.5,3.1,1.63)
arrow(6.25,4.5,6.25,3.60)
arrow(10.15,4.5,9.4,1.63)
arrow(6.25,2.72,9.4,1.63)

ax.text(6.25,9.15,"Experimental evidence",ha="center",va="center",fontsize=13,fontweight="bold")
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),"figures_bw","experimental_evidence.png")
os.makedirs(os.path.dirname(out),exist_ok=True)
fig.savefig(out,dpi=150,bbox_inches="tight")
from PIL import Image
Image.open(out).convert("L").save(os.path.join(os.path.dirname(out),"_EVID_GRAY.png"))
print("regenerated")
