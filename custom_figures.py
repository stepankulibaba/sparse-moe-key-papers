"""Static research figures for the metric comparison and counterexamples."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from custom_metrics import influence
from figures import label

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    rows=json.loads((ROOT/'data/custom_rankings.json').read_text(encoding='utf-8'))
    chosen=sorted(rows,key=lambda r:r['rank_branching_descendants'])[:12]
    fields=['rank_local_citations','rank_pagerank','rank_katz_paths','rank_unique_descendants','rank_branching_descendants']
    arr=np.array([[r[f] for f in fields] for r in chosen])
    fig,ax=plt.subplots(figsize=(10,5.3),layout='constrained')
    ax.imshow(arr,cmap='Blues_r',vmin=1,vmax=max(20,int(arr.max())),aspect='auto')
    labels=[label(r) for r in chosen]
    labels=[s if not s.startswith('Taming') else 'Stochastic Experts' for s in labels]
    ax.set_yticks(range(len(chosen)),labels)
    ax.set_xticks(range(5),['Цитирования','PageRank','Katz','Охват U','Ветви B'])
    ax.tick_params(length=0,pad=9)
    for i in range(len(chosen)):
        for j in range(5):
            ax.text(j,i,str(arr[i,j]),ha='center',va='center',color='white' if arr[i,j]<8 else '#202020')
    ax.set_title('Места в рейтингах: собственные метрики меняют порядок работ',loc='left',fontsize=13,pad=15)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.savefig(OUT/'custom_rank_comparison.png',dpi=200,bbox_inches='tight');plt.close(fig)
    specs=[('Цепочка',[('c','b'),('b','a'),('a','v')],{'v':(0,0),'a':(0,1),'b':(0,2),'c':(0,3)}),
           ('Звезда',[('a','v'),('b','v'),('c','v')],{'v':(0,0),'a':(-1,2),'b':(0,2.6),'c':(1,2)}),
           ('Ромб',[('a','v'),('b','v'),('c','a'),('c','b')],{'v':(0,0),'a':(-.8,1.4),'b':(.8,1.4),'c':(0,3)})]
    fig,axes=plt.subplots(1,3,figsize=(10,4.6),layout='constrained')
    for ax,(title,edges,positions) in zip(axes,specs):
        g=nx.DiGraph(edges);s=influence(g)['v']
        nx.draw_networkx(g,positions,ax=ax,node_color=['#dfaa38' if u=='v' else '#376da4' for u in g],
                         node_size=600,font_color='white',font_weight='bold',edgecolors='#253341',arrowsize=18,width=1.2)
        ax.set_title(title,fontsize=14)
        ax.set_ylim(-1.1,3.5);ax.set_xlim(-1.7,1.7)
        ax.text(0,-.7,f"c={s['direct_children']}   K={s['katz_paths']:.2f}\nU={s['unique_descendants']:.2f}   B={s['branching_descendants']:.2f}",ha='center',va='center',fontsize=12)
        ax.set_axis_off()
    fig.suptitle('Контрпримеры на синтетических графах, β=0,5; оценивается вершина v',fontsize=13)
    fig.savefig(OUT/'metric_counterexamples.png',dpi=200,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
