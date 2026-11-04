"""Static figures for the assignment's PDF report, with frozen source values."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
NAMES = {
    'W2581624817':'Sparse MoE (2017)', 'W3040573126':'GShard', 'W3119866685':'Switch',
    'W4402671950':'DeepSeekMoE', 'W4200634402':'GLaM', 'W4390723197':'Mixtral',
    'W7133198209':'Expert Choice', 'W4405903187':'DeepSeek-V3', 'W4396815229':'DeepSeek-V2',
    'W4225390052':'ST-MoE', 'W7133187299':'VLMo', 'W4408858386':'Cai et al. survey',
}

def label(p):
    return NAMES.get(p['id'].split('/')[-1], p['title'].split(':')[0][:33])

def main():
    papers = json.loads((ROOT/'data/papers.json').read_text(encoding='utf-8'))
    summary = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    by_id = {p['id']:p for p in papers}
    graph = nx.DiGraph()
    graph.add_nodes_from(by_id)
    with (ROOT/'data/edges.csv').open(encoding='utf-8-sig',newline='') as f:
        graph.add_edges_from((e['citing'],e['cited']) for e in csv.DictReader(f))
    top = sorted(papers,key=lambda p:p['rank_pagerank'])[:35]
    sub = graph.subgraph([p['id'] for p in top]).copy()
    connected = max(nx.weakly_connected_components(sub),key=len)
    sub = sub.subgraph(connected).copy()
    highlights = summary['algorithmic_highlights']
    decisions=json.loads((ROOT/'author_decisions.json').read_text(encoding='utf-8'))
    final_choice=bool(decisions.get('final_papers'))
    if final_choice:highlights=[p['id'] for p in decisions['final_papers']]
    positions = {}
    ordered_highlights = sorted(highlights,key=lambda u:(by_id[u]['date'],u))
    for i,u in enumerate(ordered_highlights):positions[u]=np.array([-1+.5*i,-.65])
    ordinary = sorted((u for u in sub if u not in highlights),key=lambda u:(by_id[u]['date'],u))
    for i,u in enumerate(ordinary):
        row,col=divmod(i,10)
        count=min(10,len(ordinary)-row*10)
        positions[u]=np.array([-.98+1.96*col/max(1,count-1),.12+.4*row])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,5.4),layout='constrained')
    nx.draw_networkx_edges(sub,positions,ax=ax,arrows=True,arrowsize=9,width=.6,alpha=.25,edge_color='#626b75',node_size=150,connectionstyle='arc3,rad=.05')
    nx.draw_networkx_nodes(sub,positions,nodelist=ordinary,node_size=75,node_color='#376da4',edgecolors='white',linewidths=.5,ax=ax)
    nx.draw_networkx_nodes(sub,positions,nodelist=highlights,node_size=210,node_color='#dfaa38',node_shape='s',edgecolors='#242424',linewidths=.8,ax=ax)
    for u in highlights:
        x,y=positions[u]
        ax.annotate(label(by_id[u]),(x,y),xytext=(0,-22),textcoords='offset points',ha='center',va='top',fontsize=12,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.87,'pad':1})
    ax.set_title(f"Срез лидеров PageRank: {sub.number_of_nodes()} вершины, {sub.number_of_edges()} связи",loc='left',fontsize=14,pad=16)
    ax.text(0,-.04,'Стрелка: цитирующая статья → цитируемая. Позиции условны.\n'+('Квадраты: итоговый список по показателю B.' if final_choice else 'Квадраты: алгоритмический top-5; выбор студента ещё не сделан.'),transform=ax.transAxes,fontsize=10)
    ax.set_axis_off()
    ax.margins(.18)
    ax.set_ylim(-1.15,1.2)
    fig.savefig(OUT/'citation_graph.png',dpi=200,bbox_inches='tight')
    fig.savefig(OUT/'citation_graph.svg',bbox_inches='tight')
    plt.close(fig)
    # Exact rank values; positions are ordinal with stable ID tie-breaking.
    shown = sorted(papers,key=lambda p:p['rank_pagerank'])[:12]
    arr=np.array([[p['rank_local_citations'],p['rank_pagerank'],p['rank_global_citations']] for p in shown])
    fig,ax=plt.subplots(figsize=(9,5.5),layout='constrained')
    ax.imshow(arr,cmap='Blues_r',vmin=1,vmax=max(20,int(arr.max())),aspect='auto')
    ax.set_yticks(range(len(shown)),[label(p) for p in shown])
    ax.set_xticks([0,1,2],['Цитирования в корпусе','PageRank','Цитирования OpenAlex'])
    ax.tick_params(length=0,pad=9)
    for i in range(len(shown)):
        for j in range(3):
            ax.text(j,i,str(arr[i,j]),ha='center',va='center',color='white' if arr[i,j]<arr.max()*.42 else '#202020',fontsize=11)
    ax.set_title('Места в рейтингах: меньшее число означает более высокое место',loc='left',fontsize=12,pad=14)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.savefig(OUT/'rank_comparison.png',dpi=200,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
