"""Export reusable, source-backed figures for the robustness appendix."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/robustness'
BLUE, GOLD, INK = '#235789', '#BF8D24', '#192B3C'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.labelcolor': INK,
                     'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
SHORT = {'W2581624817': 'Sparsely-Gated MoE', 'W3040573126': 'GShard', 'W3119866685': 'Switch',
         'W4200634402': 'GLaM', 'W4402671950': 'DeepSeekMoE', 'W7133198209': 'Expert Choice',
         'W4226515448': 'DeepSpeed-MoE', 'W4225390052': 'ST-MoE'}


def short(uid, title):
    return SHORT.get(uid.rsplit('/', 1)[-1], title.split(':')[0].replace('A Strong, Economical, and Efficient Mixture-of-Experts Language Model', '').strip())


def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=200, bbox_inches='tight')
    fig.savefig(OUT / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)


def main():
    with (OUT/'rankings.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    with (OUT/'stability.csv').open(encoding='utf-8-sig') as f:
        stability = list(csv.DictReader(f))
    summary = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    baseline = [r for r in rows if r['variant'] == 'baseline' and r['metric'] == 'pagerank']
    baseline.sort(key=lambda r: (float(r['midrank']), r['id']))
    selected = baseline[:8]
    lookup = {(r['variant'], r['metric'], r['id']): float(r['midrank']) for r in rows}
    cm = LinearSegmentedColormap.from_list('quiet_blue', ['#FFFFFF', '#DCE8F2', BLUE])
    labels = [short(r['id'], r['title']) for r in selected]
    metrics = ['local', 'pagerank', 'branching', 'fractional']
    data = np.array([[lookup['baseline', m, r['id']] for m in metrics] for r in selected])
    fig, ax = plt.subplots(figsize=(12, 5.7))
    ax.imshow(np.maximum(0, 11-data), cmap=cm, vmin=0, vmax=10, aspect='auto')
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(4), ['Цитирования c', 'PageRank', 'Ветвление B', 'Дробный F'])
    ax.xaxis.tick_top()
    for i in range(len(labels)):
        for j in range(4):
            ax.text(j, i, f'{data[i, j]:g}', ha='center', va='center', fontsize=15,
                    color='white' if data[i, j] <= 3 else INK)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(length=0, pad=12)
    fig.subplots_adjust(left=.26, right=.98, top=.86, bottom=.09)
    fig.text(.26, .02, '8 лидеров PageRank из 140 статей. Меньше = лучше. 5,5 = ничья на местах 5-6.', fontsize=11)
    save(fig, 'ranking_comparison')

    scenarios = ['edge_10', 'edge_30', 'edge_50', 'bibliography_30']
    freq = {(r['scenario'], r['metric'], r['id']): float(r['top5_frequency'])*100 for r in stability}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3), sharey=True)
    for ax, metric, label in zip(axes, ['pagerank', 'branching'], ['PageRank', 'Ветвление B']):
        values = np.array([[freq[s, metric, r['id']] for s in scenarios] for r in selected])
        ax.imshow(values, cmap=cm, vmin=0, vmax=100, aspect='auto')
        ax.set_title(label, fontsize=17, pad=15, weight='bold')
        ax.set_yticks(range(8), labels)
        ax.set_xticks(range(4), ['10%\nссылок', '30%\nссылок', '50%\nссылок', '30%\nбиблиогр.'])
        ax.tick_params(length=0, pad=10)
        for i in range(8):
            for j in range(4):
                ax.text(j, i, f'{values[i,j]:.1f}', ha='center', va='center',
                        color='white' if values[i,j] >= 65 else INK, fontsize=12)
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.subplots_adjust(left=.20, right=.99, top=.88, bottom=.17, wspace=.10)
    fig.text(.20, .015, 'Частота попадания в top-5, %. По 500 запусков на сценарий. Это частоты стресс-теста, не вероятность истинности.', fontsize=10)
    save(fig, 'stability_top5')

    variants = ['baseline', 'no_observed_shared_authors', 'openalex_only', 'no_review_outgoing_votes']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax, metric, label in zip(axes, ['pagerank', 'branching'], ['PageRank', 'Ветвление B']):
        values = np.array([[lookup[v, metric, r['id']] for v in variants] for r in selected[:6]])
        ax.imshow(np.maximum(0, 11-values), cmap=cm, vmin=0, vmax=10, aspect='auto')
        ax.set_title(label, fontsize=17, pad=15, weight='bold')
        ax.set_yticks(range(6), labels[:6])
        ax.set_xticks(range(4), ['Исходный', 'Без общих\nавторов*', 'Только\nOpenAlex', 'Без ссылок\nиз обзоров'])
        ax.tick_params(length=0, pad=10)
        for i in range(6):
            for j in range(4):
                ax.text(j, i, f'{values[i,j]:g}', ha='center', va='center', fontsize=14,
                        color='white' if values[i,j] <= 3 else INK)
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.subplots_adjust(left=.20, right=.99, top=.87, bottom=.20, wspace=.10)
    fig.text(.20, .025, '*Удалены только 42 связи с наблюдаемым совпадением ID авторов; сведения об авторах неполны.', fontsize=11)
    save(fig, 'ablation_comparison')

    c = summary['correlations'][1:]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    y = np.arange(3)
    for offset, key, color, hatch, label in [(-.17, 'vs_local_all140', BLUE, None, 'Все 140 статей'),
                                            (.17, 'vs_local_base_pagerank_top20', GOLD, '//', '20 лидеров исходного PageRank')]:
        values = [r[key] for r in c]
        ax.barh(y+offset, values, height=.28, color=color, hatch=hatch, label=label)
        for yy, v in zip(y+offset, values):
            ax.text(v-.025, yy, f'{v:.3f}', color='white' if color == BLUE else INK,
                    bbox={'facecolor': color, 'edgecolor': 'none', 'pad': 1.5},
                    ha='right', va='center', weight='bold', fontsize=12)
    ax.set_yticks(y, ['PageRank', 'Ветвление B', 'Дробный F'])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel('Корреляция Спирмена с числом локальных цитирований c')
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.03), ncol=2, frameon=False)
    for side in ['top', 'right']: ax.spines[side].set_visible(False)
    ax.spines['left'].set_color('#BBBBBB'); ax.spines['bottom'].set_color('#BBBBBB')
    ax.tick_params(axis='y', length=0, pad=10)
    fig.subplots_adjust(left=.18, right=.98, top=.80, bottom=.20)
    save(fig, 'correlation_scope')
    print('Exported four PNG/SVG figures to', OUT)


if __name__ == '__main__':
    main()
