"""Render a completed A/E run to a standalone UTF-8 HTML report."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
from html import escape
import json
from pathlib import Path


def table(path, columns=None):
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        names, rows = reader.fieldnames or [], list(reader)
    if columns is not None:
        names = columns
    def cell(value):
        try:
            n = float(value)
            if '.' in value or 'e' in value.lower(): return f'{n:.4f}'
        except (ValueError, TypeError): pass
        return escape(str(value))
    return ('<table><thead><tr>' + ''.join('<th>'+escape(n)+'</th>' for n in names) + '</tr></thead><tbody>'
            + ''.join('<tr>'+''.join('<td>'+cell(row.get(n,''))+'</td>' for n in names)+'</tr>' for row in rows)
            + '</tbody></table>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args=parser.parse_args()
    run=args.run_dir.resolve()
    if not (run/'completed.json').is_file():
        raise ValueError('Run is not complete. Refusing to present partial numbers as final results.')
    protocol=json.loads((run/'experiment_protocol.json').read_text(encoding='utf-8'))
    with (run/'comparison.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    with (run/'paired_bootstrap.csv').open(encoding='utf-8') as f: paired=list(csv.DictReader(f))
    with (run/'per_class_metrics.csv').open(encoding='utf-8') as f: classes=list(csv.DictReader(f))
    findings=[]
    for row in paired:
        delta,lo,hi=map(float,(row['delta_macro_f1'],row['ci_low'],row['ci_high']))
        inference='区间跨过或触及零，本次测试重采样未显示明确差异。' if lo<=0<=hi else '区间未跨零；该区间仅衡量此 seed 下的测试样本不确定性。'
        findings.append(f"seed {escape(row['seed'])}：E−A 的 macro-F1 差值为 {delta:+.4f}，95% 配对 bootstrap 区间 [{lo:.4f}, {hi:.4f}]。{inference}")
    class_findings=[]
    for seed in protocol['seeds']:
        lookup={(r['arm'],r['class']):float(r['f1-score']) for r in classes if int(r['seed'])==seed}
        if set(protocol['arms'])=={'A','E'}:
            changes={label:lookup[('E',label)]-lookup[('A',label)] for label in ['Negative','Positive','Irrelevant','Neutral']}
            class_findings.append(f"seed {seed} 的各类 F1 差值（E−A）："+'；'.join(f'{label} {delta:+.4f}' for label,delta in changes.items())+'。请结合类别召回率判断整体均值是否掩盖了类别间的取舍。')
    environment=json.loads((run/'environment.json').read_text(encoding='utf-8'))
    bars=''
    for i,row in enumerate(rows):
        score=float(row['test_macro_f1'])
        bars+=f'<text x="0" y="{32+i*45}">{escape(row["arm"])} / seed {escape(row["seed"])}</text><rect x="140" y="{14+i*45}" width="{score*450:.2f}" height="25" fill="{("#2774ae" if row["arm"]=="A" else "#2a9d8f")}"/><text x="{150+score*450:.2f}" y="{32+i*45}">{score:.4f}</text>'
    methods='''A 使用冻结 Qwen3-0.6B 的分类提示末端隐藏状态；E 使用冻结 Qwen3-Embedding-0.6B 的 Instruct/Query 输入与末端隐藏状态。两组均取 1024 维向量并 L2 归一化；StandardScaler 仅拟合训练集。独立 MLP 为 1024→128 ReLU→4，Adam，学习率 0.001，alpha=0.01，batch 32，最多 200 epoch，验证 macro-F1 连续 20 epoch 不改善则停止并恢复最佳轮次。模型全程冻结，无 LoRA 训练、教师调用或合成数据。'''
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Kai A/E 实验报告</title>
<style>body{{font-family:"Microsoft YaHei",Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#203044;line-height:1.7}}h1,h2{{color:#16324f}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}}th,td{{padding:7px;border:1px solid #d4dde5;text-align:left}}th{{background:#edf3f7}}.scroll{{overflow:auto}}pre{{white-space:pre-wrap;font-size:12px}}svg{{max-width:100%}}@media print{{body{{margin:0}}table{{font-size:10px}}}}</style>
<h1>Kai：冻结 Qwen 文本表示＋独立 MLP</h1>
<p>运行：{escape(run.name)} · 报告生成：{datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')}（北京时间） · 实验状态：已完成</p>
<h2>研究问题与方法</h2><p>在相同训练、验证与测试划分上，通用生成模型与专用嵌入模型的冻结表示，分别能支持怎样的情感分类效果？</p><p>{methods}</p>
<p>数据：473 train / 118 validation / 397 test；划分 SHA-256：<code>{escape(protocol['split_hash'])}</code>。所有分类头先在验证集选型，之后读取测试标签。提取测试向量只读取文本，不参与训练或标准化拟合。</p>
<h2>测试结果</h2><svg viewBox="0 0 720 {55+45*len(rows)}" role="img" aria-label="各组测试 macro-F1，横轴零到一">{bars}</svg><div class="scroll">{table(run/'comparison.csv', ['arm','seed','validation_macro_f1','best_epoch','test_macro_f1','test_accuracy'])}</div>
<h2>汇总与配对比较</h2><div class="scroll">{table(run/'comparison_aggregate.csv')}{table(run/'paired_bootstrap.csv')}</div><p>{'<br>'.join(findings) or '本次未同时运行 A、E，无法计算配对差值。'}</p>
<h2>各类别结果</h2><p>{'<br>'.join(class_findings)}</p><div class="scroll">{table(run/'per_class_metrics.csv')}</div>
<h2>解读边界与后续</h2><p>两组的预训练目标和输入模板不同，属于现成表示系统的比较，不能把差异归因于某一个训练环节。两组使用同一种 MLP，不是两种不同分析模型。单 seed 结果不代表跨种子稳定性；当前尚未评估解释蒸馏或反事实增强，也不构成其有效性证据。测试集已有历史实验记录，不应将其视为全新未见测试集，或据此反复调参。</p>
<p>后续可按预先确定的配置运行多个 seed，观察四类 F1 与混淆矩阵，再设计原始标签适配 B 组及教师辅助模块。本报告不以其他协议的历史基线分数直接判定胜负。</p>
<h2>运行环境与可复现信息</h2><pre>{escape(json.dumps(environment,ensure_ascii=False,indent=2))}</pre><pre>{escape(json.dumps(protocol,ensure_ascii=False,indent=2))}</pre>
<p>模型资料：<a href="https://huggingface.co/Qwen/Qwen3-0.6B">Qwen3-0.6B</a>；<a href="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B">Qwen3-Embedding-0.6B</a>。</p></html>'''
    args.output_dir.mkdir(parents=True,exist_ok=True)
    target=args.output_dir/f'Kai_AE_Report_{run.name}.html'
    target.write_text(body,encoding='utf-8')
    print(target)


if __name__=='__main__': main()
