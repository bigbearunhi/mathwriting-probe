"""Throwaway MathWriting feasibility baseline, not a production recognizer."""
import argparse
import json
import math
import random
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

PAD, BOS, EOS, UNK = range(4)
NS = '{http://www.w3.org/2003/InkML}'
# Same token boundaries as the official example notebook.
COMMAND = re.compile(r'\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)')


def tokenize(s):
    out = []
    while s:
        token = COMMAND.match(s).group(0) if s[0] == '\\' else s[0]
        out.append(token)
        s = s[len(token):]
    return out


def read_ink(path):
    root = ET.parse(path).getroot()
    meta = {e.attrib['type']: e.text for e in root.findall(NS + 'annotation')}
    strokes = [[[float(v) for v in point.split()] for point in e.text.split(',')]
               for e in root.findall(NS + 'trace')]
    assert strokes and all(len(p) == 3 for s in strokes for p in s)
    return dict(id=path.stem, label=meta['normalizedLabel'], strokes=strokes,
                split=meta['splitTagOriginal'])


def features(sample, max_points=384):
    """Normalize isotropically; downsample each stroke preserving endpoints.

    Feature 3 is pen_up_after_this_point (1 at every stroke's last point).
    Raw time remains in exported data, but is not used in this baseline.
    """
    strokes = [torch.tensor(s, dtype=torch.float32)[:, :2] for s in sample['strokes']]
    lengths = [len(s) for s in strokes]
    counts = [min(n, 2) for n in lengths]
    if sum(counts) > max_points:
        raise ValueError('Point budget cannot preserve every stroke endpoint')
    while sum(counts) < min(max_points, sum(lengths)):
        i = max(range(len(strokes)), key=lambda j: (lengths[j] - counts[j]) / lengths[j])
        counts[i] += 1
    all_xy = torch.cat(strokes)
    low, high = all_xy.amin(0), all_xy.amax(0)
    scale = (high - low).max().clamp_min(1e-6)
    out = []
    for stroke, n in zip(strokes, counts):
        idx = torch.linspace(0, len(stroke)-1, n).round().long()
        xy = (stroke[idx] - (low + high)/2) / scale
        up = torch.zeros(n, 1)
        up[-1] = 1
        out.append(torch.cat([xy, up], 1))
    result = torch.cat(out)
    assert torch.isfinite(result).all()
    return result


class Model(nn.Module):
    def __init__(self, vocab_size, dim=128, layers=2, heads=4):
        super().__init__()
        self.dim = dim
        self.input = nn.Linear(3, dim)
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.transformer = nn.Transformer(
            d_model=dim, nhead=heads, num_encoder_layers=layers,
            num_decoder_layers=layers, dim_feedforward=dim*4,
            dropout=0.0, batch_first=True)
        self.output = nn.Linear(dim, vocab_size)

    def position(self, x):
        pos = torch.arange(x.size(1), device=x.device, dtype=x.dtype)[:, None]
        freq = torch.exp(torch.arange(0, self.dim, 2, device=x.device,
                                     dtype=x.dtype) * (-math.log(10000)/self.dim))
        pe = torch.zeros(x.size(1), self.dim, device=x.device, dtype=x.dtype)
        pe[:, 0::2], pe[:, 1::2] = (pos*freq).sin(), (pos*freq).cos()
        return x + pe

    def encode(self, src, padding):
        return self.transformer.encoder(self.position(self.input(src)),
                                        src_key_padding_mask=padding)

    def decode(self, memory, target, padding):
        causal = torch.ones(target.size(1), target.size(1),
                            device=target.device, dtype=torch.bool).triu(1)
        y = self.transformer.decoder(
            self.position(self.embedding(target)*math.sqrt(self.dim)), memory,
            tgt_mask=causal, tgt_key_padding_mask=target.eq(PAD),
            memory_key_padding_mask=padding)
        return self.output(y)

    def forward(self, src, target, padding):
        return self.decode(self.encode(src, padding), target, padding)


def batch(samples, device):
    src = pad_sequence([s['features'] for s in samples], batch_first=True)
    lengths = torch.tensor([len(s['features']) for s in samples])
    padding = torch.arange(src.size(1))[None, :] >= lengths[:, None]
    target = pad_sequence([s['target'] for s in samples], batch_first=True, padding_value=PAD)
    return src.to(device), target.to(device), padding.to(device)


@torch.no_grad()
def metrics(model, samples, device, batch_size):
    model.eval()
    loss, correct, count = 0.0, 0, 0
    for start in range(0, len(samples), batch_size):
        x, target, padding = batch(samples[start:start+batch_size], device)
        pred = model(x, target[:, :-1], padding)
        gold = target[:, 1:]
        loss += nn.functional.cross_entropy(pred.reshape(-1, pred.size(-1)), gold.reshape(-1),
                                             ignore_index=PAD, reduction='sum').item()
        correct += ((pred.argmax(-1) == gold) & gold.ne(PAD)).sum().item()
        count += gold.ne(PAD).sum().item()
    return dict(loss=loss/count, teacher_forced_token_accuracy=correct/count)


@torch.no_grad()
def generate(model, sample, device, vocab, limit=128):
    model.eval()
    x, _, padding = batch([sample], device)
    memory = model.encode(x, padding)
    tokens = torch.tensor([[BOS]], device=device)
    ended = False
    for _ in range(limit):
        logits = model.decode(memory, tokens, padding)[0, -1]
        logits[[PAD, BOS]] = -float('inf')
        nxt = logits.argmax().reshape(1, 1)
        tokens = torch.cat([tokens, nxt], 1)
        if nxt.item() == EOS:
            ended = True
            break
    ids = tokens[0, 1:].tolist()
    prediction = ''.join(vocab[i] for i in ids if i != EOS)
    return dict(id=sample['id'], truth=sample['label'], prediction=prediction,
                ended=ended, exact_match=ended and prediction == sample['label'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('data/mathwriting-2024-excerpt'))
    ap.add_argument('--out', type=Path, default=Path('runs/smoke'))
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--train-limit', type=int, default=16,
                    help='Shortest training expressions for deliberate overfit; 0 uses all')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-points', type=int, default=384)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Disable nested tensor fast-path so validation follows the training math path.
    torch.backends.mha.set_fastpath_enabled(False)
    raw = {split: [read_ink(p) for p in sorted((args.data/split).glob('*.inkml'))]
           for split in ('train', 'valid')}
    assert raw['train'] and raw['valid']
    assert not ({s['id'] for s in raw['train']} & {s['id'] for s in raw['valid']})
    vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>'] + sorted(
        {t for s in raw['train'] for t in tokenize(s['label'])})
    lookup = {t: i for i, t in enumerate(vocab)}
    (args.out/'vocab.json').write_text(json.dumps(vocab, ensure_ascii=False, indent=2))
    data, unknown = {}, {}
    for split, samples in raw.items():
        with (args.out/f'{split}.jsonl').open('w') as f:
            for s in samples:
                points = [dict(x=p[0], y=p[1], time_ms=p[2], pen_up=int(i == len(stroke)-1))
                          for stroke in s['strokes'] for i, p in enumerate(stroke)]
                f.write(json.dumps(dict(id=s['id'], latex=s['label'], points=points))+'\n')
        data[split] = []
        n_unk = n_token = 0
        for s in samples:
            ids = [lookup.get(t, UNK) for t in tokenize(s['label'])]
            n_unk += ids.count(UNK)
            n_token += len(ids)
            data[split].append({**s, 'features': features(s, args.max_points),
                                'target': torch.tensor([BOS]+ids+[EOS])})
        unknown[split] = dict(unknown_tokens=n_unk, total_tokens=n_token)
    train = sorted(data['train'], key=lambda s: len(s['target']))
    if args.train_limit:
        train = train[:args.train_limit]
    model = Model(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    initial = metrics(model, train, device, args.batch_size)
    print(json.dumps(dict(device=device, parameters=sum(p.numel() for p in model.parameters()),
                          train_samples=len(train), valid_samples=len(data['valid']),
                          vocab_size=len(vocab), initial=initial)), flush=True)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history = []
    with (args.out/'training.jsonl').open('w') as log:
        for step in range(1, args.steps+1):
            model.train()
            samples = random.sample(train, min(args.batch_size, len(train)))
            x, target, padding = batch(samples, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, target[:, :-1], padding)
            loss = nn.functional.cross_entropy(pred.reshape(-1, len(vocab)),
                                                target[:, 1:].reshape(-1), ignore_index=PAD)
            assert torch.isfinite(loss), 'Non-finite loss'
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert torch.isfinite(norm), 'Non-finite gradient'
            optimizer.step()
            if step == 1 or step % 50 == 0 or step == args.steps:
                row = dict(step=step, loss=loss.item(), elapsed_s=time.perf_counter()-start)
                history.append(row)
                log.write(json.dumps(row)+'\n')
                log.flush()
                print(json.dumps(row), flush=True)
    train_seconds = time.perf_counter()-start
    report = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  device=device, torch_version=torch.__version__, seed=42,
                  parameters=sum(p.numel() for p in model.parameters()),
                  train_ids=[s['id'] for s in train], unknown_tokens=unknown,
                  initial_train=initial, final_train=metrics(model, train, device, args.batch_size),
                  validation=metrics(model, data['valid'], device, args.batch_size),
                  train_seconds=train_seconds, history=history,
                  peak_allocated_gpu_mb=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else 0)
    report['train_predictions'] = [generate(model, s, device, vocab) for s in train[:16]]
    report['validation_predictions'] = [generate(model, s, device, vocab) for s in data['valid'][:8]]
    report['warning'] = 'Tiny-data feasibility/overfit probe; not a benchmark or production accuracy.'
    torch.save(dict(model=model.state_dict(), vocab=vocab, dim=128, layers=2, heads=4,
                    max_points=args.max_points), args.out/'checkpoint.pt')
    # Verify checkpoint round trip, not just that a file was written.
    restored = Model(len(vocab)).to(device)
    restored.load_state_dict(torch.load(args.out/'checkpoint.pt', map_location=device,
                                       weights_only=True)['model'])
    assert generate(restored, train[0], device, vocab) == report['train_predictions'][0]
    report['checkpoint_reload_verified'] = True
    (args.out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ('initial_train', 'final_train', 'validation',
                                          'train_seconds', 'peak_allocated_gpu_mb') }), flush=True)


if __name__ == '__main__':
    main()
