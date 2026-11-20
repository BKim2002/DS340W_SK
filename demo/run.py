#!/usr/bin/env python3
"""CPU evaluation of the retained T5/OpenDelta models on the course Test split."""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import urllib.error
import urllib.request

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent


def digest(path: Path) -> str:
    value = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def verified_asset(record: dict) -> Path:
    path = (DEMO / record['path']).resolve(strict=True)
    if not path.is_relative_to(DEMO) or path.stat().st_size != record['size_bytes'] or digest(path) != record['sha256']:
        raise ValueError(f"Demo file failed its checksum: {record['path']}. Download a fresh repository copy.")
    return path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def ordered_test_rows(data: list[dict], reference: list[dict]) -> list[dict]:
    if len(data) != 436 or len(reference) != 436:
        raise ValueError('The full demo requires exactly 436 course Test examples')
    by_id = {f"validation:{row['idx']}": row for row in data}
    ids = [row['test_id'] for row in reference]
    if len(by_id) != 436 or len(set(ids)) != 436 or set(ids) != set(by_id):
        raise ValueError('The reference and course Test examples do not align')
    rows = [by_id[key] for key in ids]
    if any(row['label'] not in (0,1) or row['label'] != ref['label'] for row,ref in zip(rows,reference)):
        raise ValueError('Course Test labels differ from the retained record')
    return rows


def ensure_backbone(manifest: dict, cache_root: Path, offline: bool) -> Path:
    pin = manifest['backbone']
    destination = cache_root / pin['revision']
    destination.mkdir(parents=True, exist_ok=True)
    for record in pin['files']:
        name = record['name']
        if Path(name).name != name:
            raise ValueError('Backbone file names must be plain file names')
        target = destination / name
        if target.exists() and target.stat().st_size == record['size_bytes'] and digest(target) == record['sha256']:
            continue
        if offline:
            raise FileNotFoundError(f'Missing or corrupt cached file: {target}. Run once without --offline.')
        url = f"https://huggingface.co/{pin['repo_id']}/resolve/{pin['revision']}/{name}"
        partial = target.with_name(name+'.partial')
        for attempt in range(3):
            try:
                print(f'Downloading {name} ({record["size_bytes"]/1_000_000:.1f} MB)', flush=True)
                request = urllib.request.Request(url, headers={'User-Agent':'DS340W-CPU-Demo/1.0'})
                with urllib.request.urlopen(request, timeout=90) as response, partial.open('wb') as out:
                    total, last = 0, time.monotonic()
                    while block := response.read(1024*1024):
                        out.write(block)
                        total += len(block)
                        if time.monotonic()-last >= 5:
                            print(f'  {total/1_000_000:.0f}/{record["size_bytes"]/1_000_000:.0f} MB',flush=True)
                            last=time.monotonic()
                if partial.stat().st_size != record['size_bytes'] or digest(partial) != record['sha256']:
                    raise ValueError(f'Download checksum mismatch: {name}')
                partial.replace(target)
                break
            except (OSError, ValueError) as error:
                if attempt == 2:
                    raise RuntimeError(f'Could not download verified {name}. Check your Internet connection and retry.') from error
                time.sleep(2)
    return destination


def load_adapter(model, state: dict, specification: dict, low_rank_class) -> int:
    """Attach the original LowRankLinear block and restore every learned tensor."""
    import torch
    modules = dict(model.named_modules())
    prefixes = sorted(key.removesuffix('.lora.lora_A') for key in state if key.endswith('.lora.lora_A'))
    if len(prefixes) != 72:
        raise ValueError('The retained adapter must contain all 72 attention projections')
    for name in prefixes:
        base = modules[name]
        if not isinstance(base, torch.nn.Linear):
            raise ValueError(f'Expected a linear attention projection: {name}')
        layer = low_rank_class(base.in_features, base.out_features, base.weight,
                               r=specification['rank'], lora_alpha=specification['alpha'],
                               lora_dropout=0.0, lora_scaling_mode=specification['scaling_mode'])
        base.add_module('lora', layer)
        # OpenDelta inserts the same parallel correction after the base projection.
        def add_correction(module, inputs, output):
            return output + module.lora(inputs[0])
        base.register_forward_hook(add_correction)
    parameters = dict(model.named_parameters())
    expected = {key for key in parameters if '.lora.' in key or 'layer_norm' in key}
    if set(state) != expected:
        raise ValueError('Saved LoRA and LayerNorm tensor names do not match this backbone')
    with torch.no_grad():
        for name,value in state.items():
            if parameters[name].shape != value.shape:
                raise ValueError(f'Wrong tensor shape: {name}')
            parameters[name].copy_(value)
    model.requires_grad_(False)
    return sum(value.numel() for value in state.values())


def evaluate(key: str, specification: dict, manifest: dict, backbone: Path, args, low_rank_class) -> dict:
    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    checkpoint = verified_asset(specification['checkpoint'])
    reference = read_rows(verified_asset(specification['predictions']))
    rows = ordered_test_rows(read_rows(verified_asset(manifest['test'])),reference)
    print(f'\nLoading {key}: rank {specification["rank"]}, seed {specification["seed"]}, {specification["scaling_mode"]}',flush=True)
    tokenizer = AutoTokenizer.from_pretrained(backbone, local_files_only=True)
    config = transformers.AutoConfig.from_pretrained(backbone, local_files_only=True)
    config.dropout_rate=0.0
    model = AutoModelForSeq2SeqLM.from_pretrained(backbone, config=config, local_files_only=True,
                                                attn_implementation='eager',torch_dtype=torch.float32)
    model.resize_token_embeddings(len(tokenizer))
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    parameter_count = load_adapter(model,state,specification,low_rank_class)
    model.eval()
    predictions=[]
    started=time.perf_counter()
    for start in range(0,len(rows),args.batch_size):
        batch=rows[start:start+args.batch_size]
        texts=[f"sst2 sentence: {row['sentence']}" for row in batch]
        inputs=tokenizer(texts,max_length=128,padding=True,truncation=True,return_tensors='pt')
        with torch.inference_mode():
            generated=model.generate(**inputs,max_length=128,num_beams=1,do_sample=False)
        decoded=[text.strip() for text in tokenizer.batch_decode(generated,skip_special_tokens=True)]
        if any(label not in ('0','1') for label in decoded):
            raise ValueError(f'{key} generated a non-binary sentiment label')
        predictions.extend(int(label) for label in decoded)
        elapsed=time.perf_counter()-started
        print(f'[{key}] {len(predictions)}/436 predictions, {elapsed:.1f} seconds',flush=True)
    correct=sum(prediction==row['label'] for prediction,row in zip(predictions,rows))
    matches=all(prediction==ref['prediction'] for prediction,ref in zip(predictions,reference))
    records=[{'test_id':ref['test_id'],'label':row['label'],'prediction':prediction}
             for ref,row,prediction in zip(reference,rows,predictions)]
    metrics={'model':key,'experiment_id':specification['experiment_id'],'correct_count':correct,'test_rows':436,
             'accuracy_percent':100*correct/436,'checkpoint_step':specification['checkpoint_step'],
             'all_predictions_match_retained_record':matches,'inference_seconds':elapsed,
             'checkpoint_sha256':specification['checkpoint']['sha256'],'backbone_revision':manifest['backbone']['revision'],
             'loaded_trainable_parameters':parameter_count,'training_performed':False,'final_unseen_accessed':False,
             'environment':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
                            'platform':platform.platform(),'device':'cpu','threads':args.threads,'batch_size':args.batch_size}}
    out=args.output_dir/key
    out.mkdir(parents=True,exist_ok=True)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    print(f'{key}: {correct}/436 correct ({100*correct/436:.2f}%)',flush=True)
    if not matches or correct != specification['expected_correct']:
        raise RuntimeError(f'{key}: live predictions differ from the retained record. Actual results were saved to {out}.')
    print('VERIFIED: all 436 live CPU predictions match the retained training result.',flush=True)
    del model,state
    gc.collect()
    return metrics


def main() -> int:
    manifest=json.loads((DEMO/'manifest.json').read_text(encoding='utf-8'))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['both','all',*manifest['models']],default='both')
    parser.add_argument('--cache-dir',type=Path,default=Path.home()/'.cache/ds340w-demo')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/portable-demo')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--threads',type=int,default=min(4,os.cpu_count() or 1))
    parser.add_argument('--offline',action='store_true',help='Use only previously downloaded and verified backbone files')
    parser.add_argument('--download-only',action='store_true',help='Download and verify the pinned backbone without evaluating')
    args=parser.parse_args()
    if sys.version_info[:2] != (3,12) or sys.maxsize <= 2**32:
        parser.error('Use 64-bit Python 3.12; the pinned demo dependencies are separate from the legacy training environment.')
    if not 1 <= args.batch_size <= 32 or not 1 <= args.threads <= 32:
        parser.error('Batch size and threads must be between 1 and 32')
    if manifest['test_rows'] != 436 or manifest['final_unseen_included'] is not False:
        raise ValueError('This demo requires the public course Test partition only')
    verified_asset(manifest['test'])
    class_path=verified_asset(manifest['low_rank_linear'])
    backbone=ensure_backbone(manifest,args.cache_dir.resolve(),args.offline)
    if args.download_only:
        print(f'Backbone downloaded and verified: {backbone}')
        return 0
    import torch
    import transformers
    if torch.__version__.split('+')[0] != '2.5.1' or transformers.__version__ != '4.46.3':
        raise RuntimeError('Install the versions in requirements-demo.txt in a fresh virtual environment.')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    transformers.logging.set_verbosity_error()
    spec=importlib.util.spec_from_file_location('ds340w_portable_low_rank_linear',class_path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    keys=['baseline','modified'] if args.model=='both' else list(manifest['models']) if args.model=='all' else [args.model]
    print(f'CPU evaluation only. Threads: {args.threads}. No training. Final Unseen is excluded.',flush=True)
    results = [evaluate(key,manifest['models'][key],manifest,backbone,args,module.LowRankLinear) for key in keys]
    print('\nLIVE CPU RESULTS - course Test, seed 42',flush=True)
    print(f'{"Model":<16} {"Correct":>9} {"Accuracy":>10} {"Matches saved predictions":>26}',flush=True)
    for result in results:
        print(f'{result["model"]:<16} {result["correct_count"]:>5}/436 {result["accuracy_percent"]:>9.2f}% {"YES":>26}',flush=True)
    print(f'New predictions and metrics: {args.output_dir.resolve()}',flush=True)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (OSError,ValueError,RuntimeError,ImportError) as error:
        print(f'ERROR: {error}',file=sys.stderr)
        raise SystemExit(1)
