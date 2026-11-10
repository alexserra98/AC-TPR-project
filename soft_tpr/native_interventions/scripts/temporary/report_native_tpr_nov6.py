"""Audit and report native TPR channel swaps and calibrated Soft TPR slot swaps."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ('raw_centroid', 'tpr_native_role_swap', 'soft_native_slot_swap')
LAYERS = (20, 21, 22)
LABELS = dict(zip(METHODS, ('Centroide raw', 'TPR: canali nativi', 'Soft TPR: slot nativi')))
COLORS = dict(zip(METHODS, ('#64748b', '#2563eb', '#d97706')))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def flag(value):
    if value in (True, 'True', 'true', '1'): return True
    if value in (False, 'False', 'false', '0'): return False
    raise ValueError(f'Invalid flag: {value!r}')


def num(value):
    return None if value in (None, '') else float(value)


def percent(value):
    return 'n/a' if value is None else f'{100*float(value):.2f}%'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |'] +
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def baseline_index(rows):
    indexed = {}
    for row in rows:
        key = int(row['row_id'])
        if key in indexed: raise ValueError('Duplicate baseline row')
        if not all(math.isfinite(float(row[k])) for k in ('agent_logit', 'patient_logit')):
            raise ValueError('Nonfinite baseline')
        indexed[key] = row
    return indexed


def require_same_baselines(rows, reference):
    current, old = baseline_index(rows), baseline_index(reference)
    if current.keys() != old.keys(): raise ValueError('Baseline coverage differs')
    for key, row in current.items():
        for field in ('split', 'voice', 'agent', 'patient', 'agent_logit', 'patient_logit'):
            a, b = row[field], old[key][field]
            if field.endswith('_logit'): a, b = float(a), float(b)
            if a != b: raise ValueError(f'Baseline differs: {key}/{field}')
    return current


def summarize(rows):
    n = len(rows)
    eligible = sum(float(r['baseline_margin']) < 0 for r in rows)
    flips = sum(float(r['baseline_margin']) < 0 and float(r['edited_margin']) > 0 for r in rows)
    return dict(sentences=n, baseline_agent_wins=eligible, flips=flips,
                conditional_flip=flips/eligible if eligible else None,
                patient_preference=sum(float(r['edited_margin']) > 0 for r in rows)/n,
                mean_margin_change=math.fsum(float(r['edited_margin'])-float(r['baseline_margin']) for r in rows)/n,
                mean_agent_edit_norm=math.fsum(float(r['agent_edit_norm']) for r in rows)/n,
                mean_patient_edit_norm=math.fsum(float(r['patient_edit_norm']) for r in rows)/n,
                zero_sentence_edits=sum(flag(r['agent_edit_is_zero']) and flag(r['patient_edit_is_zero']) for r in rows),
                zero_token_edits=sum(flag(r[k]) for r in rows for k in ('agent_edit_is_zero','patient_edit_is_zero')))


def audit_stage(baselines, results, supplied, reference_baselines, reference_results, stage):
    indexed = require_same_baselines(baselines, reference_baselines)
    expected = {(layer, method) for layer in LAYERS for method in METHODS}
    raw_rows = [r for r in reference_results if r['method']=='raw_centroid' and float(r['alpha'])==1]
    raw = {(int(r['row_id']), int(r['layer'])):r for r in raw_rows}
    if len(raw) != len(raw_rows): raise ValueError('Duplicate reference raw condition')
    seen, groups = defaultdict(set), defaultdict(list)
    for row in results:
        key = int(row['row_id'])
        if key not in indexed: raise ValueError('Result without baseline')
        condition = (int(row['layer']), row['method'])
        if condition not in expected or condition in seen[key]: raise ValueError('Duplicate/unexpected condition')
        seen[key].add(condition)
        if float(row['alpha']) != 1: raise ValueError('Native experiment requires alpha=1')
        base = indexed[key]
        if (row['split'],row['voice']) != (base['split'],base['voice']): raise ValueError('Stratum mismatch')
        bm, em = float(row['baseline_margin']), float(row['edited_margin'])
        if bm != float(base['patient_logit'])-float(base['agent_logit']): raise ValueError('Margin mismatch')
        for name in ('baseline_margin','edited_margin','margin_change','agent_edit_norm','patient_edit_norm'):
            if not math.isfinite(float(row[name])): raise ValueError('Nonfinite result')
        if not math.isclose(float(row['margin_change']),em-bm,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Incorrect margin change')
        for name, value in (('baseline_agent_win',bm<0),('patient_win',em>0),('flip',bm<0 and em>0)):
            if flag(row[name]) != value: raise ValueError('Incorrect outcome flag')
        for role in ('agent','patient'):
            norm = float(row[f'{role}_edit_norm'])
            if norm < 0 or flag(row[f'{role}_edit_is_zero']) != (norm==0): raise ValueError('Invalid native norm/zero flag')
        origin = row['result_origin']
        if origin not in ('scored','reference_raw','exact_identity'): raise ValueError('Unknown result origin')
        if origin == 'exact_identity' and (row['method']=='raw_centroid' or bm!=em or float(row['agent_edit_norm'])!=0 or float(row['patient_edit_norm'])!=0):
            raise ValueError('Identity condition must be an exact no-op')
        if origin == 'reference_raw':
            if row['method']!='raw_centroid' or (key,condition[0]) not in raw: raise ValueError('Invalid reuse')
            old = raw[key,condition[0]]
            for name in ('baseline_margin','edited_margin','margin_change'):
                if float(row[name])!=float(old[name]): raise ValueError('Reused raw output differs')
            for name in ('agent_edit_norm','patient_edit_norm'):
                if not math.isclose(float(row[name]),float(old[name]),rel_tol=1e-7,abs_tol=1e-7):
                    raise ValueError('Raw native norm differs')
        groups[(row['split'],row['voice'],*condition)].append(row)
    if seen.keys()!=indexed.keys() or any(v!=expected for v in seen.values()): raise ValueError('Incomplete condition coverage')
    actual = {k:summarize(v) for k,v in groups.items()}
    saved = {}
    for row in supplied:
        key = (row['split'],row['voice'],int(row['layer']),row['method'])
        if key in saved or float(row['alpha'])!=1: raise ValueError('Duplicate/non-native summary')
        saved[key] = row
    if actual.keys()!=saved.keys(): raise ValueError('Summary coverage differs')
    for key, values in actual.items():
        for field, value in values.items():
            found = num(saved[key][field])
            if (found is None)!=(value is None) or (found is not None and not math.isclose(found,value,rel_tol=1e-7,abs_tol=1e-9)):
                raise ValueError(f'Summary differs: {key}/{field}')
    counts = Counter((r['split'],r['voice']) for r in baselines)
    wanted = {('val','active'):206,('val','passive'):206} if stage=='validation' else {
        ('test','active'):208,('test','passive'):208,('gen_test','active'):840,('gen_test','passive'):840}
    if counts!=wanted: raise ValueError('Unexpected split counts')
    return [dict(split=k[0],voice=k[1],layer=k[2],method=k[3],alpha=1.,**v) for k,v in sorted(actual.items())]


def ranges(summary):
    groups = defaultdict(list)
    for r in summary: groups[r['split'],r['voice'],r['method']].append(r)
    records = []
    for (split,voice,method), rows in sorted(groups.items()):
        if sorted(r['layer'] for r in rows)!=list(LAYERS): raise ValueError('Incomplete layer range')
        records.append(dict(split=split,voice=voice,method=method,
            flip_min=min(r['conditional_flip'] for r in rows),flip_max=max(r['conditional_flip'] for r in rows),
            margin_min=min(r['mean_margin_change'] for r in rows),margin_max=max(r['mean_margin_change'] for r in rows),
            norm_min=min((r['mean_agent_edit_norm']+r['mean_patient_edit_norm'])/2 for r in rows),
            norm_max=max((r['mean_agent_edit_norm']+r['mean_patient_edit_norm'])/2 for r in rows)))
    return records


def plot_results(summary, metric, ylabel, path):
    fig, axes = plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
    for ax,(split,voice) in zip(axes.flat,[(s,v) for s in ('test','gen_test') for v in ('active','passive')]):
        for method in METHODS:
            rows=sorted((r for r in summary if (r['split'],r['voice'],r['method'])==(split,voice,method)),key=lambda r:r['layer'])
            scale=100 if metric=='conditional_flip' else 1
            ax.plot([r['layer'] for r in rows],[scale*r[metric] for r in rows],marker='o',color=COLORS[method],label=LABELS[method])
        ax.set_title(f'{split} · {voice}');ax.set_xticks(LAYERS);ax.grid(alpha=.2)
        ax.set_xlabel('Layer (indice da zero)');ax.set_ylabel(ylabel)
        if metric=='conditional_flip': ax.set_ylim(-3,103)
    handles, labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
    fig.suptitle('Interventi nativi, α=1 fisso; norme non uguagliate',y=.94)
    fig.tight_layout(rect=(0,0,1,.90));fig.savefig(path,dpi=160);plt.close(fig)


def check_number(actual, expected, name, *, tolerance=1e-7):
    actual = num(actual)
    if ((actual is None) != (expected is None) or
            actual is not None and (not math.isfinite(actual) or
                not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=1e-9))):
        raise ValueError(f'Incorrect {name}')


def audit_runtime(run, stage, plan, inputs):
    path = run/stage/'runtime_source.json'
    runtime = json.loads(path.read_text())
    if runtime['source_sha256'] != plan['source_sha256'] or runtime['differences_from_plan']:
        raise ValueError('Runtime sources differ from frozen plan')
    inputs[stage+':runtime'] = path
    for name, sha in plan['source_sha256'].items():
        snapshot = run/stage/'source_snapshot'/name
        if digest(snapshot) != sha: raise ValueError('Runtime source snapshot changed')
        inputs[stage+':snapshot:'+name] = snapshot


def audit_calibration(run, plan, corpus, inputs):
    path = run/'calibration/metadata.json'
    metadata = json.loads(path.read_text())
    if (metadata['status'] != 'complete' or metadata['split'] != 'train' or
            metadata['layers'] != list(LAYERS) or metadata['training_sentences'] != 3300 or
            metadata['training_tokens'] != 6600 or metadata['shared_across_voices'] is not True):
        raise ValueError('Incomplete/non-train calibration')
    inputs['calibration'] = path
    wanted = {f'layer_{layer}.json' for layer in LAYERS}
    if set(metadata['calibration_sha256']) != wanted:
        raise ValueError('Incomplete calibration layers')
    ids = [i for i, row in enumerate(corpus) if row['split'] == 'train']
    vocabulary = sorted({corpus[i][role] for i in ids for role in ('agent','patient')})
    if len(ids) != 3300 or len(vocabulary) != 19 or metadata['vocabulary'] != vocabulary:
        raise ValueError('Calibration vocabulary/training corpus differs')
    pairing_path = run/'calibration/train_pairing.json'
    if digest(pairing_path) != metadata['train_pairing_sha256']:
        raise ValueError('Training pairing changed')
    pairing = json.loads(pairing_path.read_text())
    if pairing['split'] != 'train' or pairing['row_ids'] != ids:
        raise ValueError('Calibration pairing includes non-train rows')
    donors = pairing['donor_token_local_ids']
    if len(donors) != 6600 or sorted(donors) != list(range(6600)):
        raise ValueError('Invalid calibration donor coverage')
    roles = ('agent','patient')
    for local, donor in enumerate(donors):
        source, target = corpus[ids[local//2]], corpus[ids[donor//2]]
        if (donors[donor] != local or donor%2 == local%2 or source['swapped'] == target['swapped'] or
                any(source[k] != target[k] for k in ('quad_id','split','voice','verb','tense','subset')) or
                source[roles[local%2]] != target[roles[donor%2]]):
            raise ValueError('Calibration pairing is not same-filler opposite-role twins')
    inputs['calibration:pairing'] = pairing_path
    calibrations = []
    for name, sha in sorted(metadata['calibration_sha256'].items()):
        path = run/'calibration'/name
        if digest(path) != sha: raise ValueError('Calibration changed')
        obj = json.loads(path.read_text()); perm = obj['permutation']; d = obj['diagnostics']
        if sorted(perm) != list(range(8)) or any(perm[perm[i]] != i for i in range(8)):
            raise ValueError('Invalid native involution')
        if (name != f"layer_{obj['layer']}.json" or obj['training_tokens'] != 6600 or
                d['tokens'] != 6600 or d['slots'] != 8 or d['filler_dim'] != 32 or d['candidate_count'] != 764 or
                d['permutation'] != perm or d['best']['permutation'] != perm or
                d['identity']['permutation'] != list(range(8))):
            raise ValueError('Incorrect frozen calibration protocol')
        denominator = 6600*8*d['filler_dim']
        check_number(d['best']['mse'], d['best']['objective_sse']/denominator, 'calibration MSE')
        check_number(d['identity']['mse'], d['identity']['objective_sse']/denominator, 'identity MSE')
        identity = float(d['identity']['objective_sse'])
        check_number(d['relative_improvement'], (identity-d['best']['objective_sse'])/identity if identity else None,
                     'calibration relative improvement')
        calibrations.append(obj); inputs['calibration:'+name] = path
    audit_runtime(run, 'calibration', plan, inputs)
    return metadata, calibrations


def audit_stage_metadata(metadata, rows, run):
    if (metadata['status'] != 'complete' or metadata['alphas'] != [1.] or metadata['conditions'] != 9 or
            metadata['layers'] != list(LAYERS) or metadata['methods'] != list(METHODS) or
            metadata['norm_matching'] is not False or metadata['exact_reference_baseline'] is not True):
        raise ValueError('Incomplete/non-native stage')
    if metadata['calibration_metadata_sha256'] != digest(run/'calibration/metadata.json'):
        raise ValueError('Stage calibration metadata changed')
    if metadata['result_rows'] != len(rows) or metadata['rows']*9 != len(rows):
        raise ValueError('Incorrect stage row metadata')
    groups = defaultdict(list)
    for row in rows: groups[int(row['layer']),row['method']].append(row)
    saved = {}
    for row in metadata['native_checks']:
        key = row['layer'], row['method']
        if key in saved: raise ValueError('Duplicate native condition check')
        saved[key] = row
    if groups.keys() != saved.keys(): raise ValueError('Native condition checks incomplete')
    identities = set()
    for key, group in groups.items():
        norms = [float(row[role+'_edit_norm']) for row in group for role in ('agent','patient')]
        zero_tokens = sum(x == 0 for x in norms)
        zero_sentences = sum(float(r['agent_edit_norm']) == float(r['patient_edit_norm']) == 0 for r in group)
        values = dict(tokens=len(norms),native_norm_min=min(norms),native_norm_mean=math.fsum(norms)/len(norms),
            native_norm_max=max(norms),exact_zero_tokens=zero_tokens,exact_zero_sentences=zero_sentences)
        for name, value in values.items(): check_number(saved[key][name], value, 'native check '+name)
        if zero_tokens == len(norms): identities.add(key)
        if any((r['result_origin'] == 'exact_identity') != (key in identities) for r in group):
            raise ValueError('Exact identity origin differs from native zero condition')
    declared = [(x['layer'],x['method']) for x in metadata['exact_identity_conditions']]
    if len(set(declared)) != len(declared) or set(declared) != identities:
        raise ValueError('Incorrect exact identity metadata')


def diagnostic_groups(corpus, vocabulary):
    groups = Counter()
    known = Counter()
    for row in corpus:
        if row['split'] not in ('test','gen_test'): continue
        for role in ('agent','patient'):
            supported = row[role] in vocabulary
            for subset in ('all','known' if supported else 'oov'):
                key = row['split'],row['voice'],role,subset
                groups[key] += 1; known[key] += int(supported)
    return groups, known


def audit_diagnostics(run, plan, corpus, calibration, evaluated, inputs):
    path = run/'diagnostics/metadata.json'
    metadata = json.loads(path.read_text())
    names = ('filler_metrics.csv','soft_fidelity.csv','soft_alignment.csv')
    if (metadata['status'] != 'complete' or metadata['layers'] != list(LAYERS) or
            metadata['methods'] != list(METHODS) or metadata['alpha'] != 1. or
            metadata['norm_matching'] is not False or metadata['tokens_per_condition'] != 4192 or
            set(metadata['files_sha256']) != set(names)):
        raise ValueError('Incomplete/non-native diagnostics')
    if (metadata['scientific_source_sha256'] != plan['source_sha256'] or
            metadata['verified_reference_sha256'] != plan['reference_sha256'] or
            metadata['verified_soft_reference_sha256'] != plan['soft_reference_sha256']):
        raise ValueError('Diagnostics source/reference provenance differs')
    script = Path(__file__).with_name('diagnose_native_tpr_nov6.py')
    if digest(script) != metadata['diagnostic_script_sha256']:
        raise ValueError('Diagnostic source changed')
    inputs['diagnostic_source'] = script; inputs['diagnostics:metadata'] = path
    required = {'plan.json','calibration/metadata.json','calibration/train_pairing.json',
                'calibration/runtime_source.json','evaluation/metadata.json','evaluation/results.csv',
                'evaluation/runtime_source.json'} | {'calibration/'+n for n in calibration['calibration_sha256']}
    if set(metadata['input_sha256']) != required: raise ValueError('Incomplete diagnostic inputs')
    for name, sha in metadata['input_sha256'].items():
        if digest(run/name) != sha: raise ValueError('Diagnostic input changed')
        inputs['diagnostics:input:'+name] = run/name
    tables = {}
    for name in names:
        path = run/'diagnostics'/name
        if digest(path) != metadata['files_sha256'][name]: raise ValueError('Diagnostic CSV changed')
        tables[name] = read_csv(path); inputs['diagnostics:'+name] = path
        if len(tables[name]) != metadata['metrics_rows'][name]: raise ValueError('Diagnostic row count differs')
    edit = metadata['edit_verification']
    if edit['verified_result_rows'] != len(evaluated) or edit['verified_token_edits'] != 2*len(evaluated):
        raise ValueError('Diagnostic edit coverage differs')
    zero_tokens = sum(flag(row[role+'_edit_is_zero']) for row in evaluated for role in ('agent','patient'))
    if edit['zero_token_edits'] != zero_tokens or not 0 <= edit['maximum_relative_norm_error'] <= 1e-6:
        raise ValueError('Diagnostic edit verification differs')
    groups, known = diagnostic_groups(corpus, set(calibration['vocabulary']))
    expected = {(layer, method, *key) for layer in LAYERS for method in METHODS for key in groups}
    seen = set(); zero = Counter()
    for row in evaluated:
        for role in ('agent','patient'):
            item = corpus[int(row['row_id'])]
            subset = 'known' if item[role] in calibration['vocabulary'] else 'oov'
            for vocabulary in ('all',subset):
                zero[int(row['layer']),row['method'],row['split'],row['voice'],role,vocabulary] += int(flag(row[role+'_edit_is_zero']))
    for row in tables['filler_metrics.csv']:
        key = (int(row['layer']),row['method'],row['split'],row['voice'],row['source_role'],row['vocabulary'])
        if key not in expected or key in seen: raise ValueError('Duplicate/unexpected filler diagnostic')
        seen.add(key); token_count, supported = groups[key[2:]], known[key[2:]]
        if int(row['tokens']) != token_count or int(row['known_tokens']) != supported:
            raise ValueError('Filler diagnostic denominator differs')
        for prefix in ('filler_before','filler_after_target'):
            count = int(row[prefix+'_correct'])
            if not 0 <= count <= supported: raise ValueError('Invalid filler correct count')
            check_number(row[prefix+'_accuracy'], count/supported if supported else None, 'filler accuracy')
        retention = float(row['filler_predicted_label_retention'])
        if not math.isfinite(retention) or not 0 <= retention <= 1: raise ValueError('Invalid filler retention')
        if int(row['zero_edit_tokens']) != zero[key]: raise ValueError('Diagnostic zero-token count differs')
        profile = num(row['native_ideal_logits_relative_error'])
        if (profile is None) != (row['method'] != 'tpr_native_role_swap') or profile is not None and (not math.isfinite(profile) or profile < 0):
            raise ValueError('Invalid native profile error')
    if seen != expected: raise ValueError('Incomplete filler diagnostics')
    indexed = {(int(r['layer']),r['method'],r['split'],r['voice'],r['source_role'],r['vocabulary']):r
               for r in tables['filler_metrics.csv']}
    for key, row in indexed.items():
        if key[-1] != 'all': continue
        parts = [indexed[key[:-1]+(subset,)] for subset in ('known','oov') if key[:-1]+(subset,) in indexed]
        for name in ('filler_before_correct','filler_after_target_correct','zero_edit_tokens'):
            if int(row[name]) != sum(int(part[name]) for part in parts):
                raise ValueError('Filler vocabulary partitions disagree')
        weighted = sum(float(part['filler_predicted_label_retention'])*int(part['tokens']) for part in parts)/int(row['tokens'])
        check_number(row['filler_predicted_label_retention'],weighted,'filler retention partitions')
        raw = indexed[(key[0],'raw_centroid',*key[2:])]
        if row['filler_before_correct'] != raw['filler_before_correct']:
            raise ValueError('Filler baseline differs across methods')
    expected = {(layer,*key) for layer in LAYERS for key in groups}; seen = set()
    for row in tables['soft_fidelity.csv']:
        key = (int(row['layer']),row['split'],row['voice'],row['source_role'],row['vocabulary'])
        if key not in expected or key in seen: raise ValueError('Duplicate/unexpected fidelity diagnostic')
        seen.add(key); count = groups[key[1:]]
        if int(row['tokens']) != count or int(row['slots_per_token']) != 8 or int(row['total_slots']) != 8*count:
            raise ValueError('Fidelity denominator differs')
        for number, fraction, denominator in (
                ('slot_matches','slot_code_match_fraction',8*count),
                ('all_codes_match_tokens','all_codes_match_fraction',count),
                ('bag_retained_tokens','code_bag_retention',count),
                ('target_bag_preserved_tokens','target_code_bag_preservation',count)):
            value = int(row[number])
            if not 0 <= value <= denominator: raise ValueError('Invalid fidelity count')
            check_number(row[fraction], value/denominator, 'fidelity fraction')
        if int(row['target_bag_preserved_tokens']) != count or float(row['target_code_bag_preservation']) != 1:
            raise ValueError('Native target must preserve every code bag')
        matches, all_match, retained = (int(row[k]) for k in ('slot_matches','all_codes_match_tokens','bag_retained_tokens'))
        if not 8*all_match <= matches <= 8*all_match+7*(count-all_match) or all_match > retained:
            raise ValueError('Inconsistent fidelity counts')
    if seen != expected: raise ValueError('Incomplete fidelity diagnostics')
    indexed = {(int(r['layer']),r['split'],r['voice'],r['source_role'],r['vocabulary']):r
               for r in tables['soft_fidelity.csv']}
    for key, row in indexed.items():
        if key[-1] != 'all': continue
        parts = [indexed[key[:-1]+(subset,)] for subset in ('known','oov') if key[:-1]+(subset,) in indexed]
        for name in ('slot_matches','all_codes_match_tokens','bag_retained_tokens','target_bag_preserved_tokens'):
            if int(row[name]) != sum(int(part[name]) for part in parts):
                raise ValueError('Fidelity vocabulary partitions disagree')
    expected = {(layer,split,voice) for layer in LAYERS for split in ('val','test','gen_test') for voice in ('active','passive')}
    seen = set(); sentence_counts = Counter((r['split'],r['voice']) for r in corpus)
    for row in tables['soft_alignment.csv']:
        key = int(row['layer']),row['split'],row['voice']
        if key not in expected or key in seen: raise ValueError('Duplicate/unexpected alignment diagnostic')
        seen.add(key)
        if int(row['tokens']) != 2*sentence_counts[key[1:]]: raise ValueError('Alignment denominator differs')
        identity, permuted = float(row['identity_mse']),float(row['permuted_mse'])
        if not all(math.isfinite(x) and x >= 0 for x in (identity,permuted)): raise ValueError('Invalid alignment MSE')
        check_number(row['relative_mse_reduction'],(identity-permuted)/identity if identity else None,'alignment reduction')
    if seen != expected: raise ValueError('Incomplete alignment diagnostics')
    if set(metadata['alignment_pairing_sha256']) != {'val','test','gen_test'}:
        raise ValueError('Incomplete diagnostic pairing hashes')
    for split in ('val','test','gen_test'):
        ids = [i for i,r in enumerate(corpus) if r['split'] == split]
        lookup = {(corpus[i]['quad_id'],corpus[i]['voice'],corpus[i]['swapped']):local for local,i in enumerate(ids)}
        donors = []
        for i in ids:
            row = corpus[i]; twin = lookup[row['quad_id'],row['voice'],str(1-int(row['swapped']))]
            if row['agent'] != corpus[ids[twin]]['patient'] or row['patient'] != corpus[ids[twin]]['agent']:
                raise ValueError('Diagnostic pairing does not preserve fillers')
            donors.extend((2*twin+1,2*twin))
        expected_hash = hashlib.sha256(json.dumps(dict(row_ids=ids,donor_token_local_ids=donors),sort_keys=True).encode()).hexdigest()
        if expected_hash != metadata['alignment_pairing_sha256'][split]:
            raise ValueError('Diagnostic pairing hash differs')
    return tables, metadata


def diagnostic_tables(tables):
    filler = tables['filler_metrics.csv']; fidelity = tables['soft_fidelity.csv']; alignment = tables['soft_alignment.csv']
    filler_rows = []; fidelity_rows = []; alignment_rows = []; profile_rows = []
    def span(values, fmt=percent):
        values = list(values)
        return f'{fmt(min(values))}–{fmt(max(values))}'
    for split in ('test','gen_test'):
        for voice in ('active','passive'):
            for method in METHODS:
                rows = [r for r in filler if (r['split'],r['voice'],r['method'],r['vocabulary']) == (split,voice,method,'all')]
                per_layer = []
                for layer in LAYERS:
                    selected = [r for r in rows if int(r['layer']) == layer]
                    n = sum(int(r['tokens']) for r in selected); k = sum(int(r['known_tokens']) for r in selected)
                    per_layer.append((n,k,sum(int(r['filler_before_correct']) for r in selected)/k,
                        sum(int(r['filler_after_target_correct']) for r in selected)/k,
                        sum(float(r['filler_predicted_label_retention'])*int(r['tokens']) for r in selected)/n))
                filler_rows.append([split,voice,LABELS[method],per_layer[0][0],per_layer[0][1],
                    span(r[2] for r in per_layer),span(r[3] for r in per_layer),span(r[4] for r in per_layer)])
                if method == 'tpr_native_role_swap':
                    profile_rows.append([split,voice,span((float(r['native_ideal_logits_relative_error']) for r in rows),lambda x:f'{x:.3g}')])
            rows = [r for r in fidelity if (r['split'],r['voice'],r['vocabulary']) == (split,voice,'all')]
            totals = []
            for layer in LAYERS:
                selected = [r for r in rows if int(r['layer']) == layer]
                n = sum(int(r['tokens']) for r in selected)
                totals.append((n,sum(int(r['slot_matches']) for r in selected)/(8*n),
                    sum(int(r['all_codes_match_tokens']) for r in selected)/n,
                    sum(int(r['bag_retained_tokens']) for r in selected)/n))
            fidelity_rows.append([split,voice,totals[0][0],8*totals[0][0],span(r[1] for r in totals),
                span(r[2] for r in totals),span(r[3] for r in totals),'100%'])
    for split in ('val','test','gen_test'):
        for voice in ('active','passive'):
            rows = [r for r in alignment if (r['split'],r['voice']) == (split,voice)]
            alignment_rows.append([split,voice,int(rows[0]['tokens']),
                span((float(r['identity_mse']) for r in rows),lambda x:f'{x:.4f}'),
                span((float(r['permuted_mse']) for r in rows),lambda x:f'{x:.4f}'),
                span(float(r['relative_mse_reduction']) for r in rows)])
    return filler_rows, fidelity_rows, alignment_rows, profile_rows


def report(run_dir):
    run=Path(run_dir).resolve();root=Path(__file__).resolve().parents[2]
    plan=json.loads((run/'plan.json').read_text())
    if plan['methods']!=list(METHODS) or plan['layers']!=list(LAYERS) or plan['alphas']!=[1.] or plan['fixed_alpha']!=1 or plan['norm_matching'] is not False:
        raise ValueError('Expected frozen native protocol')
    inputs={'plan':run/'plan.json'}
    for relative,sha in plan['source_sha256'].items():
        if digest(root/relative)!=sha: raise ValueError('Scientific source changed')
        inputs['source:'+relative]=root/relative
    for prefix in ('reference','norm_reference','soft_reference'):
        directory=Path(plan[prefix+'_run'])
        for relative,sha in plan[prefix+'_sha256'].items():
            if digest(directory/relative)!=sha: raise ValueError('Reference changed')
            inputs[prefix+':'+relative]=directory/relative
    corpus_path = root/'data/active_pass_long/corpus.csv'
    if digest(corpus_path) != plan['corpus_sha256']: raise ValueError('Frozen corpus changed')
    corpus = read_csv(corpus_path); inputs['corpus'] = corpus_path
    if Counter(r['split'] for r in corpus) != plan['expected_counts']:
        raise ValueError('Corpus split counts differ')
    calibration, calibrations = audit_calibration(run, plan, corpus, inputs)
    summaries={};baseline_rows=[];result_rows={}
    reference=Path(plan['norm_reference_run'])
    for stage in ('validation','evaluation'):
        metadata=json.loads((run/stage/'metadata.json').read_text())
        paths={name:run/stage/(name+'.csv') for name in ('baseline','results','summary')}
        if digest(paths['results'])!=metadata['results_sha256']: raise ValueError('Stage results changed')
        base,res,saved=(read_csv(paths[n]) for n in ('baseline','results','summary'))
        summaries[stage]=audit_stage(base,res,saved,read_csv(reference/stage/'baseline.csv'),read_csv(reference/stage/'results.csv'),stage)
        audit_stage_metadata(metadata,res,run)
        audit_runtime(run,stage,plan,inputs)
        for row in base:
            source = corpus[int(row['row_id'])]
            if any(source[k] != row[k] for k in ('split','voice','agent','patient')):
                raise ValueError('Baseline labels differ from corpus')
        for row in res:
            source = corpus[int(row['row_id'])]
            known = sum(source[role] in calibration['vocabulary'] for role in ('agent','patient'))
            if row['quad_id'] != source['quad_id'] or int(row['known_tokens']) != known or flag(row['sentence_has_oov']) != (known < 2):
                raise ValueError('Result corpus/vocabulary labels differ')
        baseline_rows+=base;result_rows[stage]=len(res)
        inputs[stage+':metadata']=run/stage/'metadata.json'
        for name,path in paths.items(): inputs[stage+':'+name]=path
    if len(baseline_index(baseline_rows))!=2508: raise ValueError('Validation/heldout overlap')
    tables, diagnostics = audit_diagnostics(run,plan,corpus,calibration,res,inputs)
    filler_table, fidelity_table, alignment_table, profile_table = diagnostic_tables(tables)
    out=run/'report';out.mkdir(exist_ok=False)
    comparison=ranges(summaries['evaluation'])
    write_csv(out/'comparison_summary.csv',summaries['evaluation'])
    write_csv(out/'comparison_ranges.csv',comparison)
    write_csv(out/'validation_summary.csv',summaries['validation'])
    plot_results(summaries['evaluation'],'conditional_flip','Flip agente → paziente (%)',out/'native_steering.png')
    plot_results(summaries['evaluation'],'mean_margin_change','Δ logit (paziente − agente)',out/'native_margins.png')
    calibration_table=[]
    for c in sorted(calibrations,key=lambda c:c['layer']):
        d=c['diagnostics'];calibration_table.append([c['layer'],str(c['permutation']),percent(d['relative_improvement'])])
    rows=[]
    for r in comparison:
        selected = [x for x in summaries['evaluation'] if (x['split'],x['voice'],x['method']) == (r['split'],r['voice'],r['method'])]
        zeros = [x['zero_token_edits'] for x in selected]
        rows.append([r['split'],r['voice'],LABELS[r['method']],
            f"{selected[0]['baseline_agent_wins']}/{selected[0]['sentences']}",
            f"{percent(r['flip_min'])}–{percent(r['flip_max'])}",f"{r['norm_min']:.2f}–{r['norm_max']:.2f}",
            f"{min(zeros)}–{max(zeros)}/{2*selected[0]['sentences']}"])

    text=['# Steering con fattori nativi TPR e Soft TPR','',
          'Checkpoint originali congelati, nessuna nuova head e nessuna fattorizzazione di centroidi. '
          'Il protocollo del 6 novembre usa Qwen3-8B-Base, layer 20–22, prompt e dati identici ai confronti precedenti. '
          'Sono valutate 412 frasi validation e 2.096 heldout. α=1 è fissato prima della valutazione; '
          'le norme native non vengono uguagliate. Questo confronto comprende quindi sia direzione sia ampiezza degli interventi.','',
          '## TPR: scambio dei canali originali','',
          'Dai parametri originali M, R e F si calcolano x=(h−μ)/s, Z=reshape(xM), U=RZ. '
          'P scambia i due ruoli annotati. Si impone Z*=R⁻¹PU e si applica Δh=s·vec(Z*−Z)·M⁺. '
          'Poiché i logits sono L=UFᵀ, il target soddisfa L*=PL: viene trasferito l’intero profilo filler. '
          'R fornisce vettori di lettura; R⁻¹ fornisce i corrispondenti vettori di scrittura. '
          'Le direzioni sono precalcolate sulle attivazioni baseline salvate e poi iniettate nel modello; '
          'non vengono ricalcolate su stati già modificati. Non si usano etichette lessicali per costruire l’intervento. '
          'La proprietà è algebrica prima degli arrotondamenti; '
          'la diagnostica misura anche gli errori dopo il cast BF16.','',
          '## Soft TPR: permutazione dei codici originali','',
          'Gli otto slot, il codebook, encoder e decoder restano originali. Si cerca una sola permutazione involutiva '
          'per layer, condivisa fra frasi attive e passive, fra tutte le 764 possibilità inclusa l’identità. '
          'Il criterio usa soltanto 6.600 token train: avvicinare i codici quantizzati dello stesso filler '
          'nelle due frasi gemelle in cui cambia ruolo. I pareggi favoriscono meno slot mossi e poi l’ordine lessicografico. '
          'Questa è una calibrazione discreta degli slot, non l’apprendimento di nuovi vettori o la prova che '
          'gli slot abbiano già etichette agente/paziente.','',
          table(['Layer','Permutazione (indici da zero)','Riduzione errore controfattuale train'],calibration_table),'',
          'All’inferenza q*=q[:,P,:] e Δh=s·W_decoder·vec(bind(q*−q)). Lo stesso P si applica ai due token; '
          'non serve un ulteriore segno agente/paziente. Non vengono usati donatori heldout per lo steering. '
          'Il target conserva esattamente il multinsieme dei codici e dei filler latenti, ma ciò non garantisce '
          'né l’identità lessicale né il recupero dei codici target dopo ri-encoding dello stato modificato. '
          'Gli interventi nulli rimangono nulli.','',
          '## Risultati heldout','',
          'Flip condizionato: baseline strettamente favorevole all’agente e intervento strettamente favorevole '
          'al paziente. I pareggi baseline sono esclusi dal denominatore. Gli intervalli sono min–max sui tre layer '
          'storicamente selezionati, non intervalli di confidenza o selezioni del layer migliore. '
          'I token con edit esattamente nullo sono conteggiati anche quando soltanto uno dei due token della frase '
          'rimane invariato; una condizione interamente nulla riusa esattamente la baseline.','',
          table(['Split','Voce','Metodo','Frasi eleggibili/totali','Flip sui layer','Norma media/token','Token a edit nullo'],rows),'',
          '![Steering nativo](native_steering.png)','', '![Margini nativi](native_margins.png)','',
          '## Diagnostiche e limiti','',
          'Le diagnostiche in ../diagnostics ricostruiscono localmente BF16(h.float()+Δh) dalla cache: '
          'non sono attivazioni raccolte dagli hook durante la valutazione. Il controllo lessicale usa la probe '
          'TPR originale per tutti i metodi, nel ruolo sorgente prima e nel ruolo target dopo l’intervento. '
          'L’accuracy dei filler riguarda soltanto i nomi nel vocabolario train. Per gli OOV, mantenere la stessa '
          'predizione nel vocabolario noto non dimostra conservazione del nome sconosciuto. '
          'La distanza Soft dai controfattuali heldout è esclusivamente una diagnostica successiva, senza rifit.','',
          '### Filler letti dalla TPR originale','',
          'Ruoli sorgente agente e paziente aggregati entro split/voce. N è il numero di token per layer; '
          'K è il sottoinsieme con un filler nel vocabolario train di 19 elementi. Le accuracy prima/dopo '
          'hanno denominatore K; la stabilità della predizione ha denominatore N e include gli OOV. '
          'Gli intervalli coprono i tre layer. [CSV completo, con ruoli e gruppi known/OOV](../diagnostics/filler_metrics.csv).','',
          table(['Split','Voce','Intervento','N token/layer','K noti/layer','Accuracy prima (K)',
                 'Accuracy ruolo target (K)','Stessa etichetta (N)'],filler_table),'',
          'Lo scambio TPR cerca di trasferire l’intero profilo dei logits al ruolo opposto. '
          'L’errore relativo dopo l’arrotondamento BF16 è ||L_dopo − P L_prima||₂ / ||P L_prima||₂; '
          'gli intervalli seguenti coprono layer e ruolo sorgente separatamente, senza mediare questi rapporti.','',
          table(['Split','Voce','Errore relativo profilo TPR'],profile_table),'',
          '### Fedeltà dei codici Soft TPR','',
          'Ogni token ha otto slot. «Slot target» usa come denominatore il numero totale di slot; '
          '«Tutti gli slot» e «Multinsieme dopo» usano N token. Queste tre colonne misurano il re-encoding '
          'dello stato modificato localmente. «Multinsieme target» riguarda invece il target costruito '
          'per permutazione prima del decoder: il suo 100% è una proprietà dell’operazione. '
          'In tutti e tre i layer la permutazione scambia solo due slot su otto: confrontando i codici '
          'originali con il target, almeno il 75% degli slot coincide già senza intervento. '
          'La corrispondenza per slot va quindi letta insieme al recupero dell’intero insieme dei codici. '
          'Nessuna di queste misure identifica automaticamente il nome lessicale. '
          '[CSV completo](../diagnostics/soft_fidelity.csv).','',
          table(['Split','Voce','N token/layer','Slot/layer','Slot target','Tutti gli slot',
                 'Multinsieme dopo','Multinsieme target'],fidelity_table),'',
          '### Allineamento Soft ai gemelli controfattuali','',
          'MSE sui codici quantizzati di token con lo stesso filler nella frase gemella di ruolo opposto. '
          'N è il numero di token per layer; la MSE media tutti gli otto slot e le 32 coordinate filler. '
          'Identità e permutazione train congelata sono confrontate sugli stessi token. '
          'Riduzione = (MSE_identità − MSE_permutazione) / MSE_identità; un valore negativo è un peggioramento. '
          'I gemelli validation/heldout sono usati soltanto per questa diagnostica successiva, mai per scegliere '
          'o costruire lo steering. [CSV completo](../diagnostics/soft_alignment.csv).','',
          table(['Split','Voce','N token/layer','MSE identità','MSE permutazione','Riduzione MSE'],alignment_table),'',
          'gen_test misura generalizzazione a Mary, teacher e rock entro gli stessi schemi sintattici. '
          'Nelle attive il primo nome è sempre agente; nelle passive è sempre paziente. Il disegno non separa '
          'completamente ruolo e posizione. Un flip del LM non dimostra da solo uno scambio semantico puro.','',
          'Il [confronto precedente con centroidi TPR e Soft TPR]('
          +str(Path(plan['soft_reference_run'])/'report/report.md')+') usa anch’esso α=1 alla scala originale, '
          'con norme diverse fra metodi. I centroidi sono stimati separatamente per voce; lo scambio TPR nativo '
          'non usa medie di ruolo, mentre Soft TPR usa una sola permutazione train per layer condivisa fra le voci. '
          'Queste differenze vanno considerate nel confronto.','',
          '[Statistiche per condizione](comparison_summary.csv), [intervalli](comparison_ranges.csv), '
          '[validation](validation_summary.csv), [provenienza diagnostica](../diagnostics/metadata.json), '
          '[hash e verifica](metadata.json).','',
          '## Riproduzione','', '```bash',
          'TPR_PYTHON=/tmp/tpr-binding-gpu-venv/bin/python \\',
          '  bash scripts/temporary/execute_native_tpr_nov6.sh ac-tpr-cache/NOME_NUOVO', '```','']
    (out/'report.md').write_text('\n'.join(text))
    metadata=dict(status='complete',created_at=datetime.now(timezone.utc).isoformat(),fixed_alpha=1.,norm_matching=False,
                  validation_sentences=412,evaluation_sentences=2096,result_rows=result_rows,
                  exact_norm_reference_baselines=True,calibration_train_only=True,
                  diagnostic_csv_hashes_verified=True,stage_snapshots_verified=True,
                  diagnostics_rows=diagnostics['metrics_rows'],source_sha256=digest(Path(__file__)),
                  input_sha256={name:digest(path) for name,path in inputs.items()},
                  output_sha256={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True,type=Path)
    print(json.dumps(report(parser.parse_args().run_dir),indent=2))
