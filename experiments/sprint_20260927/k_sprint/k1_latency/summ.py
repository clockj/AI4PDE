import json,glob,sys,collections
import numpy as np
for f in sorted(glob.glob('/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/logs/k1_t1lat*.jsonl')):
    print(f.split('/')[-1])
    agg=collections.defaultdict(lambda: collections.defaultdict(list)); md=collections.defaultdict(set); cold=collections.defaultdict(list)
    for l in open(f):
        r=json.loads(l); t=r['tag']
        for n,v in r['warm'].items(): agg[t][n].append(v['median'])
        for n,v in r['md5'].items(): md[(n,r['dtype'])].add(v[:8])
        cold[t].append((r['cold_n'],r['dtype'],round(r['cold_ms_per_win'],3),r['batch']))
        if 'breakdown_ms_per_win_N56' in r and t=='base' and 'bd' not in agg: agg['bd']=r['breakdown_ms_per_win_N56']
    for t in agg:
        if t=='bd': print('   breakdown N56', {k:round(v,3) for k,v in agg[t].items()}); continue
        print(f"  {t:10s} warm " + " ".join(f"N{n}:{np.mean(v):.3f}" for n,v in agg[t].items()) + "  | cold(N,dtype,ms/win,batch) " + str(cold[t]))
    print('   distinct md5 per (N,dtype):', {k:len(v) for k,v in md.items()})
for f in sorted(glob.glob('/fs/scratch/PAS1187/qwu/tmp/realpde/trick0927/logs/k1_t2lat*.jsonl')):
    print(f.split('/')[-1])
    for l in open(f):
        r=json.loads(l); print("  ", r['tag'], 'mean', round(r['mean_ms'],3), 'median', round(r['median_ms'],3), 'p90', round(r['p90_ms'],3), 'graph', r['graph'], 'arena', r['arena'], r['type'], r['md5'][:8])
