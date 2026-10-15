"""Read saved Boltz results and generate the v0.7 report; never invokes Boltz or PLIP."""
import argparse
from pathlib import Path
import pandas as pd
from open_hit_expander.boltz2 import load_boltz2_run
from open_hit_expander.report import build_compound_assessment


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--timestamp',default='20261013_111348_145796')
    p.add_argument('--output-directory',default='hit_expander_output')
    args=p.parse_args()
    state,results,ranked=load_boltz2_run(args.timestamp,output_directory=args.output_directory,reparse=True)
    root=Path(state['root'])
    assert len(results)>0,'No compounds were recovered'
    required=['job_id','predicted_pic50','structure_path','best_pic50','best_structure_path']
    missing=[x for x in required if x not in results]
    assert not missing,f'Missing result columns: {missing}'
    for _,r in results.iterrows():
        assert pd.notna(r['predicted_pic50']),f'Missing median potency: {r.job_id}'
        assert pd.notna(r['best_pic50']),f'Missing best potency: {r.job_id}'
        for k in ('structure_path','best_structure_path'):
            assert pd.notna(r[k]) and Path(r[k]).is_file(),f'Missing {k} for {r.job_id}: {r[k]}'
    print(f'PASS: {len(results)} compound(s) with median and best pose paths')
    analysis=root/'interaction_analysis'
    details=analysis/'interaction_details.csv'
    status=analysis/'interaction_status.csv'
    if not (details.exists() and status.exists()):
        print('NOTE: saved PLIP CSVs missing; report will show interaction metrics unavailable')
    reportdir=root/'compound_assessment'
    summary,html=build_compound_assessment(results,details=details if details.exists() else None,
        status=status if status.exists() else None,
        replicate_predictions=root/'replicate_predictions.csv',output_dir=reportdir)
    assert html.is_file() and (reportdir/'compound_summary.csv').is_file()
    print(f'PASS: assessment ({len(summary)} rows) saved at {html}')
    if status.exists():
        print('PLIP statuses:',pd.read_csv(status).to_string(index=False,max_rows=12))
    print('PASS: no Boltz or PLIP inference called')

if __name__=='__main__':main()
