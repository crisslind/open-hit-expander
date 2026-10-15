import tempfile
from pathlib import Path
import pandas as pd
from open_hit_expander.boltz2 import collect_boltz2_replicate_results
def test_pose_metadata():
    with tempfile.TemporaryDirectory() as temp:
        rows=[]
        for job, parent, vals in [('mol_parent',True,[7.0,7.2,7.5]),('mol_a',False,[7.1,7.0,7.8])]:
            for rep,value in enumerate(vals,1):
                f=Path(temp)/f'{job}_{rep}.cif';f.write_text('dummy')
                rows.append(dict(job_id=job,source_id='mol',is_parent=parent,replicate=rep,seed=1000+rep,predicted_pic50=value,structure_path=str(f)))
        state={'root':Path(temp),'replicate_results':pd.DataFrame(rows)}
        output,_=collect_boltz2_replicate_results(state)
        a=output.set_index('job_id').loc['mol_a']
        assert abs(a['predicted_pic50']-7.1)<1e-8
        assert abs(a['best_pic50']-7.8)<1e-8
        assert int(a['best_replicate'])==3
        assert a['best_structure_path'].endswith('mol_a_3.cif')
        assert a['structure_path'].endswith('mol_a_1.cif')
        reloaded=pd.read_csv(Path(temp)/'hit_expander_boltz2_all_results.csv')
        assert 'best_structure_path' in reloaded
        print('PASS: median selection, best-pose metadata, CSV persistence')
