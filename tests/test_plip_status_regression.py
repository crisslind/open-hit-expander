import tempfile
import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from open_hit_expander.report import build_compound_assessment

@pytest.mark.parametrize("status_value", ["ok", "success"])
def test_status(status_value):
    results = pd.DataFrame([
        {'job_id':'mol_1_parent','is_parent':True,'predicted_pic50':7.2, 'n_success':2},
        {'job_id':'mol_1_analogue_0001','is_parent':False,'predicted_pic50':7.4, 'n_success':2},
    ])
    details = pd.DataFrame([
        {'job_id':'mol_1_parent','interaction':'hbond','residue':'ASN100:A'},
        {'job_id':'mol_1_parent','interaction':'hydrophobic','residue':'LEU101:A'},
        {'job_id':'mol_1_analogue_0001','interaction':'hbond','residue':'ASN100:A'},
        {'job_id':'mol_1_analogue_0001','interaction':'hbond','residue':'SER105:A'},
    ])
    status = pd.DataFrame([{'job_id':job,'status':status_value} for job in results.job_id])
    with tempfile.TemporaryDirectory() as tmp:
        table, path = build_compound_assessment(results,details=details,status=status,output_dir=tmp)
        t=table.set_index('job_id')
        assert t.loc['mol_1_parent','n_contact_features']==2
        assert t.loc['mol_1_parent','interaction_similarity_to_parent']==1
        assert t.loc['mol_1_analogue_0001','n_contact_features']==2
        assert abs(t.loc['mol_1_analogue_0001','interaction_similarity_to_parent']-1/3)<1e-9
        assert 'SER105:A (hbond)' in t.loc['mol_1_analogue_0001','gained_interactions']
        assert 'LEU101:A (hydrophobic)' in t.loc['mol_1_analogue_0001','lost_interactions']
        assert path.exists()
        assert 'Gained: SER105:A (hbond)' in path.read_text()
        persisted=pd.read_csv(Path(tmp)/'compound_summary.csv')
        assert persisted['n_contact_features'].notna().all()

