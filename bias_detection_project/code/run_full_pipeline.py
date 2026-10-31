
"""Leakage-controlled, resumable three-model pipeline for the supplied Colab project."""
from __future__ import annotations
import gc
import json
import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

LOG = logging.getLogger('bias_pipeline')
TARGETS = ('score_blind', 'score_biased_gender', 'score_biased_ethnicity')
PAIRS = [('baseline','bertidraaf_lite'),('bertidraaf_lite','improved'),('baseline','improved')]
STAGES = ('ingest','preprocess','counterfactual','prepare','baseline','improved','bertidraaf_lite','fairness','report')
CONFIG = dict(epochs=6, fine_tune=True, faircv_targets=list(TARGETS), time_budget_minutes=20, seed=42)

def _read(path):
    return pd.read_csv(path, low_memory=False, dtype={'resume_id':str,'original_id':str})

def _gpu_cleanup(*objects):
    for obj in objects:
        if obj is not None:
            del obj
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError: pass

def _prediction_keys(df):
    return df[['dataset','target','resume_id']].fillna('').astype(str).apply(tuple,axis=1)

def _assemble(scorer, test, reference_columns=None):
    src = test.source_dataset.astype(str).str.lower()
    k = scorer.predict_kaggle(test[src.str.contains('kaggle',regex=False)])
    k.insert(0,'dataset','kaggle')
    k.insert(1,'target','category_label')
    f = scorer.predict_faircv(test[src.str.contains('faircv',regex=False)])
    f.insert(0,'dataset','faircv')
    result = pd.concat([k,f],ignore_index=True)
    if reference_columns is not None:
        if set(result.columns) != set(reference_columns):
            raise ValueError(f'Prediction schema mismatch: missing {set(reference_columns)-set(result.columns)}, extra {set(result.columns)-set(reference_columns)}')
        result = result[list(reference_columns)]
    return result

def run_pipeline(project_root='/content/drive/MyDrive/bias_detection_project/', force=False,
                 force_stages=None, joint_test=None, n_iterations=1000):
    """Run stages, skipping existing outputs unless forced.

    Pass joint_test from the LIVE notebook explicitly: run_pipeline(joint_test=globals().get('joint_test')).
    If forcing upstream preparation, all dependent prediction/report stages are rerun.
    'improved_predictions_kaggle_locked.csv' is strictly read-only; never synthesized.
    """
    root = Path(project_root).expanduser().resolve()
    sys.path.insert(0,str(root/'code'))
    from resume_ingestor import ResumeIngestor
    from preprocessor import Preprocessor
    from name_swap_auditor import NameSwapAuditor
    from data_prep import load_prepared
    from baseline_scorer import BaselineScorer
    from comparison import paired_delta, verdict, placement
    from fairness_auditor import FairnessAuditor
    from report_dashboard import ReportDashboard
    from bertidraaf_lite_scorer import build_matched_pair, signature_drift, assert_matched
    from embedding_engine import EmbeddingEngine
    from improved_scorer import ImprovedScorer
    from model_artifacts import save_baseline, save_neural
    if not LOG.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s', stream=sys.stdout)
    bad = set(force_stages or []) - set(STAGES)
    if bad: raise ValueError(f'Unknown stages: {sorted(bad)}; valid: {STAGES}')
    forced = set(STAGES if force else force_stages or [])
    dependencies = {
      'ingest': STAGES, 'preprocess': STAGES[1:], 'counterfactual': STAGES[2:],
      'prepare': STAGES[3:], 'baseline': ('baseline','fairness','report'),
      'improved': ('improved','fairness','report'),
      'bertidraaf_lite': ('bertidraaf_lite','fairness','report'),
      'fairness': ('fairness','report'), 'report': ('report',)}
    rerun = {s for f in forced for s in dependencies[f]}
    proc = root/'data/processed'; reports=root/'reports'; artifacts=root/'artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts/'baseline').mkdir(exist_ok=True)
    (artifacts/'improved').mkdir(exist_ok=True)
    (artifacts/'bertidraaf_lite').mkdir(exist_ok=True)
    proc.mkdir(parents=True,exist_ok=True); reports.mkdir(parents=True,exist_ok=True)
    paths = {s:proc/f for s,f in [('ingest','ingested_resumes.csv'),('preprocess','preprocessed_resumes.csv'),
             ('counterfactual','counterfactual_resumes.csv'),('baseline','baseline_predictions.csv'),
             ('improved','improved_predictions.csv'),('bertidraaf_lite','bertidraaf_lite_predictions.csv'),
             ('fairness','fairness_comparison.csv')]}
    paths['prepare']=proc/'faircv_thresholds.json';paths['report']=reports/'fairness_report.html'
    split_path=proc/'splits.csv';paired_path=reports/'fairness_paired_differences.csv'
    def needed(stage, *also):
        output=[paths[stage],*also]
        do=stage in rerun or not all(p.is_file() and p.stat().st_size>0 for p in output)
        LOG.info('%s: %s',stage,'RUN' if do else 'SKIP (outputs exist)')
        return do
    def stage_save(df,path):
        df.to_csv(path,index=False)
        LOG.info('saved %s rows=%d',path,len(df))
    if needed('ingest'):
        df=ResumeIngestor().ingest_all(str(root/'data/raw/kaggle/Resume/Resume.csv'),
                                       str(root/'data/raw/faircvdb/FairCVdb.npy'))
        stage_save(df,paths['ingest']);del df
    if needed('preprocess'):
        stage_save(Preprocessor().process(_read(paths['ingest'])),paths['preprocess'])
    if needed('counterfactual'):
        stage_save(NameSwapAuditor(seed=42).process_dataset(_read(paths['preprocess'])),paths['counterfactual'])
    if needed('prepare',split_path):
        cf=_read(paths['counterfactual'])
        raw=np.load(root/'data/raw/faircvdb/FairCVdb.npy',allow_pickle=True).item()
        labs=[]
        for native in ('Train','Test'):
            labs.append(pd.DataFrame({'original_id':[f'faircv_{native.lower()}_{i}' for i in range(len(raw[f'Profiles {native}']))],
              **{col:raw[f'{prefix} Labels {native}'+suffix] for col,prefix,suffix in [
                  ('score_blind','Blind',''),('score_biased_gender','Biased',' (Gender)'),
                  ('score_biased_ethnicity','Biased',' (Ethnicity)')]}}))
        lab=pd.concat(labs,ignore_index=True)
        f=cf[cf.source_dataset.astype(str).str.contains('faircv',case=False)].copy()
        f=f.drop(columns=list(TARGETS),errors='ignore').merge(lab,on='original_id',how='left',validate='m:1')
        if f[list(TARGETS)].isna().any().any(): raise ValueError('Unmatched FairCV original_id labels')
        native_train=f.loc[f.original_id.str.startswith('faircv_train_'),'original_id'].drop_duplicates().to_numpy()
        if len(native_train)!=19200: raise ValueError(f'Native FairCV train has {len(native_train)} profiles, expected 19200')
        val=set(np.random.RandomState(42).choice(native_train,size=1920,replace=False))
        ids=f[['resume_id','original_id']].copy()
        ids['split']=np.where(ids.original_id.str.startswith('faircv_test_'),'test',
                        np.where(ids.original_id.isin(val),'val','train'))
        from baseline_scorer import ensure_split
        k=cf[cf.source_dataset.astype(str).str.contains('kaggle',case=False)].copy().drop(columns=['split'],errors='ignore')
        k,_=ensure_split(k,k.category_label,seed=42)
        splits=pd.concat([k[['resume_id','split']],ids[['resume_id','split']]],ignore_index=True)
        if splits.resume_id.duplicated().any(): raise ValueError('Duplicate resume_id in split map')
        stage_save(splits,split_path)
        train_profile=f[f.original_id.str.startswith('faircv_train_')].drop_duplicates('original_id')
        # Follow the revised baseline notebook: thresholds after carving the validation set.
        train_profile=train_profile[~train_profile.original_id.isin(val)]
        thr={t:float(pd.to_numeric(train_profile[t]).median()) for t in TARGETS}
        paths['prepare'].write_text(json.dumps(thr,indent=2))
        LOG.info('FairCV thresholds: %s; split profiles: %s',thr,ids.groupby('split').original_id.nunique().to_dict())
        del cf,f,k,lab,labs
    all_df,thr=load_prepared(str(root)+os.sep,apply_scrub=True)
    train=all_df[all_df.split=='train']; val=all_df[all_df.split=='val'];test=all_df[all_df.split=='test']
    src=test.source_dataset.astype(str).str.lower()
    if needed('baseline',proc/'baseline_predictions_scrubbed.csv'):
        outputs=[]
        tk=test[src.str.contains('kaggle',regex=False)]
        model=BaselineScorer();model.fit(train)
        o=model.predict_kaggle(tk);save_baseline(model,artifacts/'baseline'/'kaggle','kaggle');o.insert(0,'dataset','kaggle');o.insert(1,'target','category_label');outputs.append(o)
        del model;_gpu_cleanup()
        for t in TARGETS:
            sc=BaselineScorer(faircv_threshold=thr[t])
            tr=train.assign(suitability_score=train[t]);te=test[src.str.contains('faircv',regex=False)].assign(suitability_score=lambda d:d[t])
            sc.fit(tr);o=sc.predict_faircv(te)
            save_baseline(sc,artifacts/'baseline'/t,'faircv')
            o.insert(0,'dataset','faircv');o.insert(1,'target',t);outputs.append(o)
            if abs(sc.faircv_threshold_-thr[t])>1e-10:raise AssertionError('Baseline threshold changed')
            del sc;_gpu_cleanup()
        bp=pd.concat(outputs,ignore_index=True)
        stage_save(bp,paths['baseline']);stage_save(bp,proc/'baseline_predictions_scrubbed.csv')
    lock=proc/'improved_predictions_kaggle_locked.csv'
    if not lock.is_file(): raise FileNotFoundError(f'Locked Kaggle file missing: {lock}. Refusing to invent or regenerate it.')
    pk=_read(lock)
    # Keep the immutable locked Kaggle schema; align the other two files to it.
    bp_file=_read(paths['baseline'])
    if 'target' in pk.columns and len(pk):
        kaggle_target=pk['target'].iloc[0]
        bp_file.loc[bp_file.dataset=='kaggle','target']=kaggle_target
        if not bp_file.equals(_read(paths['baseline'])):
            for filename in ('baseline_predictions.csv','baseline_predictions_scrubbed.csv'):
                stage_save(bp_file,proc/filename)
    if set(pk.dataset.astype(str).str.lower())!={'kaggle'}:raise ValueError('Locked file must contain Kaggle rows only')
    refcols=None
    if paths['improved'].is_file():refcols=list(_read(paths['improved']).columns)
    if needed('improved'):
        tdf=test[src.str.contains('faircv',regex=False)]
        pj=None
        if joint_test is not None:
            candidate=joint_test[0] if isinstance(joint_test,(tuple,list)) else joint_test
            if isinstance(candidate,pd.DataFrame) and len(candidate)==len(tdf)*len(TARGETS):
                pj=candidate.copy();LOG.info('Reusing supplied in-memory joint_test FairCV predictions')
        if pj is None:
            sc=ImprovedScorer(weight_mode='parity',faircv_thresholds=thr,**CONFIG)
            sc.fit(train,val_df=val)
            pj=sc.predict_faircv(tdf)
            save_neural(sc,artifacts/'improved',dict(CONFIG,faircv_thresholds=thr),'improved')
            del sc;_gpu_cleanup()
        if 'dataset' in pj.columns: pj=pj.drop(columns=['dataset'])
        pj['dataset']='faircv'
        combined=pd.concat([pk,pj],ignore_index=True)
        if refcols is not None and set(combined.columns)==set(refcols):combined=combined[refcols]
        stage_save(combined,paths['improved'])
    if not (artifacts/'improved'/'manifest.json').exists():
        LOG.warning('Improved predictions exist but trained model artifact is absent. Supply the fitted improved scorer or force_stages=["improved"] to train and save it. In-memory joint_test predictions alone cannot yield the weights.')
    improved=_read(paths['improved'])
    locked_k=improved[improved.dataset=='kaggle'].reset_index(drop=True)
    if not locked_k.equals(pk[locked_k.columns].reset_index(drop=True)):
        raise AssertionError('Improved Kaggle rows are not identical to locked predictions')
    # Config audit runs EVEN when improved and/or control predictions were skipped.
    common=dict(CONFIG,faircv_thresholds=thr)
    matched_improved,control=build_matched_pair(common,'parity',engine_factory=lambda:EmbeddingEngine())
    assert not signature_drift(),signature_drift()
    assert_matched(matched_improved,control)
    if control.weight_mode!='idf':raise AssertionError('Control must use IDF')
    if needed('bertidraaf_lite',proc/'bertidraaf_lite_fit_metadata.json'):
        # Avoid holding BOTH neural scorers on GPU while training: unfit improved has no loaded model.
        del matched_improved;_gpu_cleanup()
        control.fit(train,val_df=val)
        pred=_assemble(control,test,refcols or improved.columns)
        save_neural(control,artifacts/'bertidraaf_lite',common,'bertidraaf_lite')
        locked_target=pk['target'].iloc[0] if 'target' in pk.columns else np.nan
        pred.loc[pred.dataset=='kaggle','target']=locked_target
        stage_save(pred,paths['bertidraaf_lite'])
        calc=control.ibs_calc
        LOG.info('IDF: N=%s oov_weight=%s token_median_idf=%s',calc.n_docs_,control.oov_weight,calc.token_median_idf_)
        evidence={'n_docs':int(calc.n_docs_),'oov_weight':float(control.oov_weight),
                  'token_median_idf':float(calc.token_median_idf_),
                  'oov_explicit':bool(control._oov_explicit),'weight_mode':control.weight_mode}
        (proc/'bertidraaf_lite_fit_metadata.json').write_text(json.dumps(evidence,indent=2))
        del control;_gpu_cleanup()
    else:
        del matched_improved,control;_gpu_cleanup()
    required_artifacts=[artifacts/'baseline'/'kaggle'/'model.joblib',
       *(artifacts/'baseline'/t/'model.joblib' for t in TARGETS),
       artifacts/'improved'/'manifest.json',artifacts/'improved'/'weights.pt',
       artifacts/'bertidraaf_lite'/'manifest.json',artifacts/'bertidraaf_lite'/'weights.pt']
    for missing in required_artifacts:
        if not missing.is_file(): LOG.warning('Application artifact missing: %s; force that scorer stage to regenerate it',missing)
    models={name:_read(paths[name]) for name in ('baseline','bertidraaf_lite','improved')}
    if needed('fairness',paired_path,reports/'fairness_comparison.csv'):
        auditor=FairnessAuditor(n_iterations=n_iterations,confidence=.90,seed=42)
        auditor.validate_same_test_rows(models)
        native=test[['resume_id','original_id','gender_label','ethnicity_label']].drop_duplicates('resume_id')
        cols=['intersectional_group','gender_label','ethnicity_label']
        comp=auditor.compare(models,group_cols=cols,native_labels_df=native)
        paired=auditor.pairwise_comparison(models,pairs=PAIRS,group_cols=cols,native_labels_df=native,comparison_df=comp)
        stage_save(comp,paths['fairness']);stage_save(comp,reports/'fairness_comparison.csv');stage_save(paired,paired_path)
    if needed('report'):
        ReportDashboard().export_html_report(_read(paths['fairness']),str(paths['report']),_read(paired_path))
    def normalized_kaggle_target(d):
        d=d.copy(); d.loc[d.dataset=='kaggle','target']='category_label'; return d
    for a,b in PAIRS:
        LOG.info('FINAL ACCURACY: %s vs %s',a,b)
        for ds,tgt in [('kaggle','category_label'),* [('faircv',t) for t in TARGETS],('faircv',None)]:
            def subset(d):
                x=d[d.dataset==ds]
                return x if tgt is None else x[x.target==tgt]
            left,right=subset(normalized_kaggle_target(models[a])),subset(normalized_kaggle_target(models[b]))
            d=paired_delta(left,right,key=('resume_id','target'))
            reference=0.8517 if ds=='faircv' else None
            p=placement(d,reference) if reference is not None else 'paper benchmark not supplied'
            LOG.info(' %s/%s: %s %.4f, %s %.4f; paired_delta=%+.4f SE=%.4f; verdict=%s; placement=%s',
                     ds,tgt or 'pooled',a,d['baseline'],b,d['improved'],d['delta'],d['se'],verdict(d),p)
    LOG.info('Pipeline complete: %s',paths['report'])
    return {'predictions':models,'comparison':_read(paths['fairness']),'paired':_read(paired_path),
            'common_kwargs':common,'thresholds':thr,'prepared':all_df,'root':root,'artifacts_dir':artifacts}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='/content/drive/MyDrive/bias_detection_project/')
    ap.add_argument('--force',action='store_true');ap.add_argument('--force-stages',nargs='*',default=[])
    args=ap.parse_args();run_pipeline(args.root,force=args.force,force_stages=args.force_stages)
