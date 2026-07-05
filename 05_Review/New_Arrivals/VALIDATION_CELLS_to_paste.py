# =====================================================================
# TrustXAI-Derm — SELF-CHECKING VALIDATION CELLS
# Paste each block into a NEW notebook cell at the indicated position on
# Kaggle. These print real pass/fail from real outputs. They do NOT
# fabricate anything — if a gate fails, it says so and keeps going so you
# can see every metric, rather than crashing on the first assert.
# =====================================================================


# ---------------------------------------------------------------------
# GATE A  —  paste immediately AFTER Cell 7 (DenseNet121 training)
# ---------------------------------------------------------------------
import pandas as pd, json
_hist = pd.read_csv(CFG['out_dir'] / 'DenseNet121_history.csv')
_final_epoch = int(_hist['epoch'].max())
_row = _hist[_hist['epoch'] == _final_epoch].iloc[0]
_final_acc = float(_row['train_acc'])
_final_f1  = float(_row['val_f1'])
_final_bacc = float(_row['val_bacc']) if 'val_bacc' in _hist.columns else float('nan')
_status = {}
_sp = CFG['out_dir'] / 'DenseNet121_status.json'
if _sp.exists():
    _status = json.load(open(_sp))

print('=== GATE A: DenseNet121 training completion ===')
print(f'  reached epoch      : {_final_epoch} / {CFG["epochs"]}   (stop reason: {_status.get("reason","?")})')
print(f'  final train_acc    : {_final_acc:.4f}   (gate wants >= 0.75)')
print(f'  best/last val_f1   : {_final_f1:.4f}   (gate wants >= 0.55)')
print(f'  final val_bacc     : {_final_bacc:.4f}')
_gateA = (_final_epoch >= 150 and _final_acc >= 0.75 and _final_f1 >= 0.55)
print(f'  --> GATE A: {"PASS" if _gateA else "NOT MET"}')
if not _gateA:
    print('  NOTE: a NOT-MET gate here is not automatically a bug. Check the training')
    print('  curve: if val_f1 plateaued (early_stopped) this may be the honest ceiling')
    print('  for this architecture+data. Only treat it as a bug if val_f1 was still')
    print('  climbing when it stopped, or the LR never annealed (inspect the LR schedule).')


# ---------------------------------------------------------------------
# GATE B  —  paste immediately AFTER Cell 9 (temperature scaling + eval)
# ---------------------------------------------------------------------
print('=== GATE B: calibration + test evaluation ===')
print(f'  T_dense            : {T_dense}')
print(f'  convnext_T         : {convnext_T}')
print(f'  DenseNet test acc  : {dense_results["accuracy"]:.4f}   (gate wants >= 0.72)')
print(f'  DenseNet test AUC  : {dense_results["macro_auc"]:.4f}   (gate wants >= 0.78)')
print(f'  DenseNet macro F1  : {dense_results["macro_f1"]:.4f}')
print(f'  DenseNet ECE       : {dense_results["ece"]:.4f}')
print(f'  ConvNeXt test acc  : {convnext_results["accuracy"]:.4f}')
print(f'  ConvNeXt test AUC  : {convnext_results["macro_auc"]:.4f}')
_gateB = (T_dense is not None and convnext_T is not None
          and dense_results['accuracy'] >= 0.72 and dense_results['macro_auc'] >= 0.78)
print(f'  --> GATE B: {"PASS" if _gateB else "NOT MET"}')


# ---------------------------------------------------------------------
# RED-FLAG SCAN  —  paste near the END, after Cell 17b and Cell 15/15b/18
# have all run. Reads the computed dataframes/vars, not the raw log.
# ---------------------------------------------------------------------
print('=== RED-FLAG SCAN (these should all read CLEAR after the fixes) ===')

# 1. ConvNeXt Grad-CAM hook was broken -> TIxAI ~0.0006. After the hook fix it
#    should be a plausible value. tixai_cn_df is built in Cell 16.
try:
    _cn_med = float(tixai_cn_df['tixai'].median())
    print(f'  ConvNeXt TIxAI median : {_cn_med:.4f}   -> {"CLEAR" if _cn_med >= 0.05 else "STILL BROKEN (hook?)"}')
except Exception as e:
    print(f'  ConvNeXt TIxAI median : (tixai_cn_df not available: {e})')

# 2. CCRS circularity -> deferred error rate was exactly 1.0000. Should not be.
try:
    _defer_err = 1 - ccrs_df[ccrs_df['deferred']]['correct'].mean()
    print(f'  Deferred error rate   : {_defer_err:.4f}   -> {"CLEAR" if _defer_err < 0.999 else "STILL 1.0 (leak?)"}')
except Exception as e:
    print(f'  Deferred error rate   : (ccrs_df not available: {e})')

# 3. CCRS imputed-TIxAI share (Problem 10). Should be near 0 now (only genuine failures).
try:
    if 'tixai_imputed' in ccrs_df.columns:
        _imp = 100 * ccrs_df['tixai_imputed'].mean()
        print(f'  CCRS TIxAI imputed %  : {_imp:.1f}%   -> {"CLEAR" if _imp < 30 else "HIGH"}')
except Exception:
    pass

# 4. Cell 15b conformal coverage should now print real per-bucket numbers.
try:
    print(f'  Per-FST coverage rows : {len(fst_coverage_results)}   -> {"CLEAR (computed)" if len(fst_coverage_results) > 0 else "EMPTY (still dead?)"}')
except Exception:
    print('  Per-FST coverage rows : (fst_coverage_results not defined -- may have been skipped legitimately)')

# 5. Executive-summary vars: p_xdi and rho_ercc should exist and match their cells.
print(f'  p_xdi defined         : {"yes" if "p_xdi" in dir() else "NO"}  '
      f'(XDI correct-vs-incorrect p; expect ~0.9+ i.e. null)')
print(f'  rho_ercc defined      : {"yes" if "rho_ercc" in dir() else "NO"}  '
      f'(ERCC Spearman rho)')


# ---------------------------------------------------------------------
# METRICS TABLE  —  paste as the LAST cell, after Cell 18 (exec summary).
# Writes METRICS_TABLE.csv with actual values + the prompt's expected ranges.
# Any var that does not exist is recorded as 'NOT COMPUTED' rather than crashing.
# ---------------------------------------------------------------------
def _g(expr):
    try:
        return eval(expr)
    except Exception:
        return 'NOT COMPUTED'

_rows = [
    ('DenseNet121 test accuracy',            '>=0.70',            _g("round(dense_results['accuracy'],4)")),
    ('DenseNet121 test macro AUC',           '>=0.78',            _g("round(dense_results['macro_auc'],4)")),
    ('DenseNet121 test macro F1',            '>=0.55',            _g("round(dense_results['macro_f1'],4)")),
    ('ConvNeXt-Tiny test accuracy',          '>=0.65',            _g("round(convnext_results['accuracy'],4)")),
    ('ConvNeXt-Tiny test macro AUC',         '>=0.75',            _g("round(convnext_results['macro_auc'],4)")),
    ('GAP-1 TIxAI Kruskal-Wallis p',         '<0.01 significant', _g("float(f'{p_kruskal:.2e}')")),
    ('GAP-2 failure-pred AUC-ROC',           '0.80-0.85',         _g("round(auc_meta,4)")),
    ('GAP-4 XDI Mann-Whitney p',             '>0.05 null OK',     _g("round(float(p_xdi),4)")),
    ('GAP-6 STS median',                     '0.80-0.90',         _g("round(float(sts_df['sts'].median()),4)")),
    ('GAP-7 FST TIxAI p',                    '>0.05 null OK',     _g("round(float(p_fst),4)")),
    ('GAP-7 FST entropy p',                  '<0.01 significant', _g("round(float(p_ent),4)")),
    ('GAP-8 deferred error rate',            '<0.50 (not 1.0)',   _g("round(1 - ccrs_df[ccrs_df['deferred']]['correct'].mean(),4)")),
    # NOTE: GAP-10's Spearman is stored in the notebook as `rho, p_spear = spearmanr(...)`.
    # `p_spear` is assigned ONLY in GAP-10, so its presence uniquely confirms GAP-10 ran;
    # the rho value sits in the bare `rho` (last spearmanr in the notebook), which is why
    # this metrics cell must be the LAST cell executed. If GAP-10 was skipped (DINOv2
    # unavailable), `p_spear` is undefined and this correctly reads NOT COMPUTED.
    ('GAP-10 contrastive Spearman rho',      '~0.02 null OK',     _g("round(float(rho),4) if 'p_spear' in dir() else 'NOT COMPUTED'")),
    ('GAP-10 contrastive Spearman p',        '>0.05 null OK',     _g("round(float(p_spear),4) if 'p_spear' in dir() else 'NOT COMPUTED'")),
    ('ConvNeXt TIxAI median (hook check)',   '>=0.05 not 0.0006', _g("round(float(tixai_cn_df['tixai'].median()),4)")),
]
_mt = pd.DataFrame(_rows, columns=['Metric', 'Expected', 'Actual'])
_out = CFG['out_dir'] / 'METRICS_TABLE.csv'
_mt.to_csv(_out, index=False)
print('\n=== METRICS TABLE (also saved to METRICS_TABLE.csv) ===')
print(_mt.to_string(index=False))
print(f'\nSaved: {_out}')
print('\nFill these ACTUAL numbers into the manuscript. Any row showing '
      '"NOT COMPUTED" means that cell was skipped or errored -- investigate before citing it.')
