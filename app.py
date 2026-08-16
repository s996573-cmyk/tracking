import streamlit as st

# 1. 必須置於最頂端，防止網頁渲染白屏
st.set_page_config(page_title="學生學業表現與 DSE 升學分析系統", layout="wide")

import pandas as pd
import numpy as np
import os

# 安全引入 scikit-learn
try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# 安全數值求平均函數（自動剔除文字標籤如 Pass/Attained，防止 TypeError）
def safe_mean(series):
    if series is None or len(series) == 0:
        return np.nan
    s_num = pd.to_numeric(series, errors='coerce')
    if s_num.dropna().empty:
        return np.nan
    val = s_num.mean()
    return round(float(val), 1) if not pd.isna(val) else np.nan

# 安全讀取 CSV 與 Excel (支援檔案路徑字串及 UploadedFile 物件)
def safe_read_file(file_input):
    if isinstance(file_input, str):
        filename = file_input.lower()
        if filename.endswith('.csv'):
            encodings = ['utf-8-sig', 'big5-hkscs', 'big5', 'cp950', 'gb18030', 'gbk', 'utf-8']
            for enc in encodings:
                try:
                    df = pd.read_csv(file_input, encoding=enc)
                    if not df.empty: return df
                except Exception:
                    continue
            return pd.read_csv(file_input, encoding_errors='ignore')
        else:
            return pd.read_excel(file_input)
    else:
        filename = getattr(file_input, 'name', '').lower()
        if filename.endswith('.csv'):
            encodings = ['utf-8-sig', 'big5-hkscs', 'big5', 'cp950', 'gb18030', 'gbk', 'utf-8']
            for enc in encodings:
                try:
                    if hasattr(file_input, 'seek'): file_input.seek(0)
                    df = pd.read_csv(file_input, encoding=enc)
                    if not df.empty: return df
                except Exception:
                    continue
            if hasattr(file_input, 'seek'): file_input.seek(0)
            return pd.read_csv(file_input, encoding_errors='ignore')
        else:
            if hasattr(file_input, 'seek'): file_input.seek(0)
            return pd.read_excel(file_input)

# 自動尋找預設系統檔案或使用手動上載檔案
def resolve_file_source(uploaded_file, candidate_filenames):
    if uploaded_file is not None:
        return uploaded_file, f"📤 已選用上載檔案：`{uploaded_file.name}`"
    for fn in candidate_filenames:
        if os.path.exists(fn):
            return fn, f"📁 已自動匯入系統檔案：`{fn}`"
    return None, "⚠️ 找不到預設檔案，請手動上載。"

# 標準學號清理函數：專門處理 *Reg. No. 與 Registration No.
def clean_id(v):
    if pd.isna(v): return ''
    s = str(v).strip()
    if s.startswith('#'): s = s[1:]
    if s.endswith('.0'): s = s[:-2]
    return s.strip()

# 健壯的 DSE 成績等級轉數值函數
def parse_dse_grade(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().upper()
    if s in ['N.T.', 'N.A.', 'ABS', 'X', '', 'NAN']: return np.nan
    if '5**' in s: return 7
    if '5*' in s: return 6
    exact_map = {
        '5': 5, '5.0': 5, 'LEVEL 5': 5, 'LV 5': 5, 'LV5': 5, '5級': 5, '第5級': 5,
        '4': 4, '4.0': 4, 'LEVEL 4': 4, 'LV 4': 4, 'LV4': 4, '4級': 4, '第4級': 4,
        '3': 3, '3.0': 3, 'LEVEL 3': 3, 'LV 3': 3, 'LV3': 3, '3級': 3, '第3級': 3,
        '2': 2, '2.0': 2, 'LEVEL 2': 2, 'LV 2': 2, 'LV2': 2, '2級': 2, '第2級': 2,
        '1': 1, '1.0': 1, 'LEVEL 1': 1, 'LV 1': 1, 'LV1': 1, '1級': 1, '第1級': 1,
        'U': 0, '0': 0, '0.0': 0, 'UNCLASSIFIED': 0, 'UNATTAINED': 0, '不達標': 0,
        'A': 3, 'B': 3, 'C': 3, 'D': 3, 'E': 3, 'PASS': 3, 'ATTAINED': 3, '達標': 3
    }
    if s in exact_map: return exact_map[s]
    try:
        v = float(s)
        if 0 <= v <= 7: return v
    except: pass
    return np.nan

# 欄位搜尋輔助函數
def find_col(columns, inc_keys, exc_keys):
    candidates = []
    for c in columns:
        c_str = str(c).upper()
        if any(k.upper() in c_str for k in inc_keys):
            if not any(ex.upper() in c_str for ex in exc_keys):
                candidates.append(c)
    if candidates:
        candidates.sort(key=lambda x: len(str(x)))
        return candidates[0]
    return None

# 精準鎖定 DSE 科目總成績欄位 (自動排除姓名及分卷)
def find_best_dse_subject_col(columns, subject_keywords, exclude_keywords=None):
    if exclude_keywords is None: exclude_keywords = []
    default_excludes = ['NAME', '姓名', 'READING', 'WRITING', 'LISTENING', 'SPEAKING', 'PAPER', '卷', '閱讀', '寫作', '聆聽', '說話', '口試', 'SCORE', 'MARK', '分數', '分值', 'COMPONENT', 'PART', 'INTEGRATED', 'HISTORY', '歷史']
    all_excludes = set([k.upper() for k in (exclude_keywords + default_excludes)])
    return find_col(columns, subject_keywords, list(all_excludes))

# 所有學科配置表（包含核心科與選修科）
SUBJECT_MAP = [
    {
        'name': '中文科',
        'sch_keys': ['中文'],
        'dse_keys': ['A010', 'CHINESE', '中文'],
        'sch_ex': ['閱讀', '寫作', '聆聽', '口試', '說話', '中史'],
        'dse_ex': ['NAME', '姓名', 'READING', 'WRITING', 'LISTENING', 'SPEAKING', 'PAPER', '卷', '閱讀', '寫作', '聆聽', '說話', '口試', 'INTEGRATED', 'HISTORY', '歷史']
    },
    {
        'name': '英文科',
        'sch_keys': ['英文'],
        'dse_keys': ['A020', 'ENGLISH', '英文'],
        'sch_ex': ['閱讀', '作文', '聆聽', '口試', '說話', '語文'],
        'dse_ex': ['NAME', '姓名', 'READING', 'WRITING', 'LISTENING', 'SPEAKING', 'PAPER', '卷', '閱讀', '寫作', '聆聽', '說話', '口試', 'INTEGRATED']
    },
    {
        'name': '數學必修',
        'sch_keys': ['數必', '數學'],
        'dse_keys': ['A030', 'MATH COMPULSORY', '數學必修'],
        'sch_ex': ['數一', '數二', 'M1', 'M2', 'EXTENDED', '單元'],
        'dse_ex': ['M1', 'M2', '數一', '數二', 'EXTENDED', '單元']
    },
    {
        'name': '數學延伸 (M1/M2)',
        'sch_keys': ['數一', '數二', 'M1', 'M2'],
        'dse_keys': ['A031', 'A032', 'M1', 'M2'],
        'sch_ex': [],
        'dse_ex': []
    },
    {
        'name': '中國歷史',
        'sch_keys': ['中史'],
        'dse_keys': ['A070', 'CHINESE HISTORY', '中史'],
        'sch_ex': [],
        'dse_ex': []
    },
    {
        'name': '經濟科',
        'sch_keys': ['經濟'],
        'dse_keys': ['A080', 'ECONOMICS', '經濟'],
        'sch_ex': [],
        'dse_ex': []
    },
    {
        'name': '倫理與宗教',
        'sch_keys': ['倫宗', '倫教'],
        'dse_keys': ['A090', 'ETHICS', 'RELIGIOUS', '倫理', '宗教'],
        'sch_ex': [],
        'dse_ex': []
    },
    {
        'name': '地理科',
        'sch_keys': ['地理'],
        'dse_keys': ['A100', 'GEOGRAPHY', '地理'],
        'sch_ex': [],
        'dse_ex': []
    },
    {
        'name': '歷史科',
        'sch_keys': ['歷史'],
        'dse_keys': ['A110', 'HISTORY', '歷史'],
        'sch_ex': ['中史'],
        'dse_ex': ['CHINESE HISTORY', '中史']
    },
    {
        'name': '生物科',
        'sch_keys': ['生物'],
        'dse_keys': ['A130', 'BIOLOGY', '生物'],
        'sch_ex': [],
        'dse_ex': []
    },
    {
        'name': '化學科',
        'sch_keys': ['化學'],
        'dse_keys': ['A140', 'CHEMISTRY', '化學'],
        'sch_ex': [],
        'dse_ex': []
    },
    {
        'name': '物理科',
        'sch_keys': ['物理'],
        'dse_keys': ['A150', 'PHYSICS', '物理'],
        'sch_ex': [],
        'dse_ex': []
    },
    {
        'name': '企業、會計與財務概論 (BAFS)',
        'sch_keys': ['企財', 'BAFS'],
        'dse_keys': ['A171', 'A172', 'BAFS', '企財', '企業'],
        'sch_ex': [],
        'dse_ex': []
    },
    {
        'name': '資訊及通訊科技 (ICT)',
        'sch_keys': ['資通', 'ICT'],
        'dse_keys': ['A200', 'ICT', '資訊'],
        'sch_ex': [],
        'dse_ex': []
    },
    {
        'name': '視覺藝術',
        'sch_keys': ['視憑', '視藝'],
        'dse_keys': ['A230', 'VISUAL ARTS', '視藝', '視覺藝術'],
        'sch_ex': [],
        'dse_ex': []
    }
]

# 自動整合校內試（含中英文組別 C_Score / E_Score）分數
def get_school_subject_score(df, subject_keys, exclude_keys):
    matched_cols = []
    for c in df.columns:
        c_str = str(c).upper()
        if 'SCORE' in c_str and any(k.upper() in c_str for k in subject_keys):
            if not any(ex.upper() in c_str for ex in exclude_keys):
                matched_cols.append(c)
                
    if not matched_cols:
        return None
        
    combined_series = None
    for col in matched_cols:
        s = pd.to_numeric(df[col], errors='coerce')
        combined_series = s if combined_series is None else combined_series.fillna(s)
        
    return combined_series

# 全學科（主科 + 各選修科）相關係數分析表產生器
def compute_all_subjects_correlation(d_sch, d_dse):
    sch_reg_cols = [c for c in d_sch.columns if '*Reg. No.' in str(c) or 'Reg' in str(c) or '註冊編號' in str(c) or '學號' in str(c)]
    sch_reg = sch_reg_cols[0] if sch_reg_cols else d_sch.columns[0]
    d_sch['Reg_Clean'] = d_sch[sch_reg].apply(clean_id)
    
    dse_reg_cols = [c for c in d_dse.columns if 'Registration No' in str(c) or 'Reg' in str(c) or '註冊編號' in str(c) or '學號' in str(c)]
    dse_reg = dse_reg_cols[0] if dse_reg_cols else d_dse.columns[0]
    d_dse['Reg_Clean'] = d_dse[dse_reg].apply(clean_id)
    
    m_pair = pd.merge(d_sch, d_dse, on='Reg_Clean')
    
    results = []
    for sub in SUBJECT_MAP:
        sch_s = get_school_subject_score(m_pair, sub['sch_keys'], sub['sch_ex'])
        dse_col = find_col(d_dse.columns, sub['dse_keys'], sub['dse_ex'])
        
        if sch_s is not None and dse_col is not None:
            dse_g = m_pair[dse_col].apply(parse_dse_grade)
            valid_mask = sch_s.notna() & dse_g.notna()
            n_students = valid_mask.sum()
            
            if n_students >= 3:
                r_val = sch_s[valid_mask].corr(dse_g[valid_mask])
                if not pd.isna(r_val):
                    abs_r = abs(r_val)
                    if abs_r >= 0.8:
                        strength = "🔴 極高關聯 (試卷對照度極佳)"
                    elif abs_r >= 0.6:
                        strength = "🟠 高度關聯 (試卷對照度良好)"
                    elif abs_r >= 0.4:
                        strength = "🟡 中度關聯 (建議檢視深淺度)"
                    elif abs_r >= 0.2:
                        strength = "🟢 低度關聯 (需優化考核指標)"
                    else:
                        strength = "⚪ 極低關聯 (擬題方向待調整)"
                        
                    results.append({
                        '學科名稱': sub['name'],
                        '修讀對照人數': int(n_students),
                        '校內分數 vs DSE 等級 相關係數 (r)': round(r_val, 3),
                        '試卷效度與關聯評語': strength
                    })
                    
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(by='校內分數 vs DSE 等級 相關係數 (r)', ascending=False)
    return df_res

# 健壯的核心科目校內成績自動提取函數
def extract_robust_scores(df):
    df_out = df.copy()
    
    tot_cols = [c for c in ['T2A3_Score', 'T1A3_Score', 'T2A1_Score', 'T1A1_Score', 'Score'] if c in df_out.columns]
    if not tot_cols:
        tot_cols = [c for c in df_out.columns if str(c).endswith('_Score') and not any(sub in str(c) for sub in ['生物', '化學', '中文', '英文', '數學', '數必', '公民', '企財', '經濟', '地理', '歷史', '中史', '物理', '資通', '視憑', '倫教', '體育', '學培課', '退修課'])]
    tot_col = tot_cols[0] if tot_cols else None
    
    avg_score = pd.to_numeric(df_out[tot_col], errors='coerce').round(1) if tot_col else pd.Series(np.nan, index=df_out.index)

    chi_series = get_school_subject_score(df_out, ['中文'], ['閱讀', '寫作', '聆聽', '口試', '說話', '中史'])
    eng_series = get_school_subject_score(df_out, ['英文'], ['閱讀', '作文', '聆聽', '口試', '語文'])
    math_series = get_school_subject_score(df_out, ['數必', '數學'], ['數一', '數二', 'M1', 'M2', 'EXTENDED', '單元'])

    cs_cols = [c for c in df_out.columns if any(k in str(c) for k in ['公民科', 'CS']) and any(k in str(c) for k in ['Score', 'Grade'])]
    cs_series = None
    for csc in cs_cols:
        s = pd.to_numeric(df_out[csc], errors='coerce')
        if s.notna().sum() == 0:
            s = df_out[csc]
        cs_series = s if cs_series is None else cs_series.fillna(s)

    df_out['Avg_Score'] = avg_score
    df_out['Chi_Score'] = chi_series.round(1) if chi_series is not None else np.nan
    df_out['Eng_Score'] = eng_series.round(1) if eng_series is not None else np.nan
    df_out['Math_Score'] = math_series.round(1) if math_series is not None else np.nan
    df_out['CS_Val'] = cs_series if cs_series is not None else np.nan
    return df_out

# 通用過往畢業生數據對照與清洗處理
def process_training_pair(f_sch, f_dse, dataset_label="2526"):
    d_sch = safe_read_file(f_sch)
    d_dse = safe_read_file(f_dse)
    
    sch_reg_cols = [c for c in d_sch.columns if '*Reg. No.' in str(c) or 'Reg' in str(c) or '註冊編號' in str(c) or '學號' in str(c)]
    sch_reg = sch_reg_cols[0] if sch_reg_cols else d_sch.columns[0]
    d_sch['Reg_Clean'] = d_sch[sch_reg].apply(clean_id)
    
    dse_reg_cols = [c for c in d_dse.columns if 'Registration No' in str(c) or 'Reg' in str(c) or '註冊編號' in str(c) or '學號' in str(c)]
    dse_reg = dse_reg_cols[0] if dse_reg_cols else d_dse.columns[0]
    d_dse['Reg_Clean'] = d_dse[dse_reg].apply(clean_id)
    
    m_ai = pd.merge(d_sch, d_dse, on='Reg_Clean')
    if len(m_ai) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"⚠️ 對照失敗：無法透過校內【{sch_reg}】與 DSE【{dse_reg}】匹配學生身份，請確認兩檔學號是否相符。"
        
    m_ai = extract_robust_scores(m_ai)
    
    chi_col = find_best_dse_subject_col(d_dse.columns, ['A010', 'CHINESE', '中文'])
    eng_col = find_best_dse_subject_col(d_dse.columns, ['A020', 'ENGLISH', '英文'])
    math_col = find_best_dse_subject_col(d_dse.columns, ['A030', 'MATH', '數學'], ['M1', 'M2', '數一', '數二', 'EXTENDED', '單元'])
    
    if not (chi_col and eng_col and math_col):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "⚠️ DSE 核心科目欄位辨識失敗：DSE 表格需包含中文、英文及數學必修部分成績。"
        
    m_ai['DSE_Chi_Lvl'] = m_ai[chi_col].apply(parse_dse_grade)
    m_ai['DSE_Eng_Lvl'] = m_ai[eng_col].apply(parse_dse_grade)
    m_ai['DSE_Math_Lvl'] = m_ai[math_col].apply(parse_dse_grade)
    
    m_ai['DSE_Chi_Raw'] = m_ai[chi_col]
    m_ai['DSE_Eng_Raw'] = m_ai[eng_col]
    m_ai['DSE_Math_Raw'] = m_ai[math_col]
    
    m_ai['Target_332'] = ((m_ai['DSE_Chi_Lvl'] >= 3) & (m_ai['DSE_Eng_Lvl'] >= 3) & (m_ai['DSE_Math_Lvl'] >= 2)).astype(int)
    m_ai['Target_222'] = ((m_ai['DSE_Chi_Lvl'] >= 2) & (m_ai['DSE_Eng_Lvl'] >= 2) & (m_ai['DSE_Math_Lvl'] >= 2)).astype(int)
    
    def get_tier(row):
        if row['Target_332'] == 1:
            return "🎓 大學收生門檻 (332)"
        elif row['Target_222'] == 1:
            return "🏫 大專收生門檻 (222)"
        else:
            return "未達 222 門檻"

    m_ai['達標類別'] = m_ai.apply(get_tier, axis=1)
    
    feats = ['Avg_Score', 'Chi_Score', 'Eng_Score', 'Math_Score']
    clean_df = m_ai.dropna(subset=feats + ['Target_332'])[feats + ['Target_332', 'Target_222', 'DSE_Chi_Lvl', 'DSE_Eng_Lvl', 'DSE_Math_Lvl']]
    
    df_all_corr = compute_all_subjects_correlation(d_sch, d_dse)
    
    # 提取達大學或大專收生門檻之學生成績對照表
    comp_df = m_ai[m_ai['Target_222'] == 1].copy()
    comp_df['學年'] = dataset_label
    
    # 自動匹配與對照所有選修科目（校內分數 + DSE 等級）
    for sub in SUBJECT_MAP[3:]: # 選修科目
        sub_name = sub['name']
        sch_s = get_school_subject_score(comp_df, sub['sch_keys'], sub['sch_ex'])
        dse_c = find_col(comp_df.columns, sub['dse_keys'], sub['dse_ex'])
        
        if sch_s is not None and sch_s.notna().sum() > 0:
            comp_df[f'校內{sub_name}'] = sch_s.round(1)
        if dse_c is not None and comp_df[dse_c].notna().sum() > 0:
            comp_df[f'DSE {sub_name}'] = comp_df[dse_c]

    cols_mapping = {
        '學年': '學年',
        '*Class': '班別', 'Class': '班別',
        '*Class Number': '班號', 'Class No.': '班號',
        '中文姓名': '中文姓名',
        'Name': '英文姓名', '*Student Name': '英文姓名',
        'Avg_Score': '校內總平均分',
        'Chi_Score': '校內中文分數',
        'Eng_Score': '校內英文分數',
        'Math_Score': '校內數學分數',
        'CS_Val': '校內公社科分數',
        'DSE_Chi_Raw': 'DSE 中文等級',
        'DSE_Eng_Raw': 'DSE 英文等級',
        'DSE_Math_Raw': 'DSE 數學等級',
        '達標類別': '達標類別'
    }
    
    display_cols = []
    # 基礎欄位定義
    base_order = ['學年', '班別', '班號', '中文姓名', '英文姓名', '校內總平均分', '校內中文分數', '校內英文分數', '校內數學分數', '校內公社科分數', 'DSE 中文等級', 'DSE 英文等級', 'DSE 數學等級']
    
    for c in base_order:
        found = None
        for orig, mapped in cols_mapping.items():
            if mapped == c and orig in comp_df.columns:
                found = orig
                break
        if found:
            comp_df[c] = comp_df[found]
            if c not in display_cols:
                display_cols.append(c)
                
    # 動態添加已出現的選修科對照欄位（校內分數與 DSE 等級）
    for sub in SUBJECT_MAP[3:]:
        sub_name = sub['name']
        sch_col_name = f'校內{sub_name}'
        dse_col_name = f'DSE {sub_name}'
        if sch_col_name in comp_df.columns:
            display_cols.append(sch_col_name)
        if dse_col_name in comp_df.columns:
            display_cols.append(dse_col_name)

    # 確保達標類別放在最後
    if '達標類別' in comp_df.columns and '達標類別' not in display_cols:
        display_cols.append('達標類別')

    res_comp_df = comp_df[display_cols]
    
    return clean_df, df_all_corr, res_comp_df, None

# 專門根據成功考獲 DSE 332 大學門檻學生提煉『下四分位數保底入場線 (Q1 基準)』
def extract_ai_thresholds(df_train, feats):
    successful_df = df_train[df_train['Target_332'] == 1]
    
    defaults = {
        'Avg_Score': 50.4,
        'Chi_Score': 49.8,
        'Eng_Score': 53.8,
        'Math_Score': 38.8
    }
    
    final_thresh = {}
    for f in feats:
        if not successful_df.empty and f in successful_df.columns:
            val = round(float(successful_df[f].quantile(0.25)), 1)
            final_thresh[f] = val
        else:
            final_thresh[f] = defaults[f]
            
    return final_thresh

# 載入學生中文姓名資料庫
@st.cache_data
def load_student_info():
    file_path = 'student_info.xlsx'
    if os.path.exists(file_path):
        try:
            df_info = safe_read_file(file_path)
            if '註冊編號' in df_info.columns and '中文姓名' in df_info.columns:
                df_info['註冊編號_clean'] = df_info['註冊編號'].apply(clean_id)
                return df_info[['註冊編號_clean', '中文姓名']].dropna()
        except Exception: pass
    return None

student_info_df = load_student_info()

st.title("📊 學生成績追蹤、數據建模與 DSE 升學分析系統")

# 初始化預設門檻 (Q1 基準：總平均分 50.4 分)
if 'u_score_val' not in st.session_state: st.session_state['u_score_val'] = 50.4
if 'u_chi_val' not in st.session_state: st.session_state['u_chi_val'] = 49.8
if 'u_eng_val' not in st.session_state: st.session_state['u_eng_val'] = 53.8
if 'u_math_val' not in st.session_state: st.session_state['u_math_val'] = 38.8
if 'u_cs_val' not in st.session_state: st.session_state['u_cs_val'] = 40.0

# 側邊欄門檻設定（為每個元件設定唯一的 key 以避免重複 ID 錯誤）
st.sidebar.header("⚙️ 大學門檻 (332A22) 校內分數設定")

u_score_thresh = st.sidebar.number_input("大學：總平均分門檻", value=float(st.session_state['u_score_val']), key="u_score_thresh_input")
u_chi_thresh = st.sidebar.number_input("大學：中文分數門檻", value=float(st.session_state['u_chi_val']), key="u_chi_thresh_input")
u_eng_thresh = st.sidebar.number_input("大學：英文分數門檻", value=float(st.session_state['u_eng_val']), key="u_eng_thresh_input")
u_math_thresh = st.sidebar.number_input("大學：數學分數門檻", value=float(st.session_state['u_math_val']), key="u_math_thresh_input")
u_cs_thresh = st.sidebar.number_input("大學：公社科分數門檻", value=float(st.session_state['u_cs_val']), key="u_cs_thresh_input")

st.sidebar.header("⚙️ 大專門檻 (222A22) 校內分數設定")
sub_score_thresh = st.sidebar.number_input("大專：總平均分門檻", value=40.0, key="sub_score_thresh_input")
sub_chi_thresh = st.sidebar.number_input("大專：中文分數門檻", value=40.0, key="sub_chi_thresh_input")
sub_eng_thresh = st.sidebar.number_input("大專：英文分數門檻", value=40.0, key="sub_eng_thresh_input")
sub_math_thresh = st.sidebar.number_input("大專：數學分數門檻", value=40.0, key="sub_math_thresh_input")

main_tab1, main_tab2 = st.tabs([
    "🤖 跨學年 AI 數據建模與門檻自動提煉",
    "🔮 校內成績升學預測與四類名單"
])

# ==================== 分頁一：AI 機器學習建模與對照總覽 ====================
with main_tab1:
    st.subheader("🤖 跨學年 AI 數據建模與門檻自動提煉")
    st.info("系統會優先自動偵測專案目錄中的數據檔案（如 `2526_T1A3_s6.xlsx`, `2526hkdse.xlsx`, `2425_T1A3_s6.xlsx`, `2425hkdse.xlsx`）。若未找到，亦可透過下方按鈕手動上載。")
    
    st.markdown("##### 📁 第一套歷史數據 (例如 2526 學年畢業生)")
    col_a1, col_b1 = st.columns(2)
    with col_a1: 
        f_tr_school_1_up = st.file_uploader("1A. 上傳/替換【2526】校內模擬試成績", type=["xls", "xlsx", "csv"], key="tr_s_1")
        f_tr_school_1, msg_s1 = resolve_file_source(f_tr_school_1_up, ['2526_T1A3_s6.xlsx', '2526_T1A3_s6.csv'])
        if f_tr_school_1: st.caption(msg_s1)
        else: st.caption("⚠️ 未找到 2526 校內成績預設檔，請手動上載。")

    with col_b1: 
        f_tr_dse_1_up = st.file_uploader("1B. 上傳/替換【2526】2526HKDSE 公開試成績", type=["xls", "xlsx", "csv"], key="tr_d_1")
        f_tr_dse_1, msg_d1 = resolve_file_source(f_tr_dse_1_up, ['2526hkdse.xlsx', '2526hkdse.csv'])
        if f_tr_dse_1: st.caption(msg_d1)
        else: st.caption("⚠️ 未找到 2526HKDSE 預設檔，請手動上載。")

    train_dfs = []
    corr_dfs = []
    comp_dfs = []
    
    # 處理第一套
    if f_tr_school_1 and f_tr_dse_1:
        df1, corr_df1, comp1, err1 = process_training_pair(f_tr_school_1, f_tr_dse_1, "2526")
        if err1:
            st.error(f"第一套數據：{err1}")
        else:
            if not df1.empty: train_dfs.append(df1)
            if corr_df1 is not None and not corr_df1.empty: corr_dfs.append(("2526", corr_df1))
            if comp1 is not None and not comp1.empty: comp_dfs.append(comp1)

    st.markdown("##### 📁 第二套歷史數據 (例如 2425 學年畢業生 - 增加樣本量與精準度)")
    col_a2, col_b2 = st.columns(2)
    with col_a2: 
        f_tr_school_2_up = st.file_uploader("2A. (選填) 上傳/替換【2425】校內模擬試成績", type=["xls", "xlsx", "csv"], key="tr_s_2")
        f_tr_school_2, msg_s2 = resolve_file_source(f_tr_school_2_up, ['2425_T1A3_s6.xlsx', '2425_T1A3_s6.csv'])
        if f_tr_school_2: st.caption(msg_s2)
        else: st.caption("⚠️ 未找到 2425 校內成績預設檔，可選擇手動上載。")

    with col_b2: 
        f_tr_dse_2_up = st.file_uploader("2B. (選填) 上傳/替換【2425】2425HKDSE 公開試成績", type=["xls", "xlsx", "csv"], key="tr_d_2")
        f_tr_dse_2, msg_d2 = resolve_file_source(f_tr_dse_2_up, ['2425hkdse.xlsx', '2425hkdse.csv'])
        if f_tr_dse_2: st.caption(msg_d2)
        else: st.caption("⚠️ 未找到 2425HKDSE 預設檔，可選擇手動上載。")

    # 處理第二套
    if f_tr_school_2 and f_tr_dse_2:
        df2, corr_df2, comp2, err2 = process_training_pair(f_tr_school_2, f_tr_dse_2, "2425")
        if err2:
            st.error(f"第二套數據：{err2}")
        else:
            if not df2.empty: train_dfs.append(df2)
            if corr_df2 is not None and not corr_df2.empty: corr_dfs.append(("2425", corr_df2))
            if comp2 is not None and not comp2.empty: comp_dfs.append(comp2)

    if train_dfs and HAS_SKLEARN:
        try:
            df_train = pd.concat(train_dfs, ignore_index=True)
            feats = ['Avg_Score', 'Chi_Score', 'Eng_Score', 'Math_Score']
            
            clf = DecisionTreeClassifier(max_depth=3, random_state=42)
            clf.fit(df_train[feats], df_train['Target_332'])
            
            # 提煉 AI 最佳切分門檻 (採用 Q1 下四分位數基準)
            ai_thresh = extract_ai_thresholds(df_train, feats)

            st.success(f"🎉 AI 模型訓練成功！已合併 {len(train_dfs)} 套歷史數據（總訓練樣本數：{len(df_train)} 人），並提煉 332 達標學生下四分位數保底門檻 (Q1 基準)。")
            
            col_tree1, col_tree2 = st.columns([1, 1])
            with col_tree1:
                st.markdown("##### 📊 各科目對升大學 (332) 的預測權重：")
                imp_df = pd.DataFrame({'校內指標': feats, 'AI 預測權重': clf.feature_importances_}).sort_values('AI 預測權重', ascending=False)
                st.dataframe(imp_df, use_container_width=True, hide_index=True)
                
            with col_tree2:
                st.markdown("##### 🌲 AI 自動提煉的數據決策樹規則 (Exact Cutoffs)：")
                rules = export_text(clf, feature_names=feats)
                st.code(rules, language="text")

        except Exception as e: st.error(f"建模失敗: {e}")

    # ===== 達大學 (332) 或大專 (222) 門檻之『公開試 vs 校內成績』對照表與平均分統計 =====
    if comp_dfs:
        st.divider()
        st.subheader("📊 達大學 (332) 或大專 (222) 收生門檻之『公開試成績 vs 校內成績』對照表與平均分總覽")
        
        df_all_comp = pd.concat(comp_dfs, ignore_index=True)
        
        # 統計指標計算
        u_group = df_all_comp[df_all_comp['達標類別'].str.contains("332")]
        sub_group = df_all_comp[df_all_comp['達標類別'].str.contains("222")]
        
        u_avg_val = safe_mean(u_group['校內總平均分']) if '校內總平均分' in u_group else np.nan
        sub_avg_val = safe_mean(sub_group['校內總平均分']) if '校內總平均分' in sub_group else np.nan
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("🎓 達大學門檻 (332) 總人數", f"{len(u_group)} 人")
        with c2: st.metric("🎓 達大學門檻學生【校內總平均分】", f"{u_avg_val:.1f} 分" if not pd.isna(u_avg_val) else "N/A")
        with c3: st.metric("🏫 達大專門檻 (222) 總人數", f"{len(sub_group)} 人")
        with c4: st.metric("🏫 達大專門檻學生【校內總平均分】", f"{sub_avg_val:.1f} 分" if not pd.isna(sub_avg_val) else "N/A")

        # 平均分彙整表
        avg_rows = []
        if not u_group.empty:
            avg_rows.append({
                '門檻類別': '🎓 大學收生門檻 (332A22)',
                '人數': len(u_group),
                '校內總平均分': safe_mean(u_group.get('校內總平均分')),
                '校內中文均分': safe_mean(u_group.get('校內中文分數')),
                '校內英文均分': safe_mean(u_group.get('校內英文分數')),
                '校內數學均分': safe_mean(u_group.get('校內數學分數')),
                '校內公社科均分': safe_mean(u_group.get('校內公社科分數'))
            })
        if not sub_group.empty:
            avg_rows.append({
                '門檻類別': '🏫 大專收生門檻 (222A22)',
                '人數': len(sub_group),
                '校內總平均分': safe_mean(sub_group.get('校內總平均分')),
                '校內中文均分': safe_mean(sub_group.get('校內中文分數')),
                '校內英文均分': safe_mean(sub_group.get('校內英文分數')),
                '校內數學均分': safe_mean(sub_group.get('校內數學分數')),
                '校內公社科均分': safe_mean(sub_group.get('校內公社科分數'))
            })
            
        if avg_rows:
            st.markdown("##### 📌 達標學生校內各科平均分統計 (Group Mean Summary)")
            st.dataframe(pd.DataFrame(avg_rows), use_container_width=True, hide_index=True)

        st.markdown("##### 📋 達標學生『公開試等級 vs 校內成績』明細表")
        
        # 雙條件篩選列：1. 達標類別  2. 學年
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            filter_tier = st.radio("篩選達標類別：", ["全部達標學生", "🎓 僅大學門檻 (332)", "🏫 僅大專門檻 (222)"], horizontal=True, key="filter_tier_radio")
        
        with col_f2:
            available_years = ["全部學年"] + sorted(list(df_all_comp['學年'].astype(str).unique()), reverse=True)
            filter_year = st.selectbox("篩選學年：", available_years, key="filter_year_select")

        df_disp_filtered = df_all_comp.copy()
        
        # 依類別篩選
        if filter_tier == "🎓 僅大學門檻 (332)":
            df_disp_filtered = df_disp_filtered[df_disp_filtered['達標類別'].str.contains("332")]
        elif filter_tier == "🏫 僅大專門檻 (222)":
            df_disp_filtered = df_disp_filtered[df_disp_filtered['達標類別'].str.contains("222")]
            
        # 依學年篩選
        if filter_year != "全部學年":
            df_disp_filtered = df_disp_filtered[df_disp_filtered['學年'].astype(str) == filter_year]
            
        st.dataframe(df_disp_filtered, use_container_width=True, hide_index=True)

    # 展開檢視全學科關聯度分析
    if corr_dfs:
        with st.expander("📈 檢視全學科試卷關聯度與效度分析 (All-Subject Correlation Analysis)", expanded=False):
            for label, c_df in corr_dfs:
                st.markdown(f"**【{label}】全學科關聯分析：**")
                st.dataframe(c_df, use_container_width=True, hide_index=True)

# ==================== 分頁二：校內成績預測與四類名單 ====================
with main_tab2:
    st.subheader("🔮 校內成績預測：產生「大學」、「特別支援（差一科）」、「大專」及「保底」名單")
    f_eval_up = st.file_uploader("上傳/替換校內成績表 (Excel / CSV)", type=["xls", "xlsx", "csv"], key="eval_up")
    f_eval, msg_eval_tab3 = resolve_file_source(f_eval_up, ['2526_T1A3_s6.xlsx', '2526_T1A3_s6.csv'])
    if f_eval: st.caption(msg_eval_tab3)
    
    if f_eval:
        try:
            df_ev = safe_read_file(f_eval)
            sch_reg_cols = [c for c in df_ev.columns if '*Reg. No.' in str(c) or 'Reg' in str(c) or '註冊編號' in str(c)]
            sch_reg = sch_reg_cols[0] if sch_reg_cols else df_ev.columns[0]
            df_ev['Reg_Clean'] = df_ev[sch_reg].apply(clean_id)

            if student_info_df is not None:
                df_ev = pd.merge(df_ev, student_info_df, left_on='Reg_Clean', right_on='註冊編號_clean', how='left')
            
            df_ev = extract_robust_scores(df_ev)
            
            # 自動提取所有選修科成績
            elective_cols_added = []
            for sub in SUBJECT_MAP[3:]: # 選修科配置
                s = get_school_subject_score(df_ev, sub['sch_keys'], sub['sch_ex'])
                if s is not None and s.notna().sum() > 0:
                    col_name = sub['name']
                    df_ev[col_name] = s.round(1)
                    elective_cols_added.append(col_name)

            # 公社科/公民科達標判斷函數 (保留數值分數顯示)
            def cs_is_attained(cs_val):
                if pd.isna(cs_val): return False
                val_str = str(cs_val).strip().upper()
                if val_str in ['A', 'PASS', 'ATTAINED', '達標', 'D', 'C', 'B', 'E']: return True
                if val_str in ['U', 'UNATTAINED', '不達標', 'FAIL', 'F']: return False
                try: return float(val_str) >= u_cs_thresh
                except: return False

            # 將公社科分數格式化為數值
            def format_cs_score(cs_val):
                if pd.isna(cs_val): return np.nan
                try:
                    return round(float(cs_val), 1)
                except:
                    return str(cs_val)

            df_ev['公民與社會發展科'] = df_ev['CS_Val'].apply(format_cs_score)

            def categorize_student(row):
                s, c, e, m, cs_raw = row['Avg_Score'], row['Chi_Score'], row['Eng_Score'], row['Math_Score'], row['CS_Val']
                cs_ok = cs_is_attained(cs_raw)
                
                cond_u = [
                    pd.notna(s) and s >= u_score_thresh,
                    pd.notna(c) and c >= u_chi_thresh,
                    pd.notna(e) and e >= u_eng_thresh,
                    pd.notna(m) and m >= u_math_thresh,
                    cs_ok
                ]
                cond_names = ['總平均分', '中文', '英文', '數學', '公社科']
                failed_conds = [cond_names[i] for i, passed in enumerate(cond_u) if not passed]
                
                if all(cond_u):
                    return pd.Series(["🎓 潛質入大學名單 (332A22)", "全科達標"])
                elif len(failed_conds) == 1:
                    return pd.Series(["🎯 特別支援名單 (差一科入大學)", f"僅未達標：{failed_conds[0]}"])
                elif pd.notna(s) and pd.notna(c) and pd.notna(e) and pd.notna(m) and s >= sub_score_thresh and c >= sub_chi_thresh and e >= sub_eng_thresh and m >= sub_math_thresh and cs_ok:
                    return pd.Series(["🏫 潛質入大專名單 (222A22)", "達大專要求"])
                else:
                    return pd.Series(["🛟 保底求合格名單 (關鍵科需支援)", f"未達標科目：{', '.join(failed_conds)}"])
            
            df_ev[['升學類別', '診斷與提示']] = df_ev.apply(categorize_student, axis=1)
            
            # 重命名主科欄位以保持簡潔
            df_ev = df_ev.rename(columns={
                'Avg_Score': '總平均分',
                'Chi_Score': '中文科',
                'Eng_Score': '英文科',
                'Math_Score': '數學必修'
            })
            
            counts = df_ev['升學類別'].value_counts()
            
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("🎓 潛質入大學 (332A22)", f"{counts.get('🎓 潛質入大學名單 (332A22)', 0)} 人")
            with m2: st.metric("🎯 特別支援 (差一科)", f"{counts.get('🎯 特別支援名單 (差一科入大學)', 0)} 人")
            with m3: st.metric("🏫 潛質入大專 (222A22)", f"{counts.get('🏫 潛質入大專名單 (222A22)', 0)} 人")
            with m4: st.metric("🛟 保底求合格名單", f"{counts.get('🛟 保底求合格名單 (關鍵科需支援)', 0)} 人")
            
            st.divider()
            
            tab_u, tab_sp, tab_sub, tab_base = st.tabs([
                "🎓 潛質入大學名單", 
                "🎯 特別支援名單 (差一科)", 
                "🏫 潛質入大專名單", 
                "🛟 保底求合格名單"
            ])
            
            disp_ev = []
            if '*Class' in df_ev.columns: disp_ev.append('*Class')
            if '*Class Number' in df_ev.columns: disp_ev.append('*Class Number')
            if '中文姓名' in df_ev.columns: disp_ev.append('中文姓名')
            if '*Student Name' in df_ev.columns: disp_ev.append('*Student Name')
            
            disp_ev.extend(['總平均分', '中文科', '英文科', '數學必修', '公民與社會發展科'])
            disp_ev.extend(elective_cols_added)
            disp_ev.extend(['升學類別', '診斷與提示'])
            
            sort_cols = [c for c in ['*Class', '*Class Number'] if c in df_ev.columns]
            
            with tab_u:
                df_sub_u = df_ev[df_ev['升學類別'].str.contains("大學名單")][disp_ev]
                if sort_cols: df_sub_u = df_sub_u.sort_values(sort_cols)
                st.dataframe(df_sub_u, use_container_width=True, hide_index=True)
            with tab_sp:
                df_sub_sp = df_ev[df_ev['升學類別'].str.contains("特別支援")][disp_ev]
                if sort_cols: df_sub_sp = df_sub_sp.sort_values(sort_cols)
                st.dataframe(df_sub_sp, use_container_width=True, hide_index=True)
            with tab_sub:
                df_sub_sub = df_ev[df_ev['升學類別'].str.contains("大專名單")][disp_ev]
                if sort_cols: df_sub_sub = df_sub_sub.sort_values(sort_cols)
                st.dataframe(df_sub_sub, use_container_width=True, hide_index=True)
            with tab_base:
                df_sub_base = df_ev[df_ev['升學類別'].str.contains("保底求合格")][disp_ev]
                if sort_cols: df_sub_base = df_sub_base.sort_values(sort_cols)
                st.dataframe(df_sub_base, use_container_width=True, hide_index=True)

        except Exception as e: st.error(f"分析失敗: {e}")
