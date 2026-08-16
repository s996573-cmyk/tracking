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

# 安全讀取 CSV 與 Excel (自動檢測 Big5 / UTF-8 / GB18030 編碼)
def safe_read_file(file_obj):
    filename = getattr(file_obj, 'name', '').lower()
    if filename.endswith('.csv'):
        encodings = ['utf-8-sig', 'big5-hkscs', 'big5', 'cp950', 'gb18030', 'gbk', 'utf-8']
        for enc in encodings:
            try:
                if hasattr(file_obj, 'seek'): file_obj.seek(0)
                df = pd.read_csv(file_obj, encoding=enc)
                if not df.empty: return df
            except Exception:
                continue
        if hasattr(file_obj, 'seek'): file_obj.seek(0)
        return pd.read_csv(file_obj, encoding_errors='ignore')
    else:
        if hasattr(file_obj, 'seek'): file_obj.seek(0)
        return pd.read_excel(file_obj)

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
        s = df_out[csc]
        cs_series = s if cs_series is None else cs_series.fillna(s)

    df_out['Avg_Score'] = avg_score
    df_out['Chi_Score'] = chi_series.round(1) if chi_series is not None else np.nan
    df_out['Eng_Score'] = eng_series.round(1) if eng_series is not None else np.nan
    df_out['Math_Score'] = math_series.round(1) if math_series is not None else np.nan
    df_out['CS_Val'] = cs_series if cs_series is not None else 50
    return df_out

# 通用過往畢業生數據對照與清洗處理
def process_training_pair(f_sch, f_dse):
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
        return pd.DataFrame(), pd.DataFrame(), f"⚠️ 對照失敗：無法透過校內【{sch_reg}】與 DSE【{dse_reg}】匹配學生身份，請確認兩檔學號是否相符。"
        
    m_ai = extract_robust_scores(m_ai)
    
    chi_col = find_best_dse_subject_col(d_dse.columns, ['A010', 'CHINESE', '中文'])
    eng_col = find_best_dse_subject_col(d_dse.columns, ['A020', 'ENGLISH', '英文'])
    math_col = find_best_dse_subject_col(d_dse.columns, ['A030', 'MATH', '數學'], ['M1', 'M2', '數一', '數二', 'EXTENDED', '單元'])
    
    if not (chi_col and eng_col and math_col):
        return pd.DataFrame(), pd.DataFrame(), "⚠️ DSE 核心科目欄位辨識失敗：DSE 表格需包含中文、英文及數學必修部分成績。"
        
    m_ai['DSE_Chi'] = m_ai[chi_col].apply(parse_dse_grade)
    m_ai['DSE_Eng'] = m_ai[eng_col].apply(parse_dse_grade)
    m_ai['DSE_Math'] = m_ai[math_col].apply(parse_dse_grade)
    
    m_ai['Target_332'] = ((m_ai['DSE_Chi'] >= 3) & (m_ai['DSE_Eng'] >= 3) & (m_ai['DSE_Math'] >= 2)).astype(int)
    
    feats = ['Avg_Score', 'Chi_Score', 'Eng_Score', 'Math_Score']
    clean_df = m_ai.dropna(subset=feats + ['Target_332'])[feats + ['Target_332', 'DSE_Chi', 'DSE_Eng', 'DSE_Math']]
    
    # 計算全學科相關係數表
    df_all_corr = compute_all_subjects_correlation(d_sch, d_dse)
    
    return clean_df, df_all_corr, None

# 優先提取決策樹「最頂層主幹（根節點）」切分門檻
def extract_ai_thresholds(clf, df_train, feats):
    thresholds = {}
    for i in range(clf.tree_.node_count):
        f_idx = clf.tree_.feature[i]
        if f_idx >= 0:
            f_name = feats[f_idx]
            t_val = round(float(clf.tree_.threshold[i]), 2)
            if f_name not in thresholds:
                thresholds[f_name] = t_val
                
    successful_df = df_train[df_train['Target_332'] == 1]
    
    defaults = {
        'Avg_Score': 55.0,
        'Chi_Score': 52.0,
        'Eng_Score': 55.0,
        'Math_Score': 40.0
    }
    
    final_thresh = {}
    for f in feats:
        if f in thresholds:
            final_thresh[f] = thresholds[f]
        elif not successful_df.empty and f in successful_df.columns:
            val = round(float(successful_df[f].quantile(0.10)), 2)
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

main_tab1, main_tab2, main_tab3 = st.tabs([
    "🤖 跨學年 AI 數據建模與門檻自動提煉",
    "🎓 2526HKDSE 公開試成效與中六 T2A3/模擬試對照", 
    "🔮 校內成績升學預測與四類名單"
])

# 初始化 session state 預設門檻變數
if 'u_score_val' not in st.session_state: st.session_state['u_score_val'] = 55.0
if 'u_chi_val' not in st.session_state: st.session_state['u_chi_val'] = 52.0
if 'u_eng_val' not in st.session_state: st.session_state['u_eng_val'] = 55.0
if 'u_math_val' not in st.session_state: st.session_state['u_math_val'] = 40.0
if 'u_cs_val' not in st.session_state: st.session_state['u_cs_val'] = 40.0

# 側邊欄門檻設定
st.sidebar.header("⚙️ 大學門檻 (332A22) 校內分數設定")

if st.session_state.get('ai_updated_flag', False):
    st.sidebar.success("🤖 已根據多套 AI 訓練數據自動更新大學主幹門檻！")

u_score_thresh = st.sidebar.number_input("大學：總平均分門檻", value=st.session_state['u_score_val'])
u_chi_thresh = st.sidebar.number_input("大學：中文分數門檻", value=st.session_state['u_chi_val'])
u_eng_thresh = st.sidebar.number_input("大學：英文分數門檻", value=st.session_state['u_eng_val'])
u_math_thresh = st.sidebar.number_input("大學：數學分數門檻", value=st.session_state['u_math_val'])
u_cs_thresh = st.sidebar.number_input("大學：公社科分數門檻", value=st.session_state['u_cs_val'])

st.sidebar.header("⚙️ 大專門檻 (222A22) 校內分數設定")
sub_score_thresh = st.sidebar.number_input("大專：總平均分門檻", value=40.0)
sub_chi_thresh = st.sidebar.number_input("大專：中文分數門檻", value=40.0)
sub_eng_thresh = st.sidebar.number_input("大專：英文分數門檻", value=40.0)
sub_math_thresh = st.sidebar.number_input("大專：數學分數門檻", value=40.0)

# ==================== 分頁一（最左）：AI 機器學習建模 ====================
with main_tab1:
    st.subheader("🤖 跨學年 AI 數據建模與門檻自動提煉")
    st.write("支援上傳多個學年的畢業生數據（Excel 或 CSV），透過精準對照校內 `*Reg. No.` 與 DSE `Registration No.` 建立預測模型，提煉代表性門檻。")
    
    st.markdown("##### 📁 第一套歷史數據 (例如 2526 屆畢業生)")
    col_a1, col_b1 = st.columns(2)
    with col_a1: f_tr_school_1 = st.file_uploader("1A. 上傳【2526 屆】校內模擬試/期末成績 (*Reg. No.)", type=["xls", "xlsx", "csv"], key="tr_s_1")
    with col_b1: f_tr_dse_1 = st.file_uploader("1B. 上傳【2526 屆】2526HKDSE 公開試成績 (Registration No.)", type=["xls", "xlsx", "csv"], key="tr_d_1")
    
    train_dfs = []
    
    # 處理第一套
    if f_tr_school_1 and f_tr_dse_1:
        df1, corr_df1, err1 = process_training_pair(f_tr_school_1, f_tr_dse_1)
        if err1:
            st.error(f"第一套數據：{err1}")
        elif not df1.empty:
            train_dfs.append(df1)
            with st.expander("📈 檢視【第一套數據】全學科試卷關聯度與效度分析 (All-Subject Correlation Analysis)", expanded=True):
                st.dataframe(corr_df1, use_container_width=True, hide_index=True)
            
    st.markdown("##### 📁 第二套歷史數據 (例如 2425 屆畢業生 - 選填，增加樣本量與精準度)")
    col_a2, col_b2 = st.columns(2)
    with col_a2: f_tr_school_2 = st.file_uploader("2A. (選填) 上傳【2425 屆】校內模擬試/期末成績 (*Reg. No.)", type=["xls", "xlsx", "csv"], key="tr_s_2")
    with col_b2: f_tr_dse_2 = st.file_uploader("2B. (選填) 上傳【2425 屆】2425HKDSE 公開試成績 (Registration No.)", type=["xls", "xlsx", "csv"], key="tr_d_2")

    # 處理第二套
    if f_tr_school_2 and f_tr_dse_2:
        df2, corr_df2, err2 = process_training_pair(f_tr_school_2, f_tr_dse_2)
        if err2:
            st.error(f"第二套數據：{err2}")
        elif not df2.empty:
            train_dfs.append(df2)
            with st.expander("📈 檢視【第二套數據】全學科試卷關聯度與效度分析 (All-Subject Correlation Analysis)", expanded=True):
                st.dataframe(corr_df2, use_container_width=True, hide_index=True)

    if train_dfs and HAS_SKLEARN:
        try:
            df_train = pd.concat(train_dfs, ignore_index=True)
            feats = ['Avg_Score', 'Chi_Score', 'Eng_Score', 'Math_Score']
            
            clf = DecisionTreeClassifier(max_depth=3, random_state=42)
            clf.fit(df_train[feats], df_train['Target_332'])
            
            # 提煉 AI 最佳切分門檻 (優先取主幹)
            ai_thresh = extract_ai_thresholds(clf, df_train, feats)
            
            mapping = {
                'Avg_Score': 'u_score_val',
                'Chi_Score': 'u_chi_val',
                'Eng_Score': 'u_eng_val',
                'Math_Score': 'u_math_val'
            }
            
            need_rerun = False
            for feat_name, state_key in mapping.items():
                new_val = ai_thresh.get(feat_name, st.session_state[state_key])
                if round(st.session_state[state_key], 2) != round(new_val, 2):
                    st.session_state[state_key] = new_val
                    need_rerun = True

            if need_rerun or not st.session_state.get('ai_updated_flag', False):
                st.session_state['ai_updated_flag'] = True
                st.rerun()

            st.success(f"🎉 AI 模型訓練成功！已合併 {len(train_dfs)} 套歷史數據（總訓練樣本數：{len(df_train)} 人），並將主幹最優切分線同步至左側大學門檻設定。")
            
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

# ==================== 分頁二：2526HKDSE 與中六 T2A3 分數對照 ====================
with main_tab2:
    st.subheader("🎓 2526HKDSE 公開試實際表現與 332A22 / 222A22 門檻對照（含中六 T2A3 對照）")
    col_d1, col_d2 = st.columns(2)
    with col_d1: f_dse = st.file_uploader("1. 上傳 2526hkdse 成績表 (Excel / CSV)", type=["xls", "xlsx", "csv"], key="dse_main")
    with col_d2: f_s6_t2a3 = st.file_uploader("2. (選填) 上傳中六 T2A3 / 模擬試成績表進行對照 (Excel / CSV)", type=["xls", "xlsx", "csv"], key="s6_t2a3_dse")
    
    if f_dse:
        try:
            df_dse = safe_read_file(f_dse)
            dse_reg_cols = [c for c in df_dse.columns if 'Registration No' in str(c) or 'Reg' in str(c) or '註冊編號' in str(c)]
            dse_reg = dse_reg_cols[0] if dse_reg_cols else df_dse.columns[0]
            df_dse['Registration_Clean'] = df_dse[dse_reg].apply(clean_id)
            
            if student_info_df is not None:
                df_dse = pd.merge(df_dse, student_info_df, left_on='Registration_Clean', right_on='註冊編號_clean', how='left')
                
            c_col = find_best_dse_subject_col(df_dse.columns, ['A010', 'CHINESE', '中文'])
            e_col = find_best_dse_subject_col(df_dse.columns, ['A020', 'ENGLISH', '英文'])
            m_col = find_best_dse_subject_col(df_dse.columns, ['A030', 'MATH', '數學'], ['M1', 'M2', '數一', '數二', 'EXTENDED', '單元'])

            df_dse['Chi_Lvl'] = df_dse[c_col].apply(parse_dse_grade)
            df_dse['Eng_Lvl'] = df_dse[e_col].apply(parse_dse_grade)
            df_dse['Math_Lvl'] = df_dse[m_col].apply(parse_dse_grade)
            
            df_dse['Met_332A22'] = (df_dse['Chi_Lvl'] >= 3) & (df_dse['Eng_Lvl'] >= 3) & (df_dse['Math_Lvl'] >= 2)
            df_dse['Met_222A22'] = (df_dse['Chi_Lvl'] >= 2) & (df_dse['Eng_Lvl'] >= 2) & (df_dse['Math_Lvl'] >= 2)
            
            if f_s6_t2a3:
                df_s6 = safe_read_file(f_s6_t2a3)
                sch_reg_cols = [c for c in df_s6.columns if '*Reg. No.' in str(c) or 'Reg' in str(c) or '註冊編號' in str(c)]
                sch_reg = sch_reg_cols[0] if sch_reg_cols else df_s6.columns[0]
                df_s6['Reg_Clean'] = df_s6[sch_reg].apply(clean_id)
                df_s6 = extract_robust_scores(df_s6)
                
                df_s6_sub = df_s6[['Reg_Clean', 'Avg_Score', 'Chi_Score', 'Eng_Score', 'Math_Score']].rename(columns={
                    'Avg_Score': 'S6_T2A3_總平均分',
                    'Chi_Score': 'S6_T2A3_中文',
                    'Eng_Score': 'S6_T2A3_英文',
                    'Math_Score': 'S6_T2A3_數學'
                })
                
                df_dse = pd.merge(df_dse, df_s6_sub, left_on='Registration_Clean', right_on='Reg_Clean', how='left')
                st.success("✅ 成功對照並融合「中六 T2A3 校內分數」與「DSE 實際成績」！")

            u_count = df_dse['Met_332A22'].sum()
            sub_count = df_dse['Met_222A22'].sum() - u_count
            base_count = len(df_dse) - df_dse['Met_222A22'].sum()
            
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("🎓 達大學門檻 (332A22)", f"{u_count} 人")
            with c2: st.metric("🏫 達大專門檻 (222A22)", f"{sub_count} 人")
            with c3: st.metric("🛟 需保底支援 (未達 222)", f"{base_count} 人")
            
            st.divider()
            
            disp_dse = []
            if 'Class' in df_dse.columns: disp_dse.append('Class')
            if 'Class No.' in df_dse.columns: disp_dse.append('Class No.')
            if '中文姓名' in df_dse.columns: disp_dse.append('中文姓名')
            if 'Name' in df_dse.columns: disp_dse.append('Name')
            
            if f_s6_t2a3:
                disp_dse.extend(['S6_T2A3_總平均分', 'S6_T2A3_中文', 'S6_T2A3_英文', 'S6_T2A3_數學'])
                
            disp_dse.extend([c_col, e_col, m_col, 'Met_332A22', 'Met_222A22'])
            
            st.dataframe(df_dse[disp_dse], use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"DSE 檔案讀取或對照失敗: {e}")

# ==================== 分頁三：校內成績預測與四類名單 ====================
with main_tab3:
    st.subheader("🔮 校內成績預測：產生「大學」、「特別支援（差一科）」、「大專」及「保底」名單")
    f_eval = st.file_uploader("上傳校內成績表 (Excel / CSV)", type=["xls", "xlsx", "csv"], key="eval_up")
    
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

            def cs_is_attained(cs_val):
                if pd.isna(cs_val): return False
                val_str = str(cs_val).strip().upper()
                if val_str in ['A', 'PASS', 'ATTAINED', '達標', 'D', 'C', 'B', 'E']: return True
                try: return float(val_str) >= u_cs_thresh
                except: return False

            def categorize_student(row):
                s, c, e, m, cs = row['Avg_Score'], row['Chi_Score'], row['Eng_Score'], row['Math_Score'], row['CS_Val']
                cs_ok = cs_is_attained(cs)
                
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
            
            # 依序放入主科與所有選修科分數
            disp_ev.extend(['總平均分', '中文科', '英文科', '數學必修'])
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
