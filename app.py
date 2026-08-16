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

# 安全轉換數字
def to_numeric(val):
    try: return float(val)
    except: return np.nan

# DSE 等級/分數轉為數值
def grade_to_level(g):
    g_str = str(g).strip().upper()
    mapping = {'5**': 7, '5*': 6, '5': 5, '4': 4, '3': 3, '2': 2, '1': 1, 'U': 0, 'UNCLASSIFIED': 0, 'A': 'A'}
    if g_str in mapping: return mapping[g_str]
    try: return float(g_str)
    except: return np.nan

# 健壯的核心科目成績自動提取函數
def extract_robust_scores(df):
    df_out = df.copy()
    
    # 1. 提取總平均分
    tot_cols = [c for c in ['T2A3_Score', 'T1A3_Score', 'T2A1_Score', 'T1A1_Score', 'Score'] if c in df_out.columns]
    if not tot_cols:
        tot_cols = [c for c in df_out.columns if str(c).endswith('_Score') and not any(sub in str(c) for sub in ['生物', '化學', '中文', '英文', '數學', '數必', '公民', '企財', '經濟', '地理', '歷史', '中史', '物理', '資通', '視憑', '倫教', '體育', '學培課', '退修課'])]
    tot_col = tot_cols[0] if tot_cols else None
    
    avg_score = pd.to_numeric(df_out[tot_col], errors='coerce').round(1) if tot_col else pd.Series(np.nan, index=df_out.index)

    # 2. 提取中文分數
    chi_cols = [c for c in df_out.columns if '中文' in str(c) and 'Score' in str(c) and not any(k in str(c) for k in ['閱讀', '寫作', '聆聽', '口試', '說話'])]
    chi_series = None
    for cc in chi_cols:
        s = pd.to_numeric(df_out[cc], errors='coerce')
        chi_series = s if chi_series is None else chi_series.fillna(s)

    # 3. 提取英文分數
    eng_cols = [c for c in df_out.columns if '英文' in str(c) and 'Score' in str(c) and not any(k in str(c) for k in ['閱讀', '作文', '聆聽', '口試', '語文'])]
    eng_series = None
    for ec in eng_cols:
        s = pd.to_numeric(df_out[ec], errors='coerce')
        eng_series = s if eng_series is None else eng_series.fillna(s)

    # 4. 提取數學必修分數
    math_cols = [c for c in df_out.columns if any(k in str(c) for k in ['數必', '數學']) and 'Score' in str(c) and not any(k in str(c) for k in ['數一', '數二', 'M1', 'M2'])]
    math_series = None
    for mc in math_cols:
        s = pd.to_numeric(df_out[mc], errors='coerce')
        math_series = s if math_series is None else math_series.fillna(s)

    # 5. 提取公社科 (CS)
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

# 從 AI 決策樹及歷史達標數據自動提取精準門檻
def extract_ai_thresholds(clf, df_train, feats):
    thresholds = {}
    for i in range(clf.tree_.node_count):
        f_idx = clf.tree_.feature[i]
        if f_idx >= 0:
            f_name = feats[f_idx]
            t_val = round(float(clf.tree_.threshold[i]), 1)
            if f_name not in thresholds or t_val < thresholds[f_name]:
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
            val = round(float(successful_df[f].quantile(0.10)), 1)
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
            df_info = pd.read_excel(file_path)
            if '註冊編號' in df_info.columns and '中文姓名' in df_info.columns:
                df_info['註冊編號_clean'] = df_info['註冊編號'].astype(str).str.strip()
                return df_info[['註冊編號_clean', '中文姓名']].dropna()
        except Exception: pass
    return None

student_info_df = load_student_info()

st.title("📊 學生成績追蹤、數據建模與 DSE 升學分析系統")

main_tab1, main_tab2, main_tab3 = st.tabs([
    "🤖 跨學年 AI 數據建模與預測",
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
    st.sidebar.success("🤖 已根據 AI 訓練結果自動更新大學門檻！")

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
    st.subheader("🤖 跨學年 AI 數據建模與升學概率預測")
    st.write("利用過往畢業生的「校內期末成績」與「2526HKDSE 實際成績」自動訓練 AI 模型，提煉數據驅動的精準門檻，並自動同步至左側邊欄設定。")
    
    col_a, col_b = st.columns(2)
    with col_a: f_tr_school = st.file_uploader("1. 上傳【過往畢業生】校內期末/模擬試成績 (Excel)", type=["xls", "xlsx"], key="tr_s")
    with col_b: f_tr_dse = st.file_uploader("2. 上傳【過往畢業生】2526HKDSE 公開試成績 (Excel)", type=["xls", "xlsx"], key="tr_d")
    
    if f_tr_school and f_tr_dse and HAS_SKLEARN:
        try:
            d_sch = pd.read_excel(f_tr_school)
            d_dse = pd.read_excel(f_tr_dse)
            
            d_sch['Reg_Clean'] = d_sch['*Reg. No.'].astype(str).str.replace('#', '').str.strip()
            d_dse['Registration No.'] = d_dse['Registration No.'].astype(str).str.strip()
            
            m_ai = pd.merge(d_sch, d_dse, left_on='Reg_Clean', right_on='Registration No.')
            m_ai = extract_robust_scores(m_ai)
            
            m_ai['DSE_Chi'] = m_ai['A010 Chinese'].apply(grade_to_level)
            m_ai['DSE_Eng'] = m_ai['A020 English'].apply(grade_to_level)
            m_ai['DSE_Math'] = m_ai['A030 Math Compulsory'].apply(grade_to_level)
            m_ai['Target_332'] = ((m_ai['DSE_Chi'] >= 3) & (m_ai['DSE_Eng'] >= 3) & (m_ai['DSE_Math'] >= 2)).astype(int)
            
            feats = ['Avg_Score', 'Chi_Score', 'Eng_Score', 'Math_Score']
            df_train = m_ai.dropna(subset=feats + ['Target_332'])
            
            clf = DecisionTreeClassifier(max_depth=3, random_state=42)
            clf.fit(df_train[feats], df_train['Target_332'])
            
            # 提煉 AI 最佳切分門檻
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
                if round(st.session_state[state_key], 1) != round(new_val, 1):
                    st.session_state[state_key] = new_val
                    need_rerun = True

            if need_rerun or not st.session_state.get('ai_updated_flag', False):
                st.session_state['ai_updated_flag'] = True
                st.rerun()

            st.success(f"🎉 AI 模型訓練成功！已自動將最優切分線同步至左側大學門檻設定。（歷史樣本數：{len(df_train)} 人）")
            
            col_tree1, col_tree2 = st.columns([1, 1])
            with col_tree1:
                st.markdown("##### 📊 各科目對升大學 (332) 的預測影響權重：")
                imp_df = pd.DataFrame({'校內指標': feats, 'AI 預測權重': clf.feature_importances_}).sort_values('AI 預測權重', ascending=False)
                st.dataframe(imp_df, use_container_width=True, hide_index=True)
                
            with col_tree2:
                st.markdown("##### 🌲 AI 自動提煉的數據決策樹規則 (Exact Cutoffs)：")
                rules = export_text(clf, feature_names=feats)
                st.code(rules, language="text")
            
            st.divider()
            st.markdown("### 🔮 預測【現屆中六】升學概率")
            f_next_cohort = st.file_uploader("3. 上傳【現屆中六】校內成績表進行概率預測 (Excel)", type=["xls", "xlsx"], key="next_cohort")
            
            if f_next_cohort:
                df_next = pd.read_excel(f_next_cohort)
                if student_info_df is not None:
                    df_next['Reg_Clean'] = df_next['*Reg. No.'].astype(str).str.replace('#', '').str.strip()
                    df_next = pd.merge(df_next, student_info_df, left_on='Reg_Clean', right_on='註冊編號_clean', how='left')
                
                df_next = extract_robust_scores(df_next)
                
                probs = clf.predict_proba(df_next[feats].fillna(0))[:, 1]
                df_next['AI 預測入大學概率 (%)'] = (probs * 100).round(1)
                
                def level_tag(p):
                    if p >= 70: return "🎓 高概率升學 (≥70%)"
                    elif p >= 35: return "🎯 邊緣衝刺 (35-69%)"
                    else: return "🛟 保底急救 (<35%)"
                    
                df_next['AI 評估分組'] = df_next['AI 預測入大學概率 (%)'].apply(level_tag)
                
                disp_ai = ['*Class', '*Class Number']
                if '中文姓名' in df_next.columns: disp_ai.append('中文姓名')
                disp_ai.extend(['*Student Name', 'Avg_Score', 'Chi_Score', 'Eng_Score', 'Math_Score', 'AI 預測入大學概率 (%)', 'AI 評估分組'])
                
                st.dataframe(df_next[disp_ai].sort_values(['*Class', '*Class Number']), use_container_width=True, hide_index=True)

        except Exception as e: st.error(f"建模或預測失敗: {e}")

# ==================== 分頁二：2526HKDSE 與中六 T2A3 分數對照 ====================
with main_tab2:
    st.subheader("🎓 2526HKDSE 公開試實際表現與 332A22 / 222A22 門檻對照（含中六 T2A3 對照）")
    col_d1, col_d2 = st.columns(2)
    with col_d1: f_dse = st.file_uploader("1. 上傳 2526hkdse.xlsx 公開試成績表", type=["xls", "xlsx"], key="dse_main")
    with col_d2: f_s6_t2a3 = st.file_uploader("2. (選填) 上傳中六 T2A3 / 模擬試成績表進行對照", type=["xls", "xlsx"], key="s6_t2a3_dse")
    
    if f_dse:
        try:
            df_dse = pd.read_excel(f_dse)
            df_dse['Registration No.'] = df_dse['Registration No.'].astype(str).str.strip()
            
            if student_info_df is not None:
                df_dse = pd.merge(df_dse, student_info_df, left_on='Registration No.', right_on='註冊編號_clean', how='left')
                
            df_dse['Chi_Lvl'] = df_dse['A010 Chinese'].apply(grade_to_level)
            df_dse['Eng_Lvl'] = df_dse['A020 English'].apply(grade_to_level)
            df_dse['Math_Lvl'] = df_dse['A030 Math Compulsory'].apply(grade_to_level)
            
            df_dse['Met_332A22'] = (df_dse['Chi_Lvl'] >= 3) & (df_dse['Eng_Lvl'] >= 3) & (df_dse['Math_Lvl'] >= 2)
            df_dse['Met_222A22'] = (df_dse['Chi_Lvl'] >= 2) & (df_dse['Eng_Lvl'] >= 2) & (df_dse['Math_Lvl'] >= 2)
            
            if f_s6_t2a3:
                df_s6 = pd.read_excel(f_s6_t2a3)
                df_s6['Reg_Clean'] = df_s6['*Reg. No.'].astype(str).str.replace('#', '').str.strip()
                df_s6 = extract_robust_scores(df_s6)
                
                df_s6_sub = df_s6[['Reg_Clean', 'Avg_Score', 'Chi_Score', 'Eng_Score', 'Math_Score']].rename(columns={
                    'Avg_Score': 'S6_T2A3_總平均分',
                    'Chi_Score': 'S6_T2A3_中文',
                    'Eng_Score': 'S6_T2A3_英文',
                    'Math_Score': 'S6_T2A3_數學'
                })
                
                df_dse = pd.merge(df_dse, df_s6_sub, left_on='Registration No.', right_on='Reg_Clean', how='left')
                st.success("✅ 成功對照並融合「中六 T2A3 校內分數」與「DSE 實際成績」！")

            u_count = df_dse['Met_332A22'].sum()
            sub_count = df_dse['Met_222A22'].sum() - u_count
            base_count = len(df_dse) - df_dse['Met_222A22'].sum()
            
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("🎓 達大學門檻 (332A22)", f"{u_count} 人")
            with c2: st.metric("🏫 達大專門檻 (222A22)", f"{sub_count} 人")
            with c3: st.metric("🛟 需保底支援 (未達 222)", f"{base_count} 人")
            
            st.divider()
            
            disp_dse = ['Class', 'Class No.']
            if '中文姓名' in df_dse.columns: disp_dse.append('中文姓名')
            disp_dse.append('Name')
            
            if f_s6_t2a3:
                disp_dse.extend(['S6_T2A3_總平均分', 'S6_T2A3_中文', 'S6_T2A3_英文', 'S6_T2A3_數學'])
                
            disp_dse.extend(['A010 Chinese', 'A020 English', 'A030 Math Compulsory', 'Met_332A22', 'Met_222A22'])
            
            st.dataframe(df_dse[disp_dse], use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"DSE 檔案讀取或對照失敗: {e}")

# ==================== 分頁三：校內成績預測與四類名單 ====================
with main_tab3:
    st.subheader("🔮 校內成績預測：產生「大學」、「特別支援（差一科）」、「大專」及「保底」名單")
    f_eval = st.file_uploader("上傳校內成績表 (Excel)", type=["xls", "xlsx"], key="eval_up")
    
    if f_eval:
        try:
            df_ev = pd.read_excel(f_eval)
            if student_info_df is not None:
                df_ev['Reg_Clean'] = df_ev['*Reg. No.'].astype(str).str.replace('#', '').str.strip()
                df_ev = pd.merge(df_ev, student_info_df, left_on='Reg_Clean', right_on='註冊編號_clean', how='left')
            
            df_ev = extract_robust_scores(df_ev)
            
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
            
            disp_ev = ['*Class', '*Class Number']
            if '中文姓名' in df_ev.columns: disp_ev.append('中文姓名')
            disp_ev.extend(['*Student Name', 'Avg_Score', 'Chi_Score', 'Eng_Score', 'Math_Score', '升學類別', '診斷與提示'])
            
            with tab_u:
                st.dataframe(df_ev[df_ev['升學類別'].str.contains("大學名單")][disp_ev].sort_values(['*Class', '*Class Number']), use_container_width=True, hide_index=True)
            with tab_sp:
                st.dataframe(df_ev[df_ev['升學類別'].str.contains("特別支援")][disp_ev].sort_values(['*Class', '*Class Number']), use_container_width=True, hide_index=True)
            with tab_sub:
                st.dataframe(df_ev[df_ev['升學類別'].str.contains("大專名單")][disp_ev].sort_values(['*Class', '*Class Number']), use_container_width=True, hide_index=True)
            with tab_base:
                st.dataframe(df_ev[df_ev['升學類別'].str.contains("保底求合格")][disp_ev].sort_values(['*Class', '*Class Number']), use_container_width=True, hide_index=True)

        except Exception as e: st.error(f"分析失敗: {e}")
