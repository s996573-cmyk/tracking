import streamlit as st

# 1. 必須置於最頂端，防止網頁渲染白屏
st.set_page_config(page_title="學生學業表現與 DSE 升學分析系統", layout="wide")

import pandas as pd
import numpy as np
import os

# 安全引入 scikit-learn
try:
    from sklearn.tree import DecisionTreeClassifier
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

# 健壯的核心科目成績自動提取函數 (解決 None 問題)
def extract_robust_scores(df):
    # 1. 提取總平均分 (Total Score)
    tot_cols = [c for c in ['T2A3_Score', 'T1A3_Score', 'T2A1_Score', 'T1A1_Score', 'Score'] if c in df.columns]
    if not tot_cols:
        tot_cols = [c for c in df.columns if c.endswith('_Score') and not any(sub in c for sub in ['生物', '化學', '中文', '英文', '數學', '數必', '公民', '企財', '經濟', '地理', '歷史', '中史', '物理', '資通', '視憑', '倫教', '體育', '學培課', '退修課'])]
    tot_col = tot_cols[0] if tot_cols else None
    avg_score = pd.to_numeric(df[tot_col], errors='coerce').round(1) if tot_col else np.nan

    # 2. 提取中文分數 (自動合併中英文試卷)
    chi_cols = [c for c in df.columns if '中文' in c and 'Score' in c and not any(k in c for k in ['閱讀', '寫作', '聆聽', '口試', '說話'])]
    chi_series = None
    for cc in chi_cols:
        s = pd.to_numeric(df[cc], errors='coerce')
        chi_series = s if chi_series is None else chi_series.fillna(s)

    # 3. 提取英文分數 (自動合併試卷)
    eng_cols = [c for c in df.columns if '英文' in c and 'Score' in c and not any(k in c for k in ['閱讀', '作文', '聆聽', '口試', '語文'])]
    eng_series = None
    for ec in eng_cols:
        s = pd.to_numeric(df[ec], errors='coerce')
        eng_series = s if eng_series is None else eng_series.fillna(s)

    # 4. 提取數學必修分數 (精準合併 _C_Score 與 _E_Score)
    math_cols = [c for c in df.columns if any(k in c for k in ['數必', '數學']) and 'Score' in c and not any(k in c for k in ['數一', '數二', 'M1', 'M2'])]
    math_series = None
    for mc in math_cols:
        s = pd.to_numeric(df[mc], errors='coerce')
        math_series = s if math_series is None else math_series.fillna(s)

    # 5. 提取公社科 (CS)
    cs_cols = [c for c in df.columns if any(k in c for k in ['公民科', 'CS']) and any(k in c for k in ['Score', 'Grade'])]
    cs_series = None
    for csc in cs_cols:
        s = df[csc]
        cs_series = s if cs_series is None else cs_series.fillna(s)

    df['Avg_Score'] = avg_score
    df['Chi_Score'] = chi_series.round(1) if chi_series is not None else np.nan
    df['Eng_Score'] = eng_series.round(1) if eng_series is not None else np.nan
    df['Math_Score'] = math_series.round(1) if math_series is not None else np.nan
    df['CS_Val'] = cs_series if cs_series is not None else 50
    return df

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

main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
    "📚 近兩年學期成績分析 (T1A3/T2A3)", 
    "🎓 2526HKDSE 成效與門檻對照 (332A22 / 222A22)", 
    "🔮 中五/中六升學潛質預測與三類名單",
    "🤖 跨學年 AI 數據建模與預測"
])

# 側邊欄門檻設定
st.sidebar.header("⚙️ 大學門檻 (332A22) 校內分數設定")
u_score_thresh = st.sidebar.number_input("大學：平均分門檻", value=52.0)
u_chi_thresh = st.sidebar.number_input("大學：中文分數門檻", value=52.0)
u_eng_thresh = st.sidebar.number_input("大學：英文分數門檻", value=52.0)
u_math_thresh = st.sidebar.number_input("大學：數學分數門檻", value=40.0)
u_cs_thresh = st.sidebar.number_input("大學：公社科分數門檻", value=40.0)

st.sidebar.header("⚙️ 大專門檻 (222A22) 校內分數設定")
sub_score_thresh = st.sidebar.number_input("大專：平均分門檻", value=40.0)
sub_chi_thresh = st.sidebar.number_input("大專：中文分數門檻", value=40.0)
sub_eng_thresh = st.sidebar.number_input("大專：英文分數門檻", value=40.0)
sub_math_thresh = st.sidebar.number_input("大專：數學分數門檻", value=40.0)

# ==================== 分頁一：學期成績對比 ====================
with main_tab1:
    st.subheader("📚 近兩年學期成績對比（T1A3 上學期 vs T2A3 下學期）")
    c1, c2 = st.columns(2)
    with c1: f_t1a3 = st.file_uploader("上傳 上學期考試成績 (T1A3 Excel)", type=["xls", "xlsx"], key="t1a3_up")
    with c2: f_t2a3 = st.file_uploader("上傳 下學期考試成績 (T2A3 Excel)", type=["xls", "xlsx"], key="t2a3_up")

    if f_t1a3 and f_t2a3:
        try:
            df1 = pd.read_excel(f_t1a3)
            df2 = pd.read_excel(f_t2a3)
            
            merged = pd.merge(df1, df2, on=['*School Year', '*Class Level', '*Class', '*Class Number', '*Student Name', '*Reg. No.'], suffixes=('_T1A3', '_T2A3'))
            if student_info_df is not None:
                merged['Reg_Clean'] = merged['*Reg. No.'].astype(str).str.replace('#', '').str.strip()
                merged = pd.merge(merged, student_info_df, left_on='Reg_Clean', right_on='註冊編號_clean', how='left')
                
            merged['Score_T1A3'] = pd.to_numeric(merged.get('T1A3_Score', 0), errors='coerce').fillna(0)
            merged['Score_T2A3'] = pd.to_numeric(merged.get('T2A3_Score', 0), errors='coerce').fillna(0)
            merged['Score_Diff'] = (merged['Score_T2A3'] - merged['Score_T1A3']).round(1)
            
            st.success(f"✅ 成功配對 {len(merged)} 位學生的 T1A3 與 T2A3 成績！")
            
            disp_cols = ['*Class', '*Class Number']
            if '中文姓名' in merged.columns: disp_cols.append('中文姓名')
            disp_cols.extend(['*Student Name', 'Score_T1A3', 'Score_T2A3', 'Score_Diff'])
            
            st.dataframe(merged[disp_cols].sort_values(['*Class', '*Class Number']), use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"檔案處理發生錯誤: {e}")

# ==================== 分頁二：HKDSE 公開試門檻對照 ====================
with main_tab2:
    st.subheader("🎓 2526HKDSE 公開試實際表現與 332A22 / 222A22 門檻對照")
    f_dse = st.file_uploader("上傳 2526hkdse.xlsx 公開試成績表", type=["xls", "xlsx"], key="dse_main")
    
    if f_dse:
        try:
            df_dse = pd.read_excel(f_dse)
            if student_info_df is not None:
                df_dse['Reg_Clean'] = df_dse['Registration No.'].astype(str).str.strip()
                df_dse = pd.merge(df_dse, student_info_df, left_on='Reg_Clean', right_on='註冊編號_clean', how='left')
                
            df_dse['Chi_Lvl'] = df_dse['A010 Chinese'].apply(grade_to_level)
            df_dse['Eng_Lvl'] = df_dse['A020 English'].apply(grade_to_level)
            df_dse['Math_Lvl'] = df_dse['A030 Math Compulsory'].apply(grade_to_level)
            
            df_dse['Met_332A22'] = (df_dse['Chi_Lvl'] >= 3) & (df_dse['Eng_Lvl'] >= 3) & (df_dse['Math_Lvl'] >= 2)
            df_dse['Met_222A22'] = (df_dse['Chi_Lvl'] >= 2) & (df_dse['Eng_Lvl'] >= 2) & (df_dse['Math_Lvl'] >= 2)
            
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
            disp_dse.extend(['Name', 'A010 Chinese', 'A020 English', 'A030 Math Compulsory', 'Met_332A22', 'Met_222A22'])
            
            st.dataframe(df_dse[disp_dse], use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"DSE 檔案讀取失敗: {e}")

# ==================== 分頁三：升學潛質名單產出 (已修正 None 問題) ====================
with main_tab3:
    st.subheader("🔮 校內成績預測：產生「潛質入大學」、「潛質入大專」及「保底求合格」名單")
    f_eval = st.file_uploader("上傳校內成績表 (Excel)", type=["xls", "xlsx"], key="eval_up")
    
    if f_eval:
        try:
            df_ev = pd.read_excel(f_eval)
            if student_info_df is not None:
                df_ev['Reg_Clean'] = df_ev['*Reg. No.'].astype(str).str.replace('#', '').str.strip()
                df_ev = pd.merge(df_ev, student_info_df, left_on='Reg_Clean', right_on='註冊編號_clean', how='left')
            
            # 使用健壯函數提取成績
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
                
                if pd.notna(s) and pd.notna(c) and pd.notna(e) and pd.notna(m):
                    if s >= u_score_thresh and c >= u_chi_thresh and e >= u_eng_thresh and m >= u_math_thresh and cs_ok:
                        return "🎓 潛質入大學名單 (332A22)"
                    elif s >= sub_score_thresh and c >= sub_chi_thresh and e >= sub_eng_thresh and m >= sub_math_thresh and cs_ok:
                        return "🏫 潛質入大專名單 (222A22)"
                return "🛟 保底求合格名單 (關鍵科需支援)"
            
            df_ev['升學類別'] = df_ev.apply(categorize_student, axis=1)
            counts = df_ev['升學類別'].value_counts()
            
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("🎓 潛質入大學 (332A22)", f"{counts.get('🎓 潛質入大學名單 (332A22)', 0)} 人")
            with c2: st.metric("🏫 潛質入大專 (222A22)", f"{counts.get('🏫 潛質入大專名單 (222A22)', 0)} 人")
            with c3: st.metric("🛟 保底求合格名單", f"{counts.get('🛟 保底求合格名單 (關鍵科需支援)', 0)} 人")
            
            st.divider()
            
            tab_u, tab_sub, tab_base = st.tabs(["🎓 潛質入大學名單", "🏫 潛質入大專名單", "🛟 保底求合格名單"])
            
            disp_ev = ['*Class', '*Class Number']
            if '中文姓名' in df_ev.columns: disp_ev.append('中文姓名')
            disp_ev.extend(['*Student Name', 'Avg_Score', 'Chi_Score', 'Eng_Score', 'Math_Score', '升學類別'])
            
            with tab_u:
                st.dataframe(df_ev[df_ev['升學類別'].str.contains("大學")][disp_ev].sort_values(['*Class', '*Class Number']), use_container_width=True, hide_index=True)
            with tab_sub:
                st.dataframe(df_ev[df_ev['升學類別'].str.contains("大專")][disp_ev].sort_values(['*Class', '*Class Number']), use_container_width=True, hide_index=True)
            with tab_base:
                st.dataframe(df_ev[df_ev['升學類別'].str.contains("保底")][disp_ev].sort_values(['*Class', '*Class Number']), use_container_width=True, hide_index=True)

        except Exception as e: st.error(f"分析失敗: {e}")

# ==================== 分頁四：AI 跨學年機器學習建模 ====================
with main_tab4:
    st.subheader("🤖 跨學年 AI 數據建模與預測")
    col_a, col_b = st.columns(2)
    with col_a: f_tr_school = st.file_uploader("1. 上傳【過往畢業生】校內期末成績 (Excel)", type=["xls", "xlsx"], key="tr_s")
    with col_b: f_tr_dse = st.file_uploader("2. 上傳【過往畢業生】2526HKDSE 公開試 (Excel)", type=["xls", "xlsx"], key="tr_d")
    
    if f_tr_school and f_tr_dse and HAS_SKLEARN:
        try:
            d_sch = pd.read_excel(f_tr_school)
            d_dse = pd.read_excel(f_tr_dse)
            
            d_sch['Reg_Clean'] = d_sch['*Reg. No.'].astype(str).str.replace('#', '').str.strip()
            d_dse['Reg_Clean'] = d_dse['Registration No.'].astype(str).str.strip()
            
            m_ai = pd.merge(d_sch, d_dse, on='Reg_Clean')
            m_ai = extract_robust_scores(m_ai)
            
            m_ai['DSE_Chi'] = m_ai['A010 Chinese'].apply(grade_to_level)
            m_ai['DSE_Eng'] = m_ai['A020 English'].apply(grade_to_level)
            m_ai['DSE_Math'] = m_ai['A030 Math Compulsory'].apply(grade_to_level)
            m_ai['Target_332'] = ((m_ai['DSE_Chi'] >= 3) & (m_ai['DSE_Eng'] >= 3) & (m_ai['DSE_Math'] >= 2)).astype(int)
            
            feats = ['Avg_Score', 'Chi_Score', 'Eng_Score', 'Math_Score']
            df_train = m_ai.dropna(subset=feats + ['Target_332'])
            
            clf = DecisionTreeClassifier(max_depth=3, random_state=42)
            clf.fit(df_train[feats], df_train['Target_332'])
            
            st.success(f"🎉 AI 模型訓練成功！訓練樣本數：{len(df_train)} 人。")
            imp_df = pd.DataFrame({'指標': feats, '預測影響權重': clf.feature_importances_}).sort_values('預測影響權重', ascending=False)
            st.dataframe(imp_df, use_container_width=True, hide_index=True)
            
        except Exception as e: st.error(f"建模失敗: {e}")
