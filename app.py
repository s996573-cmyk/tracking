import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier

# 處理文字名次
def extract_rank(val):
    if pd.isna(val) or not isinstance(val, str): return np.nan
    try: return int(val.split('#')[0])
    except: return np.nan

# 安全轉換數字
def to_numeric(val):
    try: return float(val)
    except: return np.nan

# DSE 等級轉為數值分數
def grade_to_level(g):
    g_str = str(g).strip().upper()
    mapping = {'5**': 7, '5*': 6, '5': 5, '4': 4, '3': 3, '2': 2, '1': 1, 'U': 0, 'UNCLASSIFIED': 0}
    if g_str in mapping: return mapping[g_str]
    try: return float(g_str)
    except: return np.nan

# --- 載入本機學生資料庫 (提取中文姓名) ---
@st.cache_data
def load_student_info():
    file_path = 'student_info.xlsx'
    if os.path.exists(file_path):
        try:
            df_info = pd.read_excel(file_path)
            if '註冊編號' in df_info.columns and '中文姓名' in df_info.columns:
                df_info['註冊編號_clean'] = df_info['註冊編號'].astype(str).str.strip()
                return df_info[['註冊編號_clean', '中文姓名']].dropna()
        except Exception as e:
            st.warning(f"讀取 {file_path} 發生錯誤。訊息: {e}")
    return None

st.set_page_config(page_title="學生成績與公開試預測系統", layout="wide")
st.title("📊 學生成績追蹤、數據建模與公開試預測系統")

student_info_df = load_student_info()

main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs([
    "📚 初中/常規測考追蹤分析", 
    "🎓 中六公開試 (DSE) 成效分析", 
    "🔮 中六測驗 (T1A1) 升學潛質預測",
    "☀️ 中五 (未來中六)：大學 (332) 與大專 (222) 分析",
    "🤖 跨學年數據建模與下屆預測"
])

# 側邊欄設定
st.sidebar.header("⚙️ 常規測考門檻設定")
target_score = st.sidebar.number_input("常規：數據2平均分門檻", value=60.0)
core_pass = st.sidebar.number_input("常規：中英數及格線", value=60.0)
underperform_cap = st.sidebar.number_input("常規：進步/保底平均分上限", value=50.0)
progress_score = st.sidebar.number_input("常規：分數提升門檻", value=3.0)

st.sidebar.header("⚙️ 中五/未來中六門檻設定")
s5_u_score = st.sidebar.number_input("大學 (332) 平均分門檻", value=50.0)
s5_u_chi = st.sidebar.number_input("大學 (332) 中文門檻", value=45.0)
s5_u_eng = st.sidebar.number_input("大學 (332) 英文門檻", value=50.0)
s5_u_math = st.sidebar.number_input("大學 (332) 數學門檻", value=40.0)

s5_sub_score = st.sidebar.number_input("大專 (222) 平均分門檻", value=40.0)
s5_sub_chi = st.sidebar.number_input("大專 (222) 中文門檻", value=38.0)
s5_sub_eng = st.sidebar.number_input("大專 (222) 英文門檻", value=35.0)
s5_sub_math = st.sidebar.number_input("大專 (222) 數學門檻", value=35.0)

# ==================== 分頁一：初中/常規測考追蹤 ====================
with main_tab1:
    st.write("上傳 數據1 與 數據2 的 Excel 檔案，系統將自動為學生進行分類。")
    col1, col2 = st.columns(2)
    with col1: file1 = st.file_uploader("上傳 數據1 (Excel)", type=["xls", "xlsx"], key="f1")
    with col2: file2 = st.file_uploader("上傳 數據2 (Excel)", type=["xls", "xlsx"], key="f2")

    if file1 and file2:
        try:
            with st.spinner('資料處理中...'):
                df1 = pd.read_excel(file1)
                df2 = pd.read_excel(file2)
                merged = pd.merge(df1, df2, on=['*School Year', '*Class Level', '*Class', '*Class Number', '*Student Name', '*Reg. No.'], suffixes=('_D1', '_D2'))
                
                if student_info_df is not None:
                    merged['Reg_No_clean'] = merged['*Reg. No.'].astype(str).str.replace('#', '').str.strip()
                    merged = pd.merge(merged, student_info_df, left_on='Reg_No_clean', right_on='註冊編號_clean', how='left')
                
                prefix1, prefix2 = 'T2A1', 'T2A3'
                chi_name_col = '中文姓名' if '中文姓名' in merged.columns else None
                
                merged['Rank_D1'] = merged[f'{prefix1}_OMF'].apply(extract_rank)
                merged['Rank_D2'] = merged[f'{prefix2}_OMF'].apply(extract_rank)
                merged['Rank_Diff'] = merged['Rank_D1'] - merged['Rank_D2'] 
                
                for prefix, new_prefix in zip([prefix1, prefix2], ['D1', 'D2']):
                    merged[f'{new_prefix}_Score_clean'] = merged[f'{prefix}_Score'].apply(to_numeric).fillna(0)
                    merged[f'{new_prefix}_Math'] = merged[f'{prefix}_數學_C_Score'].apply(to_numeric).fillna(merged[f'{prefix}_數學_E_Score'].apply(to_numeric))
                    merged[f'{new_prefix}_Sci'] = merged[f'{prefix}_科初_C_Score'].apply(to_numeric).fillna(merged[f'{prefix}_科初_E_Score'].apply(to_numeric))
                    merged[f'{new_prefix}_Chi'] = merged[f'{prefix}_中文_C_Score'].apply(to_numeric)
                    merged[f'{new_prefix}_Eng'] = merged[f'{prefix}_英文_E_Score'].apply(to_numeric)
                    
                merged['Score_Diff_clean'] = merged['D2_Score_clean'] - merged['D1_Score_clean']

                cat1_idx = (merged['D2_Score_clean'] >= target_score) & (merged['D2_Chi'] >= core_pass) & (merged['D2_Eng'] >= core_pass) & (merged['D2_Math'] >= core_pass)
                cat1 = merged[cat1_idx].sort_values(['*Class', '*Class Number'])
                
                cat2_idx = (merged['D2_Score_clean'] > 0) & (merged['D2_Score_clean'] < underperform_cap) & (~cat1_idx) & ((merged['Score_Diff_clean'] >= progress_score) | (merged['Rank_Diff'] >= 6) | (merged['D2_Math'] >= 55))
                cat2 = merged[cat2_idx].sort_values(['*Class', '*Class Number'])
                
                cat3_idx = (merged['D2_Score_clean'] > 0) & (merged['D2_Score_clean'] < underperform_cap) & (~cat1_idx) & (~cat2_idx)
                cat3 = merged[cat3_idx].sort_values(['*Class', '*Class Number'])

                display_cols = ['*Class', '*Class Number']
                rename_dict = {'*Class': '班別', '*Class Number': '班號'}
                
                if chi_name_col:
                    display_cols.extend([chi_name_col, '*Student Name'])
                    rename_dict[chi_name_col] = '中文姓名'
                    rename_dict['*Student Name'] = '英文姓名'
                else:
                    display_cols.append('*Student Name')
                    rename_dict['*Student Name'] = '姓名'
                    
                display_cols.extend(['D2_Score_clean', f'{prefix2}_OMF', 'D2_Chi', 'D2_Eng', 'D2_Math', 'Score_Diff_clean'])
                rename_dict.update({'D2_Score_clean': '數據2平均分', f'{prefix2}_OMF': '全級名次(OMF)', 'D2_Chi': '中文', 'D2_Eng': '英文', 'D2_Math': '數學', 'Score_Diff_clean': '與數據1分差'})
                
                st.success("✅ 數據合併與分析成功！")
                tab1, tab2, tab3 = st.tabs(["🎓 潛質升大學名單", "📈 具進步空間名單", "🛟 保底支援名單"])
                with tab1: st.dataframe(cat1[display_cols].rename(columns=rename_dict), use_container_width=True, hide_index=True)
                with tab2: st.dataframe(cat2[display_cols].rename(columns=rename_dict), use_container_width=True, hide_index=True)
                with tab3: st.dataframe(cat3[display_cols].rename(columns=rename_dict), use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"錯誤: {e}")
    else: st.info("💡 請上傳兩個常規成績檔案。")

# ==================== 分頁二：DSE 成效分析 ====================
with main_tab2:
    st.subheader("🎓 中六模擬試 (T1A3) 與 DSE 公開試深度分析")
    col_m1, col_m2 = st.columns(2)
    with col_m1: mock_file = st.file_uploader("上傳中六模擬試成績 (Excel)", type=["xls", "xlsx"], key="mock_up")
    with col_m2: dse_file = st.file_uploader("上傳 DSE 公開試成績 (Excel)", type=["xls", "xlsx"], key="dse_up")
        
    if mock_file and dse_file:
        try:
            df_mock = pd.read_excel(mock_file)
            df_dse = pd.read_excel(dse_file)
            df_mock['Reg_Clean'] = df_mock['*Reg. No.'].astype(str).str.replace('#', '').str.strip()
            df_dse['Reg_Clean'] = df_dse['Registration No.'].astype(str).str.strip()
            
            merged_dse = pd.merge(df_mock, df_dse, on='Reg_Clean', how='inner')
            if student_info_df is not None:
                merged_dse = pd.merge(merged_dse, student_info_df, left_on='Reg_Clean', right_on='註冊編號_clean', how='left')
                
            merged_dse['DSE_Chi_Lvl'] = merged_dse['A010 Chinese'].apply(grade_to_level)
            merged_dse['DSE_Eng_Lvl'] = merged_dse['A020 English'].apply(grade_to_level)
            merged_dse['DSE_Math_Lvl'] = merged_dse['A030 Math Compulsory'].apply(grade_to_level)
            merged_dse['Met_U_Req'] = (merged_dse['DSE_Chi_Lvl'] >= 3) & (merged_dse['DSE_Eng_Lvl'] >= 3) & (merged_dse['DSE_Math_Lvl'] >= 2)
            
            st.success(f"✅ 成功配對 {len(merged_dse)} 位中六學生公開試數據！其中達到大學基本收生門檻（中3英3數2）共 {merged_dse['Met_U_Req'].sum()} 人。")
            st.dataframe(merged_dse[['Class', 'Class No.', 'Name', 'T1A3_Score', 'A010 Chinese', 'A020 English', 'A030 Math Compulsory', 'Met_U_Req']], use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"錯誤: {e}")

# ==================== 分頁三：中六 T1A1 測驗預測 ====================
with main_tab3:
    st.subheader("🔮 中六測驗 (T1A1) 後升學潛質預測")
    t1a1_next_file = st.file_uploader("上傳中六 T1A1 測驗成績 (Excel)", type=["xls", "xlsx"], key="t1a1_next")
    if t1a1_next_file:
        try:
            df_next = pd.read_excel(t1a1_next_file)
            if student_info_df is not None:
                df_next['Reg_No_clean'] = df_next['*Reg. No.'].astype(str).str.replace('#', '').str.strip()
                df_next = pd.merge(df_next, student_info_df, left_on='Reg_No_clean', right_on='註冊編號_clean', how='left')
            
            df_next['T1A1_Score_clean'] = pd.to_numeric(df_next['T1A1_Score'], errors='coerce').fillna(0)
            df_next['T1A1_Chi'] = pd.to_numeric(df_next['T1A1_中文_C_Score'], errors='coerce')
            df_next['T1A1_Eng'] = pd.to_numeric(df_next['T1A1_英文_E_Score'], errors='coerce')
            df_next['T1A1_Math'] = pd.to_numeric(df_next['T1A1_數必_C_Score'], errors='coerce').fillna(pd.to_numeric(df_next['T1A1_數必_E_Score'], errors='coerce'))
            
            def diagnose_student(row):
                s, c, e, m = row['T1A1_Score_clean'], row['T1A1_Chi'], row['T1A1_Eng'], row['T1A1_Math']
                notes = []
                if e < 50: notes.append("🚩 英文瓶頸")
                if c < 45: notes.append("🚩 中文瓶頸")
                if m < 40: notes.append("🚩 數學未達標")
                
                if s >= 50 and c >= 45 and e >= 50 and m >= 40: cat = "🎓 升大學穩健組"
                elif s >= 40 or (m >= 55) or (e >= 45) or (c >= 42): cat = "🎯 邊緣突破培訓組"
                else: cat = "🛟 基礎保底攻堅組"
                return pd.Series([cat, " | ".join(notes) if notes else "全科發展平衡"])
            
            df_next[['培訓類別', '診斷與急救建議']] = df_next.apply(diagnose_student, axis=1)
            st.dataframe(df_next[['*Class', '*Class Number', '*Student Name', 'T1A1_Score_clean', '培訓類別', '診斷與急救建議']], use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"錯誤: {e}")

# ==================== 分頁四：中五 (未來中六) 大學與大專分析 ====================
with main_tab4:
    st.subheader("☀️ 中五 (未來中六)：大學 (332) 與大專 (222) 潛質分析")
    s5_file = st.file_uploader("上傳中五 T2A3 年終成績 (Excel)", type=["xls", "xlsx"], key="s5_t2a3")
    if s5_file:
        try:
            df_s5 = pd.read_excel(s5_file)
            if student_info_df is not None:
                df_s5['Reg_No_clean'] = df_s5['*Reg. No.'].astype(str).str.replace('#', '').str.strip()
                df_s5 = pd.merge(df_s5, student_info_df, left_on='Reg_No_clean', right_on='註冊編號_clean', how='left')
            
            prefix_s5 = 'T2A3'
            df_s5['S5_Score_clean'] = pd.to_numeric(df_s5[f'{prefix_s5}_Score'], errors='coerce').fillna(0)
            df_s5['S5_Chi'] = pd.to_numeric(df_s5[f'{prefix_s5}_中文_C_Score'], errors='coerce')
            df_s5['S5_Eng'] = pd.to_numeric(df_s5[f'{prefix_s5}_英文_E_Score'], errors='coerce')
            math_col_c = f'{prefix_s5}_數學_C_Score' if f'{prefix_s5}_數學_C_Score' in df_s5.columns else f'{prefix_s5}_數必_C_Score'
            math_col_e = f'{prefix_s5}_數學_E_Score' if f'{prefix_s5}_數學_E_Score' in df_s5.columns else f'{prefix_s5}_數必_E_Score'
            df_s5['S5_Math'] = pd.to_numeric(df_s5[math_col_c], errors='coerce') if math_col_c in df_s5.columns else 0
            if math_col_e in df_s5.columns: df_s5['S5_Math'] = df_s5['S5_Math'].fillna(pd.to_numeric(df_s5[math_col_e], errors='coerce'))
                
            def s5_classify_and_diagnose(row):
                s, c, e, m = row['S5_Score_clean'], row['S5_Chi'], row['S5_Eng'], row['S5_Math']
                notes = []
                if e < s5_u_eng and e >= s5_sub_eng: notes.append("🚩 英文瓶頸 (衝擊3級)")
                elif e < s5_sub_eng: notes.append("⚠️ 英文急救 (補救至2級)")
                if c < s5_u_chi and c >= s5_sub_chi: notes.append("🚩 中文瓶頸 (衝擊3級)")
                elif c < s5_sub_chi: notes.append("⚠️ 中文急救 (補救至2級)")
                if m < s5_u_math: notes.append("🚩 數學未達標")
                
                if s >= s5_u_score and c >= s5_u_chi and e >= s5_u_eng and m >= s5_u_math: cat = "🎓 潛質入大學名單 (目標 332)"
                elif s >= s5_sub_score and c >= s5_sub_chi and e >= s5_sub_eng and m >= s5_sub_math: cat = "🏫 潛質入大專名單 (目標 222)"
                else: cat = "🛟 基礎保底加強組"
                return pd.Series([cat, " | ".join(notes) if notes else "全科均衡"])
            
            df_s5[['升學類別', '培訓與急救建議']] = df_s5.apply(s5_classify_and_diagnose, axis=1)
            st.dataframe(df_s5[['*Class', '*Class Number', '*Student Name', 'S5_Score_clean', 'S5_Chi', 'S5_Eng', 'S5_Math', '升學類別', '培訓與急救建議']], use_container_width=True, hide_index=True)
        except Exception as e: st.error(f"錯誤: {e}")

# ==================== 分頁五：跨學年機器學習建模與預測 (新功能) ====================
with main_tab5:
    st.subheader("🤖 本屆中六 (S4-S6) 機器學習數據建模與下屆升學概率預測")
    st.write("上傳本屆中六的中四至中六歷年成績與 DSE 數據訓練 AI 模型，並輸入下屆中六歷年數據進行自動預測。")
    
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1: m_t1a1 = st.file_uploader("1. 上傳本屆 S6 T1A1 測驗", type=["xls", "xlsx"], key="m_t1a1")
    with col_t2: m_t1a3 = st.file_uploader("2. 上傳本屆 S6 T1A3 模擬試", type=["xls", "xlsx"], key="m_t1a3")
    with col_t3: m_dse = st.file_uploader("3. 上傳本屆 DSE 成績", type=["xls", "xlsx"], key="m_dse")
    
    if m_t1a1 and m_t1a3 and m_dse:
        try:
            d1 = pd.read_excel(m_t1a1)
            d2 = pd.read_excel(m_t1a3)
            d3 = pd.read_excel(m_dse)
            
            d1['Reg_Clean'] = d1['*Reg. No.'].astype(str).str.replace('#', '').str.strip()
            d2['Reg_Clean'] = d2['*Reg. No.'].astype(str).str.replace('#', '').str.strip()
            d3['Reg_Clean'] = d3['Registration No.'].astype(str).str.strip()
            
            m_all = pd.merge(d1, d2, on='Reg_Clean', suffixes=('_T1A1', '_T1A3'))
            m_all = pd.merge(m_all, d3, on='Reg_Clean')
            
            m_all['DSE_Chi'] = m_all['A010 Chinese'].apply(grade_to_level)
            m_all['DSE_Eng'] = m_all['A020 English'].apply(grade_to_level)
            m_all['DSE_Math'] = m_all['A030 Math Compulsory'].apply(grade_to_level)
            m_all['Target_332'] = ((m_all['DSE_Chi'] >= 3) & (m_all['DSE_Eng'] >= 3) & (m_all['DSE_Math'] >= 2)).astype(int)
            
            # 建立訓練特徵
            m_all['T1A1_Score_num'] = pd.to_numeric(m_all['T1A1_Score'], errors='coerce')
            m_all['T1A1_Chi_num'] = pd.to_numeric(m_all['T1A1_中文_C_Score'], errors='coerce')
            m_all['T1A1_Eng_num'] = pd.to_numeric(m_all['T1A1_英文_E_Score'], errors='coerce')
            m_all['T1A1_Math_num'] = pd.to_numeric(m_all['T1A1_數必_C_Score'], errors='coerce').fillna(pd.to_numeric(m_all['T1A1_數必_E_Score'], errors='coerce'))
            
            m_all['T1A3_Score_num'] = pd.to_numeric(m_all['T1A3_Score'], errors='coerce')
            m_all['T1A3_Chi_num'] = pd.to_numeric(m_all['T1A3_中文_C_Score'], errors='coerce')
            m_all['T1A3_Eng_num'] = pd.to_numeric(m_all['T1A3_英文_E_Score'], errors='coerce')
            m_all['T1A3_Math_num'] = pd.to_numeric(m_all['T1A3_數必_C_Score'], errors='coerce').fillna(pd.to_numeric(m_all['T1A3_數必_E_Score'], errors='coerce'))
            
            m_all['Score_Diff'] = m_all['T1A3_Score_num'] - m_all['T1A1_Score_num']
            
            features = ['T1A1_Score_num', 'T1A1_Chi_num', 'T1A1_Eng_num', 'T1A1_Math_num', 
                        'T1A3_Score_num', 'T1A3_Chi_num', 'T1A3_Eng_num', 'T1A3_Math_num', 'Score_Diff']
            
            df_train = m_all.dropna(subset=features + ['Target_332'])
            
            clf = DecisionTreeClassifier(max_depth=3, random_state=42)
            clf.fit(df_train[features], df_train['Target_332'])
            
            st.success(f"🎉 數據模型訓練成功！訓練樣本數：{len(df_train)} 人，模型的關鍵特徵權重如下：")
            
            feat_imp = pd.DataFrame({'特徵指標': features, '對 DSE 332 的影響權重': clf.feature_importances_}).sort_values('對 DSE 332 的影響權重', ascending=False)
            st.dataframe(feat_imp, use_container_width=True, hide_index=True)
            
            st.divider()
            st.markdown("### 🔮 下載/套用模型預測下屆中六")
            next_cohort_file = st.file_uploader("4. 上傳下屆中六成績檔 (含測驗/模擬試分數)", type=["xls", "xlsx"], key="next_cohort")
            
            if next_cohort_file:
                df_pred = pd.read_excel(next_cohort_file)
                if student_info_df is not None:
                    df_pred['Reg_No_clean'] = df_pred['*Reg. No.'].astype(str).str.replace('#', '').str.strip()
                    df_pred = pd.merge(df_pred, student_info_df, left_on='Reg_No_clean', right_on='註冊編號_clean', how='left')
                
                # 自動填充預測特徵
                df_pred['T1A1_Score_num'] = pd.to_numeric(df_pred.get('T1A1_Score', 0), errors='coerce').fillna(0)
                df_pred['T1A1_Chi_num'] = pd.to_numeric(df_pred.get('T1A1_中文_C_Score', 0), errors='coerce').fillna(0)
                df_pred['T1A1_Eng_num'] = pd.to_numeric(df_pred.get('T1A1_英文_E_Score', 0), errors='coerce').fillna(0)
                df_pred['T1A1_Math_num'] = pd.to_numeric(df_pred.get('T1A1_數必_C_Score', 0), errors='coerce').fillna(0)
                
                df_pred['T1A3_Score_num'] = pd.to_numeric(df_pred.get('T1A3_Score', df_pred['T1A1_Score_num']), errors='coerce').fillna(0)
                df_pred['T1A3_Chi_num'] = pd.to_numeric(df_pred.get('T1A3_中文_C_Score', df_pred['T1A1_Chi_num']), errors='coerce').fillna(0)
                df_pred['T1A3_Eng_num'] = pd.to_numeric(df_pred.get('T1A3_英文_E_Score', df_pred['T1A1_Eng_num']), errors='coerce').fillna(0)
                df_pred['T1A3_Math_num'] = pd.to_numeric(df_pred.get('T1A3_數必_C_Score', df_pred['T1A1_Math_num']), errors='coerce').fillna(0)
                df_pred['Score_Diff'] = df_pred['T1A3_Score_num'] - df_pred['T1A1_Score_num']
                
                probs = clf.predict_proba(df_pred[features])[:, 1]
                df_pred['AI預測升大學概率 (%)'] = (probs * 100).round(1)
                
                def level_tag(p):
                    if p >= 70: return "🎓 高概率升學 (≥70%)"
                    elif p >= 30: return "🎯 邊緣衝刺 (30-69%)"
                    else: return "🛟 保底急救 (<30%)"
                    
                df_pred['升學概率分組'] = df_pred['AI預測升大學概率 (%)'].apply(level_tag)
                
                st.dataframe(df_pred[['*Class', '*Class Number', '*Student Name', 'AI預測升大學概率 (%)', '升學概率分組']], use_container_width=True, hide_index=True)
                
        except Exception as e:
            st.error(f"建模過程發生錯誤: {e}")
    else:
        st.info("💡 請上傳本屆中六的 3 份檔案（T1A1 測驗、T1A3 模擬試、DSE 成績）以自動訓練數據模型。")
