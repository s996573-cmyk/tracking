import streamlit as st
import pandas as pd
import numpy as np
import os

# 處理文字名次
def extract_rank(val):
    if pd.isna(val) or not isinstance(val, str): return np.nan
    try: return int(val.split('#')[0])
    except: return np.nan

# 安全轉換數字
def to_numeric(val):
    try: return float(val)
    except: return np.nan

# DSE 等級轉為數值分數以便計算相關性與門檻
def grade_to_level(g):
    g_str = str(g).strip().upper()
    mapping = {'5**': 7, '5*': 6, '5': 5, '4': 4, '3': 3, '2': 2, '1': 1, 'U': 0, 'UNCLASSIFIED': 0}
    if g_str in mapping:
        return mapping[g_str]
    try:
        return float(g_str)
    except:
        return np.nan

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

st.set_page_config(page_title="學生成績與公開試分析系統", layout="wide")
st.title("📊 學生成績追蹤與公開試分析系統")

student_info_df = load_student_info()

main_tab1, main_tab2 = st.tabs(["📚 初中/常規測考追蹤分析", "🎓 中六公開試 (DSE) 成效、相關性與大學門檻分析"])

# === 分頁一：常規測考追蹤 ===
with main_tab1:
    st.write("上傳 數據1 與 數據2 的 Excel 檔案，系統將自動為學生進行分類。")
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("上傳 數據1 (Excel)", type=["xls", "xlsx"], key="f1")
    with col2:
        file2 = st.file_uploader("上傳 數據2 (Excel)", type=["xls", "xlsx"], key="f2")

    st.sidebar.header("⚙️ 篩選門檻設定 (常規)")
    target_score = st.sidebar.number_input("潛質大學：數據2平均分門檻", value=60.0)
    core_pass = st.sidebar.number_input("潛質大學：中英數及格線", value=60.0)
    underperform_cap = st.sidebar.number_input("進步/保底：平均分上限", value=50.0)
    progress_score = st.sidebar.number_input("進步：分數提升門檻", value=3.0)

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
                rename_dict.update({
                    'D2_Score_clean': '數據2平均分', f'{prefix2}_OMF': '全級名次(OMF)', 
                    'D2_Chi': '中文', 'D2_Eng': '英文', 'D2_Math': '數學', 'Score_Diff_clean': '與數據1分差'
                })
                
                st.success("✅ 數據合併與分析成功！")
                tab1, tab2, tab3 = st.tabs(["🎓 潛質升大學名單", "📈 具進步空間名單", "🛟 保底支援名單"])
                with tab1:
                    st.dataframe(cat1[display_cols].rename(columns=rename_dict), use_container_width=True, hide_index=True)
                with tab2:
                    st.dataframe(cat2[display_cols].rename(columns=rename_dict), use_container_width=True, hide_index=True)
                with tab3:
                    st.dataframe(cat3[display_cols].rename(columns=rename_dict), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"錯誤: {e}")
    else:
        st.info("💡 請上傳兩個常規成績檔案。")

# === 分頁二：DSE 成效與門檻對照 ===
with main_tab2:
    st.subheader("🎓 中六模擬試與 DSE 公開試深度分析（含大學入學門檻對照）")
    st.write("上傳中六模擬試成績 (如 2526_T1A3_s6.xlsx) 與實際公開試成績 (hkdse.xlsx)，系統將自動進行相關係數分析並檢核大學收生門檻（中 $\ge 3$、英 $\ge 3$、數 $\ge 2$）。")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        mock_file = st.file_uploader("上傳中六模擬試成績 (Excel)", type=["xls", "xlsx"], key="mock_up")
    with col_m2:
        dse_file = st.file_uploader("上傳 DSE 公開試成績 (Excel)", type=["xls", "xlsx"], key="dse_up")
        
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
            
            # 科目對應地圖：包含中文組（_C）與英文組（_E）自動補全機制
            subjects_map = [
                ('中文', ['T1A3_中文_C_Score'], 'A010 Chinese'),
                ('英文', ['T1A3_英文_E_Score'], 'A020 English'),
                ('數學必修', ['T1A3_數必_C_Score', 'T1A3_數必_E_Score'], 'A030 Math Compulsory'),
                ('數學 M1 (數1)', ['T1A3_數一_C_Score', 'T1A3_數一_E_Score'], 'A031 M1'),
                ('生物', ['T1A3_生物_C_Score', 'T1A3_生物_E_Score'], 'A130 Biology'),
                ('化學', ['T1A3_化學_C_Score', 'T1A3_化學_E_Score'], 'A140 Chemistry'),
                ('物理', ['T1A3_物理_C_Score', 'T1A3_物理_E_Score'], 'A150 Physics'),
                ('企業會計與財務概論 (企財/BAFS)', ['T1A3_企財_C_Score'], 'A171 BAFS(Accounting)'),
                ('資訊及通訊科技 (資通/ICT)', ['T1A3_資通_C_Score'], 'A200 ICT'),
                ('視覺藝術 (視憑/Visual Arts)', ['T1A3_視憑_C_Score'], 'A230 Visual Arts'),
                ('倫理與宗教 (倫教/ERS)', ['T1A3_倫教_C_Score'], 'A090 Ethics and Religious Studies'),
                ('經濟', ['T1A3_經濟_C_Score'], 'A080 Economics'),
                ('地理', ['T1A3_地理_C_Score'], 'A100 Geography'),
                ('歷史', ['T1A3_歷史_C_Score'], 'A110 History'),
                ('中史', ['T1A3_中史_C_Score'], 'A070 Chinese History')
            ]
            
            corr_results = []
            for name, m_cols, d_col in subjects_map:
                mock_series = None
                for mc in m_cols:
                    if mc in merged_dse.columns:
                        s = pd.to_numeric(merged_dse[mc], errors='coerce')
                        mock_series = s if mock_series is None else mock_series.fillna(s)
                
                if mock_series is not None and d_col in merged_dse.columns:
                    dse_series = merged_dse[d_col].apply(grade_to_level)
                    sub_c = pd.DataFrame({'Mock': mock_series, 'DSE': dse_series}).dropna()
                    if len(sub_c) > 2:
                        r = sub_c['Mock'].corr(sub_c['DSE'])
                        corr_results.append({'科目': name, '有效樣本數': len(sub_c), '相關係數 (r)': round(r, 3)})
            
            st.success(f"✅ 成功配對 {len(merged_dse)} 位中六學生的公開試與模擬試數據！其中達到大學基本收生要求（中3、英3、數2）共 **{merged_dse['Met_U_Req'].sum()}** 人。")
            
            st.markdown("### 📈 各主科模擬試成績與 DSE 實際等級相關係數")
            if corr_results:
                st.dataframe(pd.DataFrame(corr_results), use_container_width=True, hide_index=True)
            
            st.markdown("### 📋 學生模擬試表現與 DSE 大學門檻達標對照表")
            display_dse_cols = ['Class', 'Class No.']
            dse_rename = {'Class': '班別', 'Class No.': '班號'}
            
            if '中文姓名' in merged_dse.columns:
                display_dse_cols.append('中文姓名')
                dse_rename['中文姓名'] = '中文姓名'
            display_dse_cols.append('Name')
            dse_rename['Name'] = '英文姓名'
            
            display_dse_cols.extend(['T1A3_Score', 'A010 Chinese', 'A020 English', 'A030 Math Compulsory', 'Met_U_Req'])
            dse_rename.update({
                'T1A3_Score': '模擬試總平均分',
                'A010 Chinese': 'DSE中文',
                'A020 English': 'DSE英文',
                'A030 Math Compulsory': 'DSE數學',
                'Met_U_Req': '達大學基本門檻(中3英3數2)'
            })
            
            st.dataframe(merged_dse[display_dse_cols].rename(columns=dse_rename), use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"DSE 檔案處理發生錯誤: {e}")
    else:
        st.info("💡 請同時上傳中六模擬試成績與 hkdse.xlsx 檔案以啟動分析。")
