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

# DSE 等級轉為數值分數以便計算相關性
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

# 設定網頁標題與排版
st.set_page_config(page_title="學生成績與公開試分析系統", layout="wide")
st.title("📊 學生成績追蹤與公開試分析系統")

student_info_df = load_student_info()

# 建立多分頁架構：支援初中追蹤與中六 DSE 分析
main_tab1, main_tab2 = st.tabs(["📚 初中/常規測考追蹤分析", "🎓 中六公開試 (DSE) 成效分析"])

with main_tab1:
    st.write("上傳 數據1 與 數據2 的 Excel 檔案，系統將自動為學生進行分類。")
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("上傳 數據1 (Excel)", type=["xls", "xlsx"], key="f1")
    with col2:
        file2 = st.file_uploader("上傳 數據2 (Excel)", type=["xls", "xlsx"], key="f2")

    st.sidebar.header("⚙️ 篩選門檻設定 (常規)")
    target_score = st.sidebar.number_input("潛質大學：數據2平均分門檻", value=55.0)
    core_pass = st.sidebar.number_input("潛質大學：中英數及格線", value=45.0)
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

with main_tab2:
    st.subheader("🎓 中六模擬試與香港中學文憑考試 (DSE) 關聯分析")
    st.write("上傳中六模擬試成績 (如 2526_T1A3_s6.xlsx) 與實際公開試成績 (hkdse.xlsx)，系統將自動對比並分析預測準確度。")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        mock_file = st.file_uploader("上傳中六模擬試成績 (Excel)", type=["xls", "xlsx"], key="mock_up")
    with col_m2:
        dse_file = st.file_uploader("上傳 DSE 公開試成績 (Excel)", type=["xls", "xlsx"], key="dse_up")
        
    if mock_file and dse_file:
        try:
            df_mock = pd.read_excel(mock_file)
            df_dse = pd.read_excel(dse_file)
            
            # 清洗學號以作配對
            df_mock['Reg_Clean'] = df_mock['*Reg. No.'].astype(str).str.replace('#', '').str.strip()
            df_dse['Reg_Clean'] = df_dse['Registration No.'].astype(str).str.strip()
            
            merged_dse = pd.merge(df_mock, df_dse, on='Reg_Clean', how='inner')
            
            if student_info_df is not None:
                merged_dse = pd.merge(merged_dse, student_info_df, left_on='Reg_Clean', right_on='註冊編號_clean', how='left')
                
            # 轉換 DSE 等級
            merged_dse['DSE_Chi_Lvl'] = merged_dse['A010 Chinese'].apply(grade_to_level)
            merged_dse['DSE_Eng_Lvl'] = merged_dse['A020 English'].apply(grade_to_level)
            merged_dse['DSE_Math_Lvl'] = merged_dse['A030 Math Compulsory'].apply(grade_to_level)
            
            # 模擬試分數
            merged_dse['Mock_Overall'] = pd.to_numeric(merged_dse['T1A3_Score'], errors='coerce')
            merged_dse['Mock_Chi'] = pd.to_numeric(merged_dse['T1A3_中文_C_Score'], errors='coerce')
            merged_dse['Mock_Eng'] = pd.to_numeric(merged_dse['T1A3_英文_E_Score'], errors='coerce')
            merged_dse['Mock_Math'] = pd.to_numeric(merged_dse['T1A3_數必_C_Score'], errors='coerce').fillna(pd.to_numeric(merged_dse['T1A3_數必_E_Score'], errors='coerce'))
            
            st.success(f"✅ 成功配對 {len(merged_dse)} 位中六學生的公開試與模擬試數據！")
            
            # 顯示核心科目相關係數分析
            st.markdown("### 📈 模擬試分數與 DSE 實際等級相關性報告")
            c1, c2, c3 = st.columns(3)
            with c1:
                chi_corr = merged_dse['Mock_Chi'].corr(merged_dse['DSE_Chi_Lvl'])
                st.metric("中國語文相關度", f"{chi_corr:.2f}")
            with c2:
                eng_corr = merged_dse['Mock_Eng'].corr(merged_dse['DSE_Eng_Lvl'])
                st.metric("英國語文相關度", f"{eng_corr:.2f}")
            with c3:
                math_corr = merged_dse['Mock_Math'].corr(merged_dse['DSE_Math_Lvl'])
                st.metric("數學必修科相關度", f"{math_corr:.2f}")
                
            # 呈現詳細對比清單
            st.markdown("### 📋 學生個人模擬試與 DSE 成績對照表")
            display_dse_cols = ['Class', 'Class No.']
            dse_rename = {'Class': '班別', 'Class No.': '班號'}
            
            if '中文姓名' in merged_dse.columns:
                display_dse_cols.append('中文姓名')
                dse_rename['中文姓名'] = '中文姓名'
            display_dse_cols.append('Name')
            dse_rename['Name'] = '英文姓名'
            
            display_dse_cols.extend(['Mock_Overall', 'A010 Chinese', 'A020 English', 'A030 Math Compulsory'])
            dse_rename.update({
                'Mock_Overall': '模擬試總平均分',
                'A010 Chinese': 'DSE中文等級',
                'A020 English': 'DSE英文等級',
                'A030 Math Compulsory': 'DSE數學等級'
            })
            
            st.dataframe(merged_dse[display_dse_cols].rename(columns=dse_rename), use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"DSE 檔案處理發生錯誤: {e}")
    else:
        st.info("💡 請同時上傳中六模擬試成績與 hkdse.xlsx 檔案以啟動分析。")
