import streamlit as st
import pandas as pd
import numpy as np

# 處理文字名次 (例如將 "1#114" 轉換為數字 1)
def extract_rank(val):
    if pd.isna(val) or not isinstance(val, str): return np.nan
    try: return int(val.split('#')[0])
    except: return np.nan

# 安全地轉換分數為數字
def to_numeric(val):
    try: return float(val)
    except: return np.nan

# 設定網頁標題與排版
st.set_page_config(page_title="學生成績分析系統", layout="wide")
st.title("📊 學生成績追蹤與分析系統")
st.write("上傳 T2A1 (測驗) 與 T2A3 (考試) 的 Excel 檔案，系統將自動為學生進行分類。")

# 建立兩個上傳區塊
col1, col2 = st.columns(2)
with col1:
    file1 = st.file_uploader("上傳 T2A1 測驗成績 (Excel)", type=["xls", "xlsx"])
with col2:
    file3 = st.file_uploader("上傳 T2A3 考試成績 (Excel)", type=["xls", "xlsx"])

# 側邊欄：參數設定，保留未來擴充與調整的彈性
st.sidebar.header("⚙️ 篩選門檻設定")
target_score = st.sidebar.number_input("潛質大學：考試總分門檻", value=55.0)
core_pass = st.sidebar.number_input("潛質大學：中英數及格線", value=45.0)
underperform_cap = st.sidebar.number_input("進步/保底：總分上限", value=50.0)
progress_score = st.sidebar.number_input("進步：分數提升門檻", value=3.0)

# 當兩個檔案都上傳後，開始執行分析
if file1 and file3:
    try:
        with st.spinner('資料處理中...'):
            df1 = pd.read_excel(file1)
            df3 = pd.read_excel(file3)
            
            # 以學號等唯一資訊合併兩次成績
            merged = pd.merge(df1, df3, on=['*School Year', '*Class Level', '*Class', '*Class Number', '*Student Name', '*Reg. No.'], suffixes=('_T2A1', '_T2A3'))
            
            # 資料清洗與預處理
            merged['Rank_T2A1'] = merged['T2A1_OMF'].apply(extract_rank)
            merged['Rank_T2A3'] = merged['T2A3_OMF'].apply(extract_rank)
            merged['Rank_Diff'] = merged['Rank_T2A1'] - merged['Rank_T2A3'] 
            
            for prefix in ['T2A1', 'T2A3']:
                merged[f'{prefix}_Score_clean'] = merged[f'{prefix}_Score'].apply(to_numeric).fillna(0)
                # 兼容中文或英文應考的數學及科學科
                merged[f'{prefix}_Math'] = merged[f'{prefix}_數學_C_Score'].apply(to_numeric).fillna(merged[f'{prefix}_數學_E_Score'].apply(to_numeric))
                merged[f'{prefix}_Sci'] = merged[f'{prefix}_科初_C_Score'].apply(to_numeric).fillna(merged[f'{prefix}_科初_E_Score'].apply(to_numeric))
                merged[f'{prefix}_Chi'] = merged[f'{prefix}_中文_C_Score'].apply(to_numeric)
                merged[f'{prefix}_Eng'] = merged[f'{prefix}_英文_E_Score'].apply(to_numeric)
                
            merged['Score_Diff_clean'] = merged['T2A3_Score_clean'] - merged['T2A1_Score_clean']

            # --- 核心篩選邏輯 ---
            # 1. 潛質升大學
            cat1_idx = (merged['T2A3_Score_clean'] >= target_score) & (merged['T2A3_Chi'] >= core_pass) & (merged['T2A3_Eng'] >= core_pass) & (merged['T2A3_Math'] >= core_pass)
            cat1 = merged[cat1_idx].sort_values(['*Class', '*Class Number'])
            
            # 2. 有機會進步
            cat2_idx = (merged['T2A3_Score_clean'] > 0) & (merged['T2A3_Score_clean'] < underperform_cap) & (~cat1_idx) & ((merged['Score_Diff_clean'] >= progress_score) | (merged['Rank_Diff'] >= 6) | (merged['T2A3_Math'] >= 55))
            cat2 = merged[cat2_idx].sort_values(['*Class', '*Class Number'])
            
            # 3. 保底
            cat3_idx = (merged['T2A3_Score_clean'] > 0) & (merged['T2A3_Score_clean'] < underperform_cap) & (~cat1_idx) & (~cat2_idx)
            cat3 = merged[cat3_idx].sort_values(['*Class', '*Class Number'])

            # 設定表格要在畫面上顯示的欄位
            display_cols = ['*Class', '*Class Number', '*Student Name', 'T2A3_Score_clean', 'T2A3_Chi', 'T2A3_Eng', 'T2A3_Math', 'Score_Diff_clean']
            rename_dict = {'*Class': '班別', '*Class Number': '班號', '*Student Name': '姓名', 'T2A3_Score_clean':'考試總分', 'T2A3_Chi': '中文', 'T2A3_Eng': '英文', 'T2A3_Math': '數學', 'Score_Diff_clean':'與測驗分差'}
            
            st.success("✅ 數據合併與分析成功！")
            
            # 建立三個分頁來展示資料
            tab1, tab2, tab3 = st.tabs(["🎓 潛質升大學名單", "📈 具進步空間名單", "🛟 保底支援名單"])
            
            with tab1:
                st.subheader(f"潛質升大學同學 (共 {len(cat1)} 人)")
                st.dataframe(cat1[display_cols].rename(columns=rename_dict), use_container_width=True)
                
            with tab2:
                st.subheader(f"具進步空間同學 (共 {len(cat2)} 人)")
                st.dataframe(cat2[display_cols].rename(columns=rename_dict), use_container_width=True)
                
            with tab3:
                st.subheader(f"保底支援同學 (共 {len(cat3)} 人)")
                st.dataframe(cat3[display_cols].rename(columns=rename_dict), use_container_width=True)

    except Exception as e:
        st.error(f"檔案處理發生錯誤，請確認上傳的格式是否正確。詳細錯誤: {e}")
else:
    st.info("💡 請在上方上傳兩個 Excel 檔案以開始分析。")