import pandas as pd
import numpy as np

def score_to_dse_level(score):
    """校內分數轉換為 HKDSE 等級"""
    if pd.isna(score): return 0
    if score >= 65: return 4
    elif score >= 52: return 3
    elif score >= 40: return 2
    elif score >= 30: return 1
    else: return 0

def cs_to_attained(score):
    """公民與社會發展科達標判定"""
    if pd.isna(score): return 'U'
    return 'A' if score >= 40 else 'U'

def analyze_student_performance(file_path):
    # 讀取數據 (假設包含 Chi, Eng, Math, CS, Elec1, Elec2 欄位)
    df = pd.read_excel(file_path)

    # 1. 轉化各科預測等級
    df['Chi_Lvl'] = df['Chi_Score'].apply(score_to_dse_level)
    df['Eng_Lvl'] = df['Eng_Score'].apply(score_to_dse_level)
    df['Math_Lvl'] = df['Math_Score'].apply(score_to_dse_level)
    df['CS_Lvl'] = df['CS_Score'].apply(cs_to_attained)
    df['Elec1_Lvl'] = df['Elec1_Score'].apply(score_to_dse_level)
    df['Elec2_Lvl'] = df['Elec2_Score'].apply(score_to_dse_level)

    # 2. 條件判定
    is_degree = (
        (df['Chi_Lvl'] >= 3) & 
        (df['Eng_Lvl'] >= 3) & 
        (df['Math_Lvl'] >= 2) & 
        (df['CS_Lvl'] == 'A') & 
        (df['Elec1_Lvl'] >= 2) & 
        (df['Elec2_Lvl'] >= 2)
    )

    is_subdegree = (
        (df['Chi_Lvl'] >= 2) & 
        (df['Eng_Lvl'] >= 2) & 
        (df['Math_Lvl'] >= 2) & 
        (df['CS_Lvl'] == 'A') & 
        (df['Elec1_Lvl'] >= 2) & 
        (df['Elec2_Lvl'] >= 2)
    )

    is_borderline = (
        (df['Chi_Lvl'] < 2) | 
        (df['Eng_Lvl'] < 2) | 
        (df['Math_Lvl'] < 2) | 
        (df['CS_Lvl'] == 'U')
    )

    # 3. 標籤分類
    conditions = [
        is_degree,
        (~is_degree & is_subdegree),
        is_borderline
    ]
    choices = ['潛質入大學 (332A22)', '潛質入大專 (222A22)', '保底求合格 (關鍵科未達標)']
    df['Status'] = np.select(conditions, choices, default='一般進步組')

    # 4. 分頁導出 Excel 報告
    with pd.ExcelWriter('DSE_Student_Analysis_Report.xlsx') as writer:
        df[df['Status'] == '潛質入大學 (332A22)'].to_excel(writer, sheet_name='潛質入大學', index=False)
        df[df['Status'] == '潛質入大專 (222A22)'].to_excel(writer, sheet_name='潛質入大專', index=False)
        df[df['Status'] == '保底求合格 (關鍵科未達標)'].to_excel(writer, sheet_name='保底求合格', index=False)
        df.to_excel(writer, sheet_name='全校總表', index=False)

    print("分析完成，報告已生成為 DSE_Student_Analysis_Report.xlsx")

# 執行腳本範例：
# analyze_student_performance('2526_T2A3_s6.xlsx')
