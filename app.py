# 專門根據成功考獲 DSE 332 大學門檻學生提煉『下四分位數保底入場線 (Q1 基準)』
def extract_ai_thresholds(clf, df_train, feats):
    successful_df = df_train[df_train['Target_332'] == 1]
    
    defaults = {
        'Avg_Score': 50.0,
        'Chi_Score': 50.0,
        'Eng_Score': 55.0,
        'Math_Score': 40.0
    }
    
    final_thresh = {}
    
    # 改用成功達 332 門檻學生的 Q1 (25th percentile)
    # 覆蓋 75% 成功學生，有效排除校內極端低分特例
    for f in feats:
        if not successful_df.empty and f in successful_df.columns:
            val = round(float(successful_df[f].quantile(0.25)), 1)
            final_thresh[f] = val
        else:
            final_thresh[f] = defaults[f]
            
    return final_thresh
