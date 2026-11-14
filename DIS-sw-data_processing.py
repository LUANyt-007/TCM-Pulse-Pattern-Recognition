import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, detrend
input_dir = "/Users/luan/Desktop/test"

output_dir = "/Users/luan/Desktop/processed_dataset_multiple_test"
os.makedirs(output_dir,exist_ok=True)
def preprocess_file(file_path):
    data = pd.read_csv(file_path,sep=r"\s+",header=None)
    press_data = data.iloc[:, :12]
    variance = press_data.var()
    cun = variance[[0, 1, 2]].idxmax()
    guan = variance[[3, 4, 5]].idxmax()
    chi = variance[[6, 7, 8]].idxmax()
    chixia = variance[[9, 10, 11]].idxmax()
    selected_channels = [cun,guan,chi,chixia]
    selected_data = press_data.iloc[
        :,
        selected_channels
    ].copy()
    n_rows = selected_data.shape[0]
    window_len = 11
    if n_rows < window_len:
        window_len = n_rows if (n_rows % 2 == 1) else n_rows - 1
        if window_len < 3:
            raise ValueError(f"样本过短，行数={n_rows}")
    # 3. Savitzky-Golay滤波去噪
    for col in selected_data.columns:
        selected_data[col] = savgol_filter(
            selected_data[col],
            window_length=11,
            polyorder=3
        )
    # 4. 去除基线漂移
    for col in selected_data.columns:
        selected_data[col] = detrend(selected_data[col])
    return selected_data.values
total_files = 0
success_cnt = 0
fail_cnt = 0

for file_name in os.listdir(input_dir):
    if not file_name.endswith(".txt"):
        continue
    total_files += 1
    file_path = os.path.join(input_dir, file_name)
    try:
        processed_data = preprocess_file(file_path)
        save_name = file_name.replace(".txt",".npy")
        np.save(os.path.join(output_dir, save_name), processed_data)
        success_cnt += 1
    except Exception as e:
        fail_cnt += 1
        print("处理失败:", file_path, e)

print(f"总数{total_files}，成功{success_cnt}，失败{fail_cnt}")
print("全部预处理完成")