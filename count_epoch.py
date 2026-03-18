import sys
import os
from pathlib import Path
from collections import Counter

def process_files(base_path, target_filename):
    """
    内部函数：专门用于统计特定文件名的 epoch 分布并打印表格
    """
    print(f"\n{'='*20} 正在统计: {target_filename} {'='*20}")
    
    epoch_counter = Counter()
    file_count = 0

    # 遍历查找指定的文件名
    for file_path in base_path.rglob(target_filename):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            if not content:
                continue

            # 解析格式 f"{score}_{epoch}"
            if '_' in content:
                _, epoch = content.rsplit('_', 1)
                epoch_counter[epoch] += 1
                file_count += 1
        except Exception:
            pass 

    # 如果没找到文件，直接返回
    if file_count == 0:
        print(f"  [提示] 未找到任何 {target_filename} 文件。")
        return

    # 开始打印结果
    print(f"  找到文件数: {file_count}")
    print(f"  {'-'*45}")
    # 格式说明
    print(f"  {'Epoch':<10} | {'频数':<8} | {'累计百分比'}")
    print(f"  {'-'*45}")

    # 排序逻辑 (数字优先)
    def sort_key(item):
        key = item[0]
        try:
            return int(key)
        except ValueError:
            return key

    sorted_epochs = sorted(epoch_counter.items(), key=sort_key)

    # 计算累计百分比并输出
    running_count = 0 
    for epoch, count in sorted_epochs:
        running_count += count 
        cumulative_percentage = (running_count / file_count) * 100
        
        # 打印行
        print(f"  {epoch:<10}: {count:<8} ({cumulative_percentage:.2f}%)")

def analyze_epochs():
    # 1. 检查命令行参数 (只需要文件夹名 A)
    if len(sys.argv) < 2:
        print("使用错误。")
        print("正确格式: python analyze_epochs.py <文件夹名称A>")
        return

    folder_a = sys.argv[1]
    base_path = Path("results") / folder_a
    
    if not base_path.exists():
        print(f"错误: 找不到路径 '{base_path}'")
        return

    print(f"开始扫描目录: {base_path}")

    # 2. 依次调用处理函数
    # 第一个：统计 best_score.txt
    process_files(base_path, "best_score.txt")
    
    # 第二个：统计 best_score_50.txt
    process_files(base_path, "best_score_50.txt")
    
    print(f"\n{'='*56}")
    print("所有统计结束。")

if __name__ == "__main__":
    analyze_epochs()