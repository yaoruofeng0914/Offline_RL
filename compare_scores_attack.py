import os
import argparse
import sys

# ================= 配置区域 =================
BASE_DIR = "results"

# 需要自动对比的文件列表
TARGET_FILES = [
    "all_best_score_processed.txt",
    "all_final_score_processed.txt",
    "all_best_50_score_processed.txt"
]
# ===========================================

def parse_score_file(file_path):
    """
    读取文件并解析为字典: {pure_task_name: score_float}
    逻辑：将 "Algo_TaskName" 解析为 "TaskName"，忽略 Algo 前缀。
    """
    scores = {}
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    if ":" not in line:
                        continue
                    
                    full_key_part, score_part = line.split(":", 1)
                    full_key = full_key_part.strip()
                    
                    # 忽略 Algo 前缀，只取第一个 "_" 之后的部分
                    if "_" in full_key:
                        _, task_name = full_key.split("_", 1)
                    else:
                        task_name = full_key

                    score_str = score_part.strip().split("(")[0].strip()
                    score = float(score_str)
                    
                    scores[task_name] = score
                    
                except ValueError:
                    continue
    except Exception as e:
        print(f"[!] 读取错误 {file_path}: {e}")
        return None
                
    return scores

def compare_and_write(f_out, filename, exp_a, sub_exp_a, exp_b, sub_exp_b, data_a, data_b):
    """
    对比逻辑：
    1. 将详细结果写入 txt 文件 (f_out)
    2. 在命令行打印简要统计 (print)
    """
    
    # --- 1. 计算逻辑 ---
    common_tasks = sorted(list(set(data_a.keys()) & set(data_b.keys())))
    total_comparisons = len(common_tasks)
    
    a_wins = []
    b_wins = []

    for task in common_tasks:
        score_a = data_a[task]
        score_b = data_b[task]

        if score_a > score_b:
            a_wins.append(task)
        elif score_b > score_a:
            b_wins.append(task)
        else:
            # 平局
            a_wins.append(task)
            b_wins.append(task)

    # --- 2. 写入文件 (详细报告) ---
    header = f"FILE: {filename}"
    f_out.write("=" * 50 + "\n")
    f_out.write(f"{header:^50}\n")
    f_out.write("=" * 50 + "\n")

    if total_comparisons == 0:
        f_out.write("Result: No common tasks found.\n\n\n")
        # 命令行也提示一下
        print(f"[{filename}] 未找到共同任务 (Total: 0)")
        print("-" * 30)
        return

    f_out.write(f"Total common tasks: {total_comparisons}\n")
    # f_out.write(f"{exp_a}-{sub_exp_a} wins: {len(a_wins)}\n")
    f_out.write(f"{exp_a} wins: {len(a_wins)}\n")
    # f_out.write(f"{exp_b}-{sub_exp_b} wins: {len(b_wins)}\n")
    f_out.write(f"{exp_b} wins: {len(b_wins)}\n")
    f_out.write("-" * 50 + "\n")

    # f_out.write(f"\n[ {exp_a}-{sub_exp_a} Better or Equal ]\n")
    f_out.write(f"\n[ {exp_a} Better or Equal ]\n")
    if a_wins:
        for task in a_wins:
            f_out.write(f"{task}\n")
    else:
        f_out.write("(None)\n")

    # f_out.write(f"\n[ {exp_b}-{sub_exp_b} Better or Equal ]\n")
    f_out.write(f"\n[ {exp_b} Better or Equal ]\n")
    if b_wins:
        for task in b_wins:
            f_out.write(f"{task}\n")
    else:
        f_out.write("(None)\n")
    
    f_out.write("\n\n")

    # --- 3. 命令行输出 (简要统计) ---
    print(f"[{filename}]")
    print(f"  > 共比较任务: {total_comparisons}")
    # print(f"  > {exp_a}-{sub_exp_a} 胜出: {len(a_wins)}")
    print(f"  > {exp_a} 胜出: {len(a_wins)}")
    # print(f"  > {exp_b}-{sub_exp_b} 胜出: {len(b_wins)}")
    print(f"  > {exp_b} 胜出: {len(b_wins)}")
    print("-" * 30)

def main():
    parser = argparse.ArgumentParser(description="批量比较 results 下三个结果文件")
    parser.add_argument("exp_a", type=str, help="实验A (例如: A)")
    # parser.add_argument("sub_exp_a", type=str, help="实验A (例如: A)")
    parser.add_argument("exp_b", type=str, help="实验B (例如: B)")
    # parser.add_argument("sub_exp_b", type=str, help="实验A (例如: A)")

    args = parser.parse_args()
    exp_name_a = args.exp_a
    # sub_exp_a = f"eval_{args.sub_exp_a}"
    sub_exp_a = "eval_clean"
    exp_name_b = args.exp_b
    # sub_exp_b = f"eval_{args.sub_exp_b}"
    sub_exp_b = "eval_clean"

    # output_filename = f"{exp_name_a}-{sub_exp_a}-{exp_name_b}-{sub_exp_b}.txt"
    output_filename = f"{exp_name_a}-{exp_name_b}.txt"
    output_path = os.path.join(BASE_DIR, output_filename)

    print("=" * 40)
    # print(f"对比开始: {exp_name_a}-{sub_exp_a} vs {exp_name_b}-{sub_exp_b}")
    print(f"对比开始: {exp_name_a} vs {exp_name_b}")
    print(f"详细报告将写入: {output_path}")
    print("=" * 40 + "\n")

    try:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            # f_out.write(f"Comparisons Report: {exp_name_a}-{sub_exp_a} vs {exp_name_b}-{sub_exp_b}\n")
            f_out.write(f"Comparisons Report: {exp_name_a} vs {exp_name_b}\n")
            f_out.write(f"Note: Ignoring 'Algo_' prefix for task matching.\n\n")

            for target_file in TARGET_FILES:
                path_a = os.path.join(BASE_DIR, exp_name_a, sub_exp_a, target_file)
                path_b = os.path.join(BASE_DIR, exp_name_b, sub_exp_b, target_file)

                data_a = parse_score_file(path_a)
                data_b = parse_score_file(path_b)

                # 检查文件缺失情况
                if data_a is None or data_b is None:
                    # 文件写入记录
                    f_out.write("=" * 50 + "\n")
                    f_out.write(f"FILE: {target_file}\n")
                    f_out.write("=" * 50 + "\n")
                    f_out.write("Skipped: File missing.\n\n\n")
                    
                    # 命令行记录
                    print(f"[{target_file}]")
                    print("  > 跳过 (文件缺失)")
                    print("-" * 30)
                    continue
                
                # 执行对比
                compare_and_write(f_out, target_file, exp_name_a, sub_exp_a, exp_name_b, sub_exp_b, data_a, data_b)

        print(f"\n[Done] 所有对比完成。")

    except IOError as e:
        print(f"\n[!] 写入文件失败: {e}")

if __name__ == "__main__":
    main()