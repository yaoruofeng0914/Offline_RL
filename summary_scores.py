import os

def process_best_scores(root_dir, target_filename):
    """
    遍历文件夹，寻找每个任务的最高分并汇总。
    支持 Task 名称中包含下划线的情况。
    """
    # 用于存储每个任务的最佳结果
    # 格式: { 'task_name': {'score': float, 'line_content': str} }
    best_records = {}

    print(f"正在遍历目录: {root_dir} ...")

    file_count = 0
    for subdir, _, files in os.walk(root_dir):
        if target_filename in files and os.path.basename(subdir) == "eval_clean":
            file_path = os.path.join(subdir, target_filename)
            file_count += 1
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # 格式: algorithm_task_name_extra: score (std)
                    try:
                        # 步骤 A: 分割 header 和 value
                        parts = line.split(':')
                        if len(parts) < 2:
                            continue
                        
                        header = parts[0].strip()
                        value_part = parts[1].strip()

                        # 步骤 B: 提取 task (关键修改点)
                        # 使用 split('_', 1) 只分割第一个下划线
                        # 例子: "AlgoA_Task_v1_beta" -> ["AlgoA", "Task_v1_beta"]
                        header_split = header.split('_', 1)
                        
                        if len(header_split) != 2:
                            # 如果没有下划线，说明格式不对
                            print(f"[警告] 格式异常 (找不到分隔用的下划线): {line}")
                            continue
                        
                        # header_split[0] 是 algorithm
                        # header_split[1] 是 task (包含了剩余所有的下划线)
                        task = header_split[1]

                        # 步骤 C: 提取分数 (float)
                        score_str = value_part.split('(')[0].strip()
                        score = float(score_str)

                        # 步骤 D: 比较并更新最高分
                        if task not in best_records:
                            best_records[task] = {
                                'score': score,
                                'line_content': line
                            }
                        else:
                            # 如果当前分数更高，则替换
                            if score > best_records[task]['score']:
                                best_records[task] = {
                                    'score': score,
                                    'line_content': line
                                }
                                
                    except ValueError:
                        print(f"[警告] 无法解析数值: {line} 在文件 {file_path}")
                        continue
                    except Exception as e:
                        print(f"[错误] 处理行时出错: {line} ({e})")
                        continue

            except Exception as e:
                print(f"[错误] 无法读取文件 {file_path}: {e}")

    # 2. 将结果写入输出文件
    if not best_records:
        print("未找到任何有效的分数记录。")
        return

    output_path = os.path.join(root_dir, target_filename)
    
    try:
        # 按任务名排序输出
        sorted_tasks = sorted(best_records.keys())
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for task in sorted_tasks:
                line_to_write = best_records[task]['line_content']
                f.write(line_to_write + '\n')
        
        print(f"-" * 30)
        print(f"处理完成！")
        print(f"共扫描文件数: {file_count}")
        print(f"提取任务数: {len(best_records)}")
        print(f"结果已保存至: {output_path}")
        
    except IOError as e:
        print(f"[错误] 无法写入输出文件: {e}")

if __name__ == "__main__":
    # 配置路径
    BASE_DIR = os.path.join("results", "baseline")
    TARGET_FILE1 = "all_best_score_processed.txt"
    TARGET_FILE2 = "all_best_50_score_processed.txt"
    TARGET_FILE3 = "all_final_score_processed.txt"

    if not os.path.exists(BASE_DIR):
        print(f"错误: 找不到目录 '{BASE_DIR}'。请确保你在正确的根目录下运行此脚本。")
    else:
        process_best_scores(BASE_DIR, TARGET_FILE1)
        process_best_scores(BASE_DIR, TARGET_FILE2)
        process_best_scores(BASE_DIR, TARGET_FILE3)