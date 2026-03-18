import os
import argparse
import sys
import argparse
import statistics

def collect_scores(root_dir, tag):
    # 定义输出文件路径：在目标文件夹根目录下
    output_file_name = f"all_{tag}_score.txt"
    output_path = os.path.join(root_dir, "eval_clean")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, output_file_name)
    
    results = []
    
    # 简单的计数器
    count = 0

    print(f"正在扫描目录: {root_dir}")

    # os.walk 递归遍历
    for current_root, dirs, files in os.walk(root_dir):
        if tag == "best_50":
            file_name = "best_score_50.txt"
        else:
            file_name = f"{tag}_score.txt"

        if file_name in files and "eval_" not in os.path.basename(current_root):
            file_path = os.path.join(current_root, file_name)   
            try:
                # 1. 读取分数
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # 尝试转换浮点数，确保格式正确
                    if "_" in content:
                        content = content.split("_")[0]
                    score = float(content)

                # 2. 处理文件夹名称
                folder_name = os.path.basename(current_root)
                
                # 按 "_" 分割，取前5部分
                parts = folder_name.split('_')
                identifier_parts = parts[:5]
                identifier = "_".join(identifier_parts)

                # 3. 格式化并添加
                result_line = f"{identifier}: {score}"
                results.append(result_line)
                count += 1

            except ValueError:
                print(f"[警告] 跳过无效数据: {file_path}")
            except Exception as e:
                print(f"[错误] 处理 {file_path} 失败: {e}")

    # 4. 排序（按文件夹名字母序）
    results.sort()

    # 5. 写入文件
    if count > 0:
        try:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for line in results:
                    f_out.write(line + "\n")
            print(f"---")
            print(f"处理完成！共汇总 {count} 条数据。")
            print(f"结果已保存至: {output_path}")
        except IOError as e:
            print(f"写入结果文件失败: {e}")
    else:
        print(f"未找到任何包含有效数据的txt文件。")

def calculate_and_write(f_out, config, scores):
    if not scores:
        return
    mean_val = statistics.mean(scores)
    std_val = statistics.stdev(scores) if len(scores) > 1 else 0.0
    f_out.write(f"{config}: {mean_val:.2f} ({std_val:.2f})\n")

def process_directory(folder_path):
    folder_path = os.path.join(folder_path, "eval_clean")
    if not os.path.isdir(folder_path):
        print(f"Error: Directory '{folder_path}' not found.")
        return

    files_found = 0

    for filename in os.listdir(folder_path):
        if filename.startswith("all_") and filename.endswith(".txt") and "_processed" not in filename:
            output_filename = filename.replace(".txt", "_processed.txt")
            files_found += 1
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(folder_path, output_filename)

            print(f"Processing {filename}...")

            try:
                with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
                    current_config = None
                    current_scores = []

                    for line in f_in:
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            identifier, score_str = line.split(': ')
                            config = identifier.rsplit('_', 1)[0]
                            score = float(score_str)

                            if config != current_config:
                                if current_config is not None:
                                    calculate_and_write(f_out, current_config, current_scores)
                                current_config = config
                                current_scores = [score]
                            else:
                                current_scores.append(score)
                        except ValueError:
                            continue

                    if current_config is not None:
                        calculate_and_write(f_out, current_config, current_scores)
                
                print(f"Saved to {output_filename}")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    
    if files_found == 0:
        print("No matching files found.")
    else:
        print("Done.")

if __name__ == "__main__":
    # 初始化参数解析器
    parser = argparse.ArgumentParser(description=f"提取子文件夹中的 *_score.txt 并汇总。")
    
    # 添加位置参数 'folder'
    parser.add_argument("folder", type=str, help="包含各级子文件夹的目标目录路径 (例如 result)")

    # 解析参数
    args = parser.parse_args()
    
    target_dir = args.folder
    target_dir = os.path.join("results", target_dir)  # 将输入路径与 "results" 文件夹结合

    # 检查路径有效性
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        collect_scores(target_dir, "final")
        collect_scores(target_dir, "best")
        collect_scores(target_dir, "best_50")
        process_directory(target_dir)
    else:
        print(f"错误: 路径 '{target_dir}' 不存在或不是一个文件夹。")
        sys.exit(1)