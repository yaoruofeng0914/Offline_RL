import os
import argparse
import pandas as pd
import shutil


def aggregate_results(base_dir):
    output_csv = "WT_RDT_final_highest_scores.csv"
    # 更改了总文件夹的名字，表示收集所有模型
    models_dir = "WT_RDT_collected_all_models"

    os.makedirs(models_dir, exist_ok=True)
    results = []

    print(f"🔍 正在扫描 '{base_dir}' 目录下的所有实验结果...")

    pth_count = 0  # 统计总共拷贝了多少个权重文件

    # 遍历所有子目录
    for root, dirs, files in os.walk(base_dir):
        # 只要目录下存在 best_score.txt，就说明这是一个跑完的实验
        if "best_score.txt" in files:
            path_parts = root.split(os.sep)

            # 提取 group_name
            group_name = next((p for p in path_parts if p.startswith("seed_")), None)
            if not group_name:
                continue

            # 解析配置信息
            _, seed_str, mode, tag = group_name.split("_")
            seed = int(seed_str)

            # 获取环境名
            group_idx = path_parts.index(group_name)
            env = path_parts[group_idx + 1]

            # 1. 提取最高分并记录到表格
            with open(os.path.join(root, "best_score.txt"), "r") as f:
                content = f.read().strip()
                try:
                    score_str, epoch_str = content.split("_")
                    score = float(score_str)
                    epoch = int(epoch_str)
                except ValueError:
                    continue

            results.append({
                "Environment": env,
                "Seed": seed,
                "Noise_Type": mode.capitalize(),
                "Attack_Type": tag,
                "Highest_Score": score,
                "Best_Epoch": epoch,
                "Original_Log_Path": root
            })

            # 2. 收集该局的所有 .pth 文件
            # 为当前 Case 创建一个专属的子文件夹，保持内部整洁
            case_specific_dir = os.path.join(models_dir, f"{env}_seed{seed}_{mode}_{tag}")
            os.makedirs(case_specific_dir, exist_ok=True)

            # 遍历当前目录下的所有文件，只要后缀是 .pth 就全部拷走
            for file in files:
                if file.endswith(".pth"):
                    src_model = os.path.join(root, file)
                    dest_model = os.path.join(case_specific_dir, file)
                    shutil.copy2(src_model, dest_model)
                    pth_count += 1

    # 转换为 DataFrame 并导出 CSV
    df = pd.DataFrame(results)
    if not df.empty:
        # 按环境和攻击类型排序，让表格更整洁美观
        df = df.sort_values(by=["Environment", "Noise_Type", "Attack_Type", "Seed"])
        df.to_csv(output_csv, index=False)
        print(f"✅ 成功！最高分统计已保存至表格: {output_csv}")
        print(f"📦 收集完毕！共提取了 {pth_count} 个 .pth 模型文件。")
        print(f"📂 所有模型已按 Case 分类存放在: {models_dir}/ 目录下。")
    else:
        print("❌ 未在目录中找到任何 best_score.txt 文件，请检查训练是否成功跑完。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="Full_Scale_Experiments")
    args = parser.parse_args()
    aggregate_results(args.base_dir)