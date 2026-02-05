import subprocess
import sys
import os
import shlex


def run_command_and_tee(command, log_file_path, cwd):
    """
    执行一个 shell 命令，将其输出实时打印到 stdout，并同时写入日志文件。
    """

    header = f"""
{'=' * 80}
[RUNNER] 开始执行命令:
[RUNNER] {command}
[RUNNER] 工作目录: {cwd}
[RUNNER] 日志文件: {log_file_path}
{'=' * 80}
"""
    print(header)

    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 将 "python" 替换为 sys.executable，并添加 -u (无缓冲) 标志
    if command.startswith("python "):
        command_list = [sys.executable, '-u'] + shlex.split(command[7:])
    else:
        command_list = shlex.split(command)

    return_code = -1  # 默认为失败

    try:
        # 1. 打开日志文件准备写入 (模式 'w' 会自动覆盖旧文件)
        with open(log_file_path, 'w', encoding='utf-8') as log_f:
            log_f.write(header + "\n")

            try:
                # 启动子进程
                process = subprocess.Popen(
                    command_list,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0  # 0=unbuffered (配合 -u 使用)
                )

                # --- 逐字节读取以捕获 \r ---
                line_buffer = b""

                if process.stdout:
                    # 循环读取 1 个字节，直到进程结束 (b'' 被返回)
                    for byte in iter(lambda: process.stdout.read(1), b''):
                        line_buffer += byte

                        # 检查是否遇到了行尾（\n 或 \r）
                        if byte == b'\n' or byte == b'\r':
                            # 解码并写入
                            line_str = line_buffer.decode('utf-8', errors='replace')

                            # 实时打印到控制台
                            sys.stdout.write(line_str)
                            sys.stdout.flush()

                            # 写入日志文件
                            log_f.write(line_str)
                            log_f.flush()

                            # 重置缓冲区
                            line_buffer = b""

                # 进程结束后，处理可能遗留在缓冲区的最后一行
                if line_buffer:
                    line_str = line_buffer.decode('utf-8', errors='replace')
                    sys.stdout.write(line_str)
                    sys.stdout.flush()
                    log_f.write(line_str)
                    log_f.flush()
                # -------------------------------------

                process.wait()
                return_code = process.returncode

            except Exception as e:
                error_msg = f"\n[RUNNER] 脚本执行失败: {e}\n"
                print(error_msg)
                log_f.write(error_msg)
                return_code = -1

    except IOError as e:
        error_msg = f"\n[RUNNER] 严重错误: 无法打开日志文件 {log_file_path}. 错误: {e}\n"
        print(error_msg)
        return_code = -1

    # 以追加模式 ('a') 重新打开它来写入页脚
    footer = f"""
{'=' * 80}
[RUNNER] 命令执行完毕，返回码: {return_code}
[RUNNER] 日志已保存到: {log_file_path}
{'=' * 80}
"""
    print(footer)

    try:
        with open(log_file_path, 'a', encoding='utf-8') as log_f_append:
            log_f_append.write(footer)
    except IOError as e:
        print(f"[RUNNER] 警告: 无法写入页脚到日志文件. {e}")

    return return_code


def main():
    # 1. 定义路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    log_dir = os.path.join(project_root, 'train_result')

    # 定义 AdaLoGN 特征文件路径
    train_feats = "features_cache_nodes/adalogn_nodes_train.pt"
    val_feats = "features_cache_nodes/adalogn_nodes_val.pt"

    # 检查特征文件是否存在 (可选但推荐)
    train_feats_abs = os.path.join(project_root, train_feats)
    val_feats_abs = os.path.join(project_root, val_feats)
    if not os.path.exists(train_feats_abs) or not os.path.exists(val_feats_abs):
        print(f"[RUNNER] 警告: 未找到特征文件:\n  {train_feats_abs}\n  {val_feats_abs}")
        print("[RUNNER] 请先运行 extract_features-2.py 生成特征，否则模型将以基线模式运行。")
        # 您可以选择在这里 sys.exit(1) 如果不想跑基线

    # 2. 定义要运行的命令和对应的日志文件
    # [修改] 仅修改 logfile 的名称，加上 "UNFROZEN_" 前缀
    commands_config = [
        {
            "cmd": f"python tqa_dq_mc.py -r IR -s --train_feats {train_feats} --val_feats {val_feats}",
            "logfile": os.path.join(log_dir, "dmc_UNFROZEN_IR_training.txt")
        },
        {
            "cmd": f"python tqa_dq_mc.py -r NSP -s --train_feats {train_feats} --val_feats {val_feats}",
            "logfile": os.path.join(log_dir, "dmc_UNFROZEN_NSP_training.txt")
        },
        {
            "cmd": f"python tqa_dq_mc.py -r NN -s --train_feats {train_feats} --val_feats {val_feats}",
            "logfile": os.path.join(log_dir, "dmc_UNFROZEN_NN_training.txt")
        }
    ]

    # 3. 依次执行所有命令
    print(f"[RUNNER] 训练脚本启动... 项目根目录: {project_root}")
    for config in commands_config:
        return_code = run_command_and_tee(config["cmd"], config["logfile"], cwd=project_root)

        if return_code != 0:
            print(f"\n[RUNNER] 严重错误: 命令 '{config['cmd']}' 失败，返回码 {return_code}。")
            print("[RUNNER] 停止执行后续任务。请检查日志文件。")
            sys.exit(return_code)

    print("\n[RUNNER] 所有训练任务均已成功完成。")


if __name__ == "__main__":
    main()