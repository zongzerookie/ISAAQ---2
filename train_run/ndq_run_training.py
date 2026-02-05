import subprocess
import sys
import os
import shlex


def run_command_and_tee(command, log_file_path, cwd):
    """
    执行一个 shell 命令，将其输出实时打印到 stdout，并同时写入日志文件。
    (逻辑复用自 dmc_run_training.py，支持 Python 3.6+ 及 tqdm 进度条捕获)

    参数:
    command (str): 要执行的完整 shell 命令字符串。
    log_file_path (str): 保存日志的文件路径。
    cwd (str): 命令的执行工作目录 (即项目根目录)。

    返回:
    int: 命令的返回码 (0 表示成功)。
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

    # 将 "python" 替换为 sys.executable (确保使用当前虚拟环境)，并添加 -u (无缓冲) 标志
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

                # --- 逐字节读取以捕获 \r (进度条关键逻辑) ---
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
    project_root = os.path.dirname(script_dir)  # 向上两级找到项目根目录
    log_dir = os.path.join(project_root, 'train_result')

    # 2. 定义要运行的命令列表 (NDQ 任务)
    # 参数说明:
    # -r: 检索方式 (IR, NSP, NN)
    # -s: 保存模型
    # -b 2: 物理 Batch Size
    # -x 128: 最大序列长度
    # -acc 8: 梯度累积步数 (等效 Batch Size = 16)
    commands_config = [
        {
            "cmd": "python tqa_ndq_mc.py -r IR -s -b 2 -x 128 -acc 8",
            "logfile": os.path.join(log_dir, "ndq_IR_training.txt")
        },
        {
            "cmd": "python tqa_ndq_mc.py -r NSP -s -b 2 -x 128 -acc 8",
            "logfile": os.path.join(log_dir, "ndq_NSP_training.txt")
        },
        {
            "cmd": "python tqa_ndq_mc.py -r NN -s -b 2 -x 128 -acc 8",
            "logfile": os.path.join(log_dir, "ndq_NN_training.txt")
        }
    ]

    # 3. 依次执行所有命令
    print(f"[RUNNER] AdaLoGN NDQ 自动化训练脚本启动...")
    print(f"[RUNNER] 项目根目录: {project_root}")

    for config in commands_config:
        return_code = run_command_and_tee(config["cmd"], config["logfile"], cwd=project_root)

        if return_code != 0:
            print(f"\n[RUNNER] 严重错误: 命令 '{config['cmd']}' 失败，返回码 {return_code}。")
            print("[RUNNER] 停止执行后续任务。请检查日志文件。")
            sys.exit(return_code)

    print("\n[RUNNER] 所有 NDQ (IR, NSP, NN) 训练任务均已成功完成。")


if __name__ == "__main__":
    main()