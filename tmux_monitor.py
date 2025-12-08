#!/usr/bin/env python3
"""
TMUX窗口监控脚本
功能：监控5个tmux窗口，如果发现窗口空闲则自动启动wandb agent
"""

import subprocess
import time
from datetime import datetime
import os

# 配置
TMUX_WINDOWS = ["sweep2", "sweep4", "sweep5","sweep6","sweep8"]
CHECK_INTERVAL = 30  # 检查间隔（秒）
LOG_FILE = "log.txt"
SWEEP_ID = "kuangpenghao-shanghaitech-university/pt-scaling-hypers/wnhcj0p7"

# SLURM命令模板
SLURM_COMMAND = (
    "srun -N 1 -n 1 -X -u -p normal --gres=gpu:1 -c 2 --mem=1M -t 0-96:00:00 "
    f"wandb agent {SWEEP_ID}"
)


def check_tmux_window_active(window_name):
    """
    检查指定的tmux窗口是否有进程正在运行
    
    Args:
        window_name: tmux窗口名称
    
    Returns:
        tuple: (bool, str) - (是否有进程运行, 当前命令)
    """
    try:
        # 获取窗口的pane列表和进程信息
        cmd = ["tmux", "list-panes", "-t", window_name, "-F", "#{pane_current_command}"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            # 窗口不存在或其他错误
            error_msg = result.stderr.strip()
            print(f"⚠️  窗口 {window_name} 访问失败: {error_msg}")
            return None, error_msg
        
        # 获取当前运行的命令
        current_command = result.stdout.strip()
        
        # 如果只是bash/zsh等shell，说明没有实际任务运行
        idle_commands = ["bash", "zsh", "sh", "fish", "tcsh", "csh"]
        if current_command in idle_commands:
            return False, current_command
        
        # 有其他进程在运行
        return True, current_command
        
    except subprocess.TimeoutExpired:
        print(f"⚠️  检查窗口 {window_name} 超时")
        return None, "timeout"
    except Exception as e:
        print(f"❌ 检查窗口 {window_name} 出错: {str(e)}")
        return None, str(e)


def send_command_to_window(window_name, command):
    """
    向指定的tmux窗口发送命令
    
    Args:
        window_name: tmux窗口名称
        command: 要执行的命令
    
    Returns:
        bool: True表示发送成功，False表示失败
    """
    try:
        # 使用tmux send-keys发送命令 - 使用列表形式避免shell转义问题
        cmd = ["tmux", "send-keys", "-t", window_name, command, "C-m"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return True
        else:
            error_msg = result.stderr.strip()
            print(f"❌ 发送命令到 {window_name} 失败: {error_msg}")
            log_message(f"Error sending command to {window_name}: {error_msg}")
            return False
            
    except Exception as e:
        print(f"❌ 发送命令到 {window_name} 异常: {str(e)}")
        log_message(f"Error sending command to {window_name}: {str(e)}")
        return False


def log_message(message):
    """
    写入日志到log.txt
    
    Args:
        message: 日志消息
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    
    try:
        with open(LOG_FILE, "a") as f:
            f.write(log_entry)
        print(log_entry.strip())  # 同时打印到控制台
    except Exception as e:
        print(f"Error writing to log: {str(e)}")


def main():
    """主循环：定期检查tmux窗口并重启空闲窗口"""
    
    print("=" * 70)
    log_message("🚀 TMUX监控脚本已启动")
    log_message(f"📋 监控窗口: {', '.join(TMUX_WINDOWS)}")
    log_message(f"⏰ 检查间隔: {CHECK_INTERVAL}秒")
    log_message(f"🔍 Sweep ID: {SWEEP_ID}")
    print("=" * 70)
    
    cycle_count = 0
    
    try:
        while True:
            cycle_count += 1
            print(f"\n{'='*70}")
            print(f"🔄 第 {cycle_count} 次检查 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}")
            
            for window in TMUX_WINDOWS:
                # 检查窗口是否空闲
                is_active, current_cmd = check_tmux_window_active(window)
                
                if is_active is None:
                    # 访问窗口失败
                    print(f"❌ {window}: 无法访问 (错误: {current_cmd})")
                    
                elif is_active:
                    # 窗口有任务运行
                    print(f"✅ {window}: 运行中 (进程: {current_cmd})")
                    
                else:
                    # 窗口空闲，需要启动任务
                    print(f"⚠️  {window}: 空闲 (当前: {current_cmd})")
                    print(f"   → 正在发送启动命令...")
                    
                    # 发送命令
                    success = send_command_to_window(window, SLURM_COMMAND)
                    
                    if success:
                        print(f"   ✓ 成功发送命令到窗口 {window}")
                        log_message(f"成功发送命令到窗口 {window}")
                    else:
                        print(f"   ✗ 发送命令到窗口 {window} 失败")
                        log_message(f"发送命令到窗口 {window} 失败")
            
            # 等待下一次检查
            print(f"\n💤 等待 {CHECK_INTERVAL} 秒后进行下一次检查...")
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        log_message("⛔ 收到中断信号，监控脚本停止")
        print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        log_message(f"💥 意外错误: {str(e)}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    main()
