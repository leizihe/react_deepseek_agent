#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线启动入口：自动优先加载本地vendor目录下的依赖，无需pip安装
仅需Python 3.7+环境即可直接运行
"""
import sys
import os

# 将本地vendor目录加入sys.path最前面，优先使用本地打包的依赖
VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

# 现在导入主程序
from agent import main

if __name__ == "__main__":
    main()
