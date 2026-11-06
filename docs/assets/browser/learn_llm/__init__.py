"""浏览器版 src/learn_llm/__init__.py：空的包入口。

仓库里的 __init__.py 会导入 attention、model、lora 等 torch 模块。浏览器里的运行只用 learn_llm.rag 和
learn_llm.agent 这两个纯标准库模块，它们逐字来自仓库；包入口换成这一份，不导入任何东西。
"""
