"""tests 包标记。

必须保留：若无此文件，tests 是命名空间包，会被 site-packages 里第三方
库附带的普通 `tests` 包劫持（远程环境实测：datasets 依赖链会装一个），
导致 `from tests.helpers import ...` 导入到错误的模块。
"""
