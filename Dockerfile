# 沙箱镜像：python:3.12-slim 加上一套命令行工具。
#
# 为什么不是现成的镜像：非 slim 的 python:3.12 自带 git，但要多背一倍多的体积；slim 干净
# 但很多工具没有。要哪几样就自己加哪几样。
#
# 构建一次（在仓库根目录）：
#
#     docker build -t mca_sandbox .
#
# 名字要和 shell.py 里的 DEFAULT_IMAGE 对上；忘了构建的话，跑 --shell 时报的是"容器没
# 起来"——那句错看不出是镜像不在，所以 shell.py 里专门补了一句提示。
#
# 容器里**没有网络**（--network none），apt 装不了任何东西：以后还想让沙箱里多点什么，
# 只能往这个文件里加，别指望在容器里现装。
#
# 装什么：
#   git        读本地历史（status / diff / log）。写历史那条路被 .git 的只读挂载挡着
#   jq         处理 JSON（package.json、各种工具的输出）
#   procps     给 ps——命令卡住的时候能看一眼
#   gawk       比自带的 mawk 多 gensub、更强的正则；有些脚本点名要 gawk
#   make       仓库里有 Makefile 时直接能跑；它不必跟着 build-essential 一起装
#   ripgrep    模型自己敲的 rg，在大仓库上比 grep 快一个量级
#   patch      git diff > x.patch 之后能应用回去
#   file       认出二进制/编码，读之前先判断
#   unzip      遇到 zip 不用抓瞎（tar / gzip 本来就有）
#   sqlite3    项目里真用 SQLite 时能直接查
#
# 不装的，都是有原因的：
#   curl / wget      没有网络，装了也够不着任何东西，只会让人以为沙箱有网
#   build-essential  没网就装不了 PyPI 上的包，编译能力大半用不上；体积还翻倍
#   vi / nano / less 没有 TTY、输出是被捕获的，装了只是给模型机会把自己挂住
#
# 还没做的：把项目的依赖烤进来（构建时装、运行时仍然断网）。那件事会把这个镜像和本仓库
# 绑死——pyproject 一变就得重建，所以先没动。
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git \
      jq \
      procps \
      gawk \
      make \
      ripgrep \
      patch \
      file \
      unzip \
      sqlite3 \
 && rm -rf /var/lib/apt/lists/*
