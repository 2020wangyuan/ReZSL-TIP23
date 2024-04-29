import subprocess

# 读取vim_requirements.txt文件
with open("vim_requirements.txt", "r") as f:
    required_packages = [line.split('==')[0] for line in f.readlines() if line.strip()]

# 使用pip freeze获取当前环境中已安装的包
installed_packages_output = subprocess.check_output(["pip", "freeze"]).decode("utf-8")
installed_packages_lines = installed_packages_output.splitlines()
installed_packages = [line.split('==')[0] for line in installed_packages_lines]

# 对比两个列表
required_set = set(required_packages)
installed_set = set(installed_packages)

# 找出未安装的包
missing_packages = required_set - installed_set

# 找出多余安装的包
extra_packages = installed_set - required_set

# 输出结果
print("未安装的包:", sorted(missing_packages))
print("多余安装的包:", sorted(extra_packages))
