import sys
from pathlib import Path

# 将项目根目录添加到 sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print(project_root)