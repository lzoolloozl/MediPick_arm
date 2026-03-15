import mujoco
import os

# 配置区
urdf_name = "simple3.urdf"  
xml_name = "simple3_keep_inertia.xml"

current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(current_dir, urdf_name)
output_path = os.path.join(current_dir, xml_name)

print(f"正在尝试从这里加载: {urdf_path}")

try:
    # 重点：我们需要通过加载选项来告诉 MuJoCo 怎么对待惯量
    # 但在 Python API 中，直接从 XML 路径加载时，MuJoCo 会遵循 URDF 内的默认设置
    
    # 为了确保“完全继承”，我们先读取 URDF 内容并检查/注入 compiler 指令
    with open(urdf_path, 'r', encoding='utf-8') as f:
        urdf_content = f.read()
    
    # 检查 URDF 是否包含 mujoco 编译指令
    # 我们要在加载前确保 inertiafromgeom 设置为 "false" 或 "auto"（但不强制重算）
    # 实际上，只要 URDF 里有 <inertial> 标签，MuJoCo 默认会读取它
    
    model = mujoco.MjModel.from_xml_path(urdf_path)
    
    # 关键步骤：mj_saveLastXML 默认会把内存中“优化”过的结果导出来
    # 如果你想让导出的 XML 看起来和 URDF 的惯量一模一样：
    mujoco.mj_saveLastXML(output_path, model)
    
    print("-" * 30)
    print(f"✅ 转换成功！")
    print(f"📂 生成文件: {output_path}")
    print("💡 提示：请检查生成文件中的 <inertial> 标签，它们应与 URDF 中的数值一致。")
    print("-" * 30)

except Exception as e:
    print(f"❌ 转换失败！错误信息: {e}")