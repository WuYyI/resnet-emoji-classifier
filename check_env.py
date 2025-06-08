import sys
import pkg_resources

def check_package(package_name):
    try:
        pkg_resources.get_distribution(package_name)
        print(f"✓ {package_name} 已安装")
        return True
    except pkg_resources.DistributionNotFound:
        print(f"✗ {package_name} 未安装")
        return False

def main():
    print("Python版本:", sys.version)
    print("\n检查必要的包：")
    
    required_packages = [
        "torch",
        "torchvision",
        "Pillow",
        "numpy",
        "matplotlib",
        "tensorboard"
    ]
    
    missing_packages = []
    for package in required_packages:
        if not check_package(package):
            missing_packages.append(package)
    
    if missing_packages:
        print("\n缺少以下包，请使用以下命令安装：")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        print("\n所有必要的包都已安装！")

if __name__ == "__main__":
    main() 