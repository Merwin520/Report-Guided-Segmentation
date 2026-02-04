import zipfile
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='解压ZIP文件工具')
    parser.add_argument('zipfile', help='要解压的ZIP文件路径')
    parser.add_argument('-d', '--directory', help='解压目标目录', default=None)
    parser.add_argument('-o', '--overwrite', help='覆盖已存在的文件', action='store_true')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.zipfile):
        print(f"错误: 文件 {args.zipfile} 不存在")
        return
    
    extract_to = args.directory if args.directory else os.path.splitext(args.zipfile)[0]
    
    try:
        os.makedirs(extract_to, exist_ok=True)
        
        with zipfile.ZipFile(args.zipfile, 'r') as zip_ref:
            if args.overwrite:
                # 直接解压所有文件
                zip_ref.extractall(extract_to)
            else:
                # 检查文件是否已存在
                for file in zip_ref.namelist():
                    target_path = os.path.join(extract_to, file)
                    if os.path.exists(target_path):
                        print(f"跳过已存在的文件: {file}")
                    else:
                        zip_ref.extract(file, extract_to)
            
            print(f"解压完成！共解压 {len(zip_ref.namelist())} 个文件到 {extract_to}")
            
    except zipfile.BadZipFile:
        print("错误: 无效的ZIP文件")
    except Exception as e:
        print(f"解压过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()