import os
import csv

def delete_files_csv_mask_match(folder_path, csv_file_path, dry_run=True):
    """
    删除文件夹中那些文件名不匹配CSV第一列去掉"mask_"前缀的文件
    
    逻辑：
    - CSV第一列：mask_image1, mask_image2, mask_document1
    - 文件夹中的文件：image1.jpg, image2.png, image3.jpg, document1.pdf, random.txt
    - 去掉CSV中的"mask_"前缀后得到：image1, image2, document1
    - 匹配结果：保留image1.jpg, image2.png, document1.pdf，删除image3.jpg, random.txt
    
    参数:
    folder_path: 要处理的文件夹路径
    csv_file_path: CSV文件路径  
    dry_run: 是否为预览模式（True=只显示不删除，False=真正删除）
    """
    try:
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"错误: 文件夹不存在: {folder_path}")
            return False
            
        # 检查CSV文件是否存在
        if not os.path.exists(csv_file_path):
            print(f"错误: CSV文件不存在: {csv_file_path}")
            return False
        
        # 读取CSV文件第一列，并去掉"mask_"前缀
        print(f"正在读取CSV文件: {csv_file_path}")
        csv_names_without_mask = set()
        csv_original_names = []
        
        with open(csv_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    print(f"CSV标题行: {row[0] if row else '空行'}")
                    continue  # 跳过标题行
                if row and len(row) > 0:
                    original_name = str(row[0]).strip()
                    csv_original_names.append(original_name)
                    
                    if original_name.startswith("mask_"):
                        # 去掉"mask_"前缀
                        name_without_mask = original_name[5:]  # 去掉前5个字符"mask_"
                        if name_without_mask:  # 确保去掉前缀后不是空字符串
                            csv_names_without_mask.add(name_without_mask)
                    else:
                        # 如果没有"mask_"前缀，也加入匹配列表
                        csv_names_without_mask.add(original_name)
        
        print(f"CSV第一列包含 {len(csv_original_names)} 个条目")
        print(f"去掉'mask_'前缀后得到 {len(csv_names_without_mask)} 个唯一名称")
        
        if len(csv_names_without_mask) <= 10:
            print(f"去掉前缀后的名称: {list(csv_names_without_mask)}")
        else:
            print(f"去掉前缀后的前10个名称: {list(csv_names_without_mask)[:10]}")
        
        # 获取文件夹中的所有文件
        print(f"\n正在扫描文件夹: {folder_path}")
        folder_files = []
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):  # 只处理文件，不处理子文件夹
                folder_files.append(item)
        
        print(f"文件夹中共有 {len(folder_files)} 个文件")
        
        # 分析每个文件
        files_to_delete = []
        files_to_keep = []
        
        for filename in folder_files:
            # 检查文件名（带扩展名和不带扩展名）是否在CSV去掉前缀的名称中
            name_with_ext = filename
            name_without_ext = os.path.splitext(filename)[0]
            
            # 检查是否匹配
            if name_with_ext in csv_names_without_mask or name_without_ext in csv_names_without_mask:
                files_to_keep.append(filename)
            else:
                files_to_delete.append(filename)
        
        # 显示结果
        print(f"\n===== 分析结果 =====")
        print(f"需要保留的文件: {len(files_to_keep)} 个")
        print(f"需要删除的文件: {len(files_to_delete)} 个")
        
        if files_to_keep:
            print(f"\n保留的文件（前10个）:")
            for f in files_to_keep[:10]:
                print(f"  ✓ {f}")
        
        if files_to_delete:
            print(f"\n{'【预览模式】' if dry_run else '【删除模式】'}要删除的文件:")
            for f in files_to_delete:
                print(f"  ✗ {f}")
        
        # 执行删除操作
        if files_to_delete:
            if dry_run:
                print(f"\n这是预览模式，没有实际删除文件。")
                print(f"如要真正删除，请将 dry_run=False")
            else:
                confirm = input(f"\n确定要删除这 {len(files_to_delete)} 个文件吗? (输入 'yes' 确认): ")
                if confirm.lower() == 'yes':
                    deleted_count = 0
                    for filename in files_to_delete:
                        try:
                            file_path = os.path.join(folder_path, filename)
                            os.remove(file_path)
                            deleted_count += 1
                            print(f"已删除: {filename}")
                        except Exception as e:
                            print(f"删除失败 {filename}: {e}")
                    
                    print(f"\n删除完成！成功删除 {deleted_count} 个文件")
                else:
                    print("取消删除操作")
        else:
            print("\n没有需要删除的文件")
            
        return True
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        return False

# 使用示例
if __name__ == "__main__":
    # 配置参数
    folder_path = "/root/autodl-tmp/QaTa/Test_Folder/img"      # 改为你的文件夹路径
    csv_file_path = "/root/autodl-tmp/QaTa/Test_Folder/Test_text.csv"       # 改为你的CSV文件路径
    
    print("文件清理工具 - CSV有mask_前缀，文件夹无前缀匹配")
    print("=" * 60)
    
    # 先预览模式运行
    print("第一步：预览模式 - 查看哪些文件会被删除")
    delete_files_csv_mask_match(folder_path, csv_file_path, dry_run=True)
    
    print("\n" + "=" * 60)
    confirm = input("如果预览结果正确，输入 'delete' 来真正删除文件: ")
    
    if confirm.lower() == 'delete':
        print("第二步：删除模式 - 真正执行删除")
        delete_files_csv_mask_match(folder_path, csv_file_path, dry_run=False)
    else:
        print("已取消删除操作")