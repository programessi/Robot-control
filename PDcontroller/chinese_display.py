import matplotlib.font_manager as fm

# 列出所有字体的完整信息
for font in fm.fontManager.ttflist:
    if 'wen' in font.name.lower() or 'hei' in font.name.lower():
        print(f"字体名称: {font.name}, 文件路径: {font.fname}")