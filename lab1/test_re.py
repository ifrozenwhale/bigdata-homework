import re
b = re.match('^1[358]\d{9}$|^147\d{8}$|^179\d{8}$', '13552996099')
if b:
    print(b)
