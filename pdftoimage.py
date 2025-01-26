'''
pip install pdf2image
'''

import os
from pdf2image import convert_from_path

pdf_file = '/Users/chun-haoliu/Desktop/23435-i20.pdf'
output_folder = '/Users/chun-haoliu/Desktop/23435-i20'

pages = convert_from_path(pdf_file, 500)

for count, page in enumerate(pages):
    page.save(os.path.join(output_folder, f'p{count+1}.jpg'), 'JPEG')