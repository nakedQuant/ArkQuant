#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:11:34 2019

@author: python
"""
from reportlab.platypus import Paragraph, Table, Image
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


style_sheet = getSampleStyleSheet()

body = style_sheet['body']
print('body', body)

paragrah = ParagraphStyle(
            leading=10, # leading means  spacing between adjacent lines of text
            autoLeading=False,
            spaceBefore=10,
            spaceAfter=10,
            rightIndent=5,
            leftIndent=5,
            firstLineIndent=5,

)

# Pdf :
#  title
#  Utility
# Porfolio cash
# Position_value
# Weights_series
# Rnl
# Position_rnl
# Returns
# statistics
