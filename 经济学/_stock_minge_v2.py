# Source Generated with Decompyle++
# File: _stock_minge_v2.cpython-311.pyc (Python 3.11)

__doc__ = '\nSPUM 奇门遁甲 × 股票命格分析 v2\n================================\nPhase 4 静默下载 + 增量保存结果文件\n'
import sys
import time
import json
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TQDM_DISABLE'] = '1'
import numpy as np
import pandas as pd
from datetime import datetime, date
from collections import defaultdict
from zhdate import ZhDate
import akshare as ak
TIANGAN = [
    '甲',
    '乙',
    '丙',
    '丁',
    '戊',
    '己',
    '庚',
    '辛',
    '壬',
    '癸']
DIZHI = [
    '子',
    '丑',
    '寅',
    '卯',
    '辰',
    '巳',
    '午',
    '未',
    '申',
    '酉',
    '戌',
    '亥']
SHENGXIAO = [
    '鼠',
    '牛',
    '虎',
    '兔',
    '龙',
    '蛇',
    '马',
    '羊',
    '猴',
    '鸡',
    '狗',
    '猪']

def year_ganzhi(year = None):
    return (TIANGAN[(year - 4) % 10], DIZHI[(year - 4) % 12])


def day_ganzhi(d = None):
    ref = date(1900, 1, 1)
    days = (d - ref).days
    stem = days % 10
    branch = (days + 10) % 12
    return (TIANGAN[stem], DIZHI[branch])


def ganzhi_for_listing(listing_date = None):
    (yg, yz) = year_ganzhi(listing_date.year)
    lunar = ZhDate.from_datetime(datetime.combine(listing_date, datetime.min.time()))
    mg = (((listing_date.year - 4) % 10 % 5) * 2 + lunar.lunar_month) % 10
    mz = DIZHI[(lunar.lunar_month + 1) % 12]
    (dg, dz) = day_ganzhi(listing_date)
    return {
        'year': f'''{yg}{yz}''',
        'month': f'''{TIANGAN[mg]}{mz}''',
        'day': f'''{dg}{dz}''',
        'lunar': f'''{lunar.lunar_year}年{lunar.lunar_month}月{lunar.lunar_day}日''',
        'shengxiao': SHENGXIAO[(listing_date.year - 4) % 12],
        'year_stem': yg,
        'year_branch': yz,
        'day_stem': dg,
        'day_branch': dz }

STOCKS = [
    ('sz300750', '宁德时代', '2018-06-11'),
    ('sz002415', '海康威视', '2010-05-28'),
    ('sz300124', '汇川技术', '2010-09-28'),
    ('sz002475', '立讯精密', '2010-09-15'),
    ('sz000725', '京东方A', '2001-01-12'),
    ('sh688981', '中芯国际', '2020-07-16'),
    ('sh603501', '韦尔股份', '2017-05-04'),
    ('sh603986', '兆易创新', '2016-08-18'),
    ('sz002230', '科大讯飞', '2008-05-12'),
    ('sz300033', '同花顺', '2009-12-25'),
    ('sh600570', '恒生电子', '2003-12-16'),
    ('sz300059', '东方财富', '2010-03-19'),
    ('sh600519', '贵州茅台', '2001-08-27'),
    ('sh600887', '伊利股份', '1996-03-12'),
    ('sz000333', '美的集团', '2013-09-18'),
    ('sz000651', '格力电器', '1996-11-18'),
    ('sz000858', '五粮液', '1998-04-27'),
    ('sh603288', '海天味业', '2014-02-11'),
    ('sh601888', '中国中免', '2009-10-15'),
    ('sz002304', '洋河股份', '2009-11-06'),
    ('sh600809', '山西汾酒', '1994-01-06'),
    ('sh600882', '妙可蓝多', '1995-12-06'),
    ('sh600276', '恒瑞医药', '2000-10-18'),
    ('sz300760', '迈瑞医疗', '2018-10-16'),
    ('sh603259', '药明康德', '2018-05-08'),
    ('sz000538', '云南白药', '1993-12-15'),
    ('sz300015', '爱尔眼科', '2009-10-30'),
    ('sh600196', '复星医药', '1998-08-07'),
    ('sz002007', '华兰生物', '2004-06-25'),
    ('sh600085', '同仁堂', '1997-06-25'),
    ('sh600036', '招商银行', '2002-04-09'),
    ('sh601318', '中国平安', '2007-03-01'),
    ('sh601398', '工商银行', '2006-10-27'),
    ('sh600030', '中信证券', '2003-01-06'),
    ('sh601166', '兴业银行', '2007-02-05'),
    ('sh600016', '民生银行', '2000-12-19'),
    ('sh600031', '三一重工', '2003-07-03'),
    ('sh600309', '万华化学', '2001-01-05'),
    ('sh601899', '紫金矿业', '2008-04-25'),
    ('sh600585', '海螺水泥', '2002-02-07'),
    ('sz000338', '潍柴动力', '2007-04-30'),
    ('sh600690', '海尔智家', '1993-11-19'),
    ('sh601012', '隆基绿能', '2012-04-11'),
    ('sz300274', '阳光电源', '2011-11-02'),
    ('sz002594', '比亚迪', '2011-06-30'),
    ('sh601633', '长城汽车', '2011-09-28'),
    ('sh600104', '上汽集团', '1997-11-25'),
    ('sz002460', '赣锋锂业', '2010-08-10'),
    ('sh600438', '通威股份', '2004-03-02'),
    ('sz002129', '中环股份', '2007-04-20'),
    ('sz000002', '万科A', '1991-01-29'),
    ('sh600048', '保利发展', '2006-07-31'),
    ('sh600900', '长江电力', '2003-11-18'),
    ('sh601668', '中国建筑', '2009-07-29'),
    ('sz002352', '顺丰控股', '2010-02-05'),
    ('sh601857', '中国石油', '2007-11-05'),
    ('sh600941', '中国移动', '2022-01-05'),
    ('sz001979', '招商蛇口', '2015-12-30')]
day_groups = defaultdict(list)
sx_groups = defaultdict(list)
