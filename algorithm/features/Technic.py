# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
import pandas as pd,numpy as np

from gateWay import Event,GateReq,Quandle
from algorithm.features import BaseFeature,MA,XEMA,SMA,remove_na,EMA,AEMA

class TEMA(BaseFeature):
    '''
    三重指数移动平均
    3 * EMA - 3 * EMA的EMA + EMA的EMA的EMA
    '''
    _weight = None

    @classmethod
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        EMA._weight = cls._weight
        ema = EMA.calc_feature(raw,window)
        # print('ema',ema)
        xema = XEMA.calc_feature(raw,window)
        # print('xema',xema)
        XEMA._dimension = 3
        xema3 = XEMA.calc_feature(raw,window)
        # print('xema3',xema3)
        tema_windowed = 3 * ema[-len(xema3):]  - xema[-len(xema3):] - xema3
        return tema_windowed

class DEMA(BaseFeature):

    '''
        双指数移动平均线，一个单指数移动平均线和一个双指数移动平均线 : 2 * n日EMA - n日EMA的EMA
    '''
    _weight = None

    @classmethod
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        EMA._weight = cls._weight
        ema = EMA.calc_feature(raw, window)
        XEMA._weight = cls._weight
        xema2 = XEMA.calc_feature(raw,window)
        dema_windowed = 2 * ema[-len(xema2):] - xema2
        return dema_windowed

class SMMA(BaseFeature):
    '''
        平滑移动平均线(SMMA)
        SMMA1 = SUM(CLOSE(i), N) / N
        经过运算转换公式可以简化为：SMMA (i) = (SMMA (i - 1) * (N - 1) + CLOSE (i)) / N
    '''
    _weight = None

    @classmethod
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        init_smma = raw.rolling(window = window).mean()
        wgt = (window - 1)/window
        smma = raw * (1-wgt) + init_smma * wgt
        return smma

class IMI(BaseFeature):
    '''
    日内动量指标
    类似于CMO，ISu /（ISu + ISd), 其中ISu表示收盘价大于开盘价的交易日的收盘价与开盘价之差
    ISd则与ISu相反
    '''
    _n_fields = ['open','close']

    @classmethod
    @remove_na
    def calc_feature(cls,feed):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        diff = raw['close'] - raw['open']
        isu = diff[raw['close'] > raw['open']]
        imi = isu.sum() / abs(diff).sum()
        return imi

class RAVI(BaseFeature):
    '''
        区间运动辨识指数（RAVI）7日的SMA与65 天的SMA之差占65天的SMA绝对值，一般来讲，RAVI被限制在3 % 以下，市场做区间运动
        _pariwise True means the length of window is two
    '''
    _pairwise = True

    @classmethod
    def calc_feature(cls,feed,window):
        if cls._pairwise and not isinstance(window,(tuple,list)):
            raise TypeError('when pairwise is True,window must be list or tuple of length of 2')
        raw = feed.copy()
        smaShort = SMA.calc_feature(raw,np.array(window).min())
        smaLong = SMA.calc_feature(raw,np.array(window).max())
        ravi_windowed = (smaShort - smaLong) / smaLong
        return ravi_windowed

class DMA(BaseFeature):
    '''
        DMA: 收盘价的短期平均与长期平均的差得DMA；DMA的M日平均为AMA;
        DMA与AMA比较判断交易方向,参数：12、50、14
    '''
    _pairwise = True

    @classmethod
    def calc_feature(cls,feed,window):
        if cls._pairwise and not isinstance(window,(tuple,list)):
            raise TypeError('when pairwise is True,window must be list or tuple of length of 2')
        raw = feed.copy()
        dmaShort = MA.calc_feature(raw,np.array(window).min())
        dmaLong = MA.calc_feature(raw,np.array(window).max())
        dma_windowed = dmaShort[-len(dmaLong):] - dmaLong
        return dma_windowed

class ER(BaseFeature):
    '''
        效率系数，净价格移动除以总的价格移动（位移除以路程绝对值之和），表明趋势的有效性
        基于几何原理 两点之间直线距离最短，位移量 / 直线距离判断效率
        ER(i) = Sinal(i) / Noise(i)
        ER(i) ― 效率比率的当前值;
        Signal(i) = ABS(Price(i) - Price(i - N)) ― 当前信号值，当前价格和N周期前的价格差的绝对值;
        Noise(i) = Sum(ABS(Price(i) - Price(i - 1)), N) ― 当前噪声值，对当前周期价格和前一周期价格差的绝对值求N周期的和。
        在很强趋势下效率比倾向于1；如果无定向移动，则稍大于0
    '''

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        raw  = feed.copy()
        displacement = raw - raw.shift(window)
        distance = abs(raw.diff()).rolling(window = window).sum()
        er_windowed  = displacement / distance
        return er_windowed

class AMA(BaseFeature):
    '''
        自适应函数
        指数平滑公式：EMA(i) = Price(i) * SC + EMA(i - 1) * (1 - SC),SC = 2 / (n + 1) ― EMA平滑常数;
        对于快速市场平滑比必须是关于EMA的2周期(fast SC = 2 / (2 + 1) = 0.6667), 而无趋势EMA周期必须等于30(
        slow SC = 2 / (30 + 1) = 0.06452)

        新的变化中的平滑常量为SSC（成比例的平滑常量）:
        SSC(i) = (ER(i) * (fast SC - slow SC) + slow SC

        AMA(i) = AMA(i - 1) * (1 - SSC(i) ^ 2) + Price(i) * (SSC(i) ^ 2)
    '''
    _fast = 2/3
    _low = 2/31

    @classmethod
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        er_windowed = ER.calc_feature(raw,window)
        ssc = (er_windowed * (cls._fast - cls._low) + cls._low) ** 2
        allign_raw = raw[-len(ssc)-1:]
        AEMA._weight = ssc
        ama_win = [AEMA.calc_feature(allign_raw[:idx]) for idx in range(window,len(allign_raw) + 1)]
        ama_windowed = pd.Series(ama_win,index = allign_raw.index[-len(ama_win):])
        return ama_windowed

class VEMA(BaseFeature):
    '''
        变量移动平均线基于数据系列的波动性而调整百分比的指数移动平均线
        (SM * VR） *收盘价 + （1 - SM * VR） * （昨天）MA
        SM = 2 / (n + 1);
        VR可以采用钱德勒摆动指标绝对值 / 100
    '''

    @remove_na
    def _init_vema(self,raw,window):
        vema_std = raw.rolling(window = window).std()
        vema_ratio = vema_std * (1/(window + 1))
        return vema_ratio

    @classmethod
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        ins = VEMA()
        vr = ins._init_vema(raw,window)
        allign_raw = raw[-len(vr) - 1:]
        AEMA._weight = vr
        vema_win = [AEMA.calc_feature(allign_raw[:idx]) for idx in range(window,len(allign_raw) + 1)]
        vema_windowed = pd.Series(vema_win,index = allign_raw.index[-len(vema_win):])
        return vema_windowed

class CMO(BaseFeature):
    '''
        钱德动量摆动指标 归一化
        Su(上涨日的收盘价之差之和） Sd(下跌日的收盘价之差的绝对值之和） (Su - Sd) / (Su + Sd)
    '''

    @staticmethod
    def _calc_init(data):
        #标准化
        data =(data - data.min())/(data.max() - data.min())
        data_diff = data - data.shift(1)
        su = data[data_diff > 0 ].sum()
        sd =  data[data_diff < 0 ].sum()
        cmo = (su + sd) / (su  - sd)
        return cmo

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        cmo = cls._calc_init(raw)
        return cmo

class Vidya(BaseFeature):
    '''
    变量指数动态平均线技术指标(VIDYA): 计算平均周期动态变化的指数移动平均线(EMA)
    平均周期取决于市场波动情况；振荡器(CMO) /100
    CMO值用作平滑因素EMA比率
    '''

    @staticmethod
    def _init_vidya(feed,window):
        #标准化
        cmo = CMO.calc_feature(feed,window)
        print('cmo',cmo)
        vidya_ratio = cmo * (2/(window+1))
        return vidya_ratio

    @classmethod
    def calc_feature(cls, feed,window):
        raw = feed.copy()
        vidya = cls._init_vidya(raw,window)
        print('vidya',vidya/100)
        EMA._weight = vidya/100
        vidya_windowed = EMA.calc_feature(raw,window)
        return vidya_windowed

class Jump(BaseFeature):
    """
        跳空动能量化 jump threshold --- 度量跳空能量（21D change_pct 均值）
        jump diff --- preclose * jump threshold
        向上跳空 low - preclose > jump diff
        向下跳空 preclose  - high > jump diff
        fields :'close','low','high'
    """
    _n_fields = ['high','low','close']

    @staticmethod
    @remove_na
    def _cal_jump(raw,window):
        pct = raw['close'] / raw['close'].shift(1) -1
        pct_thres = pct.rolling(window = window).mean()
        jump_diff = pct_thres * raw['close'].shift(1)
        jump_up = raw['low'] - raw['close'].shift(1)
        jump_down = raw['close'].shift(1) - raw['high']
        power_up = sum(jump_up > jump_diff) / window
        power_down = sum(jump_down < jump_diff) / window
        power = power_up - power_down
        return power

    @classmethod
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        jump = cls._cal_jump(raw,window)
        return jump

class Speed(BaseFeature):
    '''
    趋势速度跟踪策略 基于符号来定义定义趋势变化的速度值（比率），波动中前进，总会存在回调，度量回调的个数
    计算曲线跟随趋势的速度，由于速度是相对的，所以需要在相同周期内与参数曲线进行比对
    符号相同的占所有和的比例即为趋势变化敏感速度值，注意如果没有在相同周期内与参数曲线进行比对的曲线，本速度值即无意义
    所对应的趋势变化敏感速度数值, 以及相关性＊敏感度＝敏感度置信度
    '''

    @classmethod
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        raw_trend = (raw.rolling(window = window).mean()).diff()
        raw_trend.dropna(inplace = True)
        trend_sign = pd.Series(np.where(raw_trend > 0 ,1,-1))
        trend_speed = trend_sign * trend_sign.shift(-1)
        print('trend_speed',trend_speed)
        speed = trend_speed[trend_speed == 1].sum() / len(trend_speed)
        return speed

class MassIndex(BaseFeature):
    '''
    质量指标计算公式
    sum(（最高价 - 最低价）的9日指数移动平均 / （最高价 - 最低价）的9日指数移动平均的9日指数移动平均)
    '''
    _n_fields = ['high','low']

    @classmethod
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        u_b= raw['high'] - raw['low']
        ema = EMA.calc_feature(u_b,window)
        xema = XEMA.calc_feature(u_b,window)
        massIndex = ema[-len(xema):] - xema
        return massIndex

class Dpo(BaseFeature):
    '''
    非趋势价格摆动指标 ： 收盘价减去 （n / 2) +1的平均移动平均数具有将DPO后移（n / 2) + 1的效果
    '''

    @classmethod
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        window_ = int(window/2) + 1
        ma = MA.calc_feature(raw,window_)
        dpo_window = raw - ma
        return dpo_window

class RVI(BaseFeature):
    '''
        相对活力指标,衡量活跃度 --- 度量收盘价处于的位置
        代表当前技术线值在当前的位置，非停盘 (self.close - self.low) / (self.high - self.low)
    '''
    _n_fields = ['high','close','low']
    _windowed = False

    @classmethod
    def calc_feature(cls,feed):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        rvi = (raw['close'] - raw['open'])/(raw['high'] - raw['low'])
        return rvi

class SMI(BaseFeature):
    '''
        stochastic momentum index 随意摆动指收盘价相对于近期的最高价 / 最低价区间的位置进行两次EMA平滑
    '''
    _n_fields = ['high','low','close']

    @classmethod
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        rvi = RVI.calc_feature(raw)
        smi_window = XEMA.calc_feature(rvi,window)
        return smi_window

class RPower(BaseFeature):
    '''
        动量指标度量证券价格变化
        收盘价 / n期前收盘价
        动能指标计算:1、价格涨，动能为+1，价格跌，动能为-1；2、价格与成交量的变动乘积作为动能
        （score ** 2） *delta_vol
    '''
    _n_fields = ['open', 'high', 'close', 'low', 'volume']

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        rvi = RVI.calc_feature(feed)
        delta_vol = raw['volume'].diff()
        momentum = (rvi ** 2) * delta_vol
        return momentum

class DPower(BaseFeature):
    """
        动量策略 --- 基于价格、成交量等指标量化动能定义，同时分析动能 ；属于趋势范畴 -- 惯性特质 ，不适用于反弹
        逻辑：
        1、价格 --- 收盘价、最高价，与昨天相比判断方向，交集
        2、成交量 --- 与昨天成交量比值
        3、1与2结果相乘极为动量
        4、计算累计动量值
    """
    _n_fields = ['close','high','volume']

    @classmethod
    def calc_feature(cls,feed):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        delta = raw['volume'] / raw['volume'].shift(1)
        #初始为0
        delta.fillna(0,inplace= True)
        direction = (raw['close'] > raw['close'].shift(1)) & (raw['high'] > raw['high'].shift(1))
        sign = direction.apply(lambda x : 1 if x else -1)
        momentum = (sign * delta).cumsum()
        return momentum

class MAO(BaseFeature):
    '''
    MA oscillator 价格摆动， 短期MA / 长期MA的百分比
    '''
    _pairwise = True

    @classmethod
    @remove_na
    def calc_feature(cls, feed,window):
        if cls._pairwise and not isinstance(window,(tuple,list)):
            raise TypeError('when pairwise is True,window must be list or tuple of length of 2')
        raw = feed.copy()
        ma_short = MA.calc_feature(raw,np.array(window).min())
        ma_long = MA.calc_feature(raw,np.array(window).max())
        mao = ma_short / ma_long
        return mao

class MFI_(BaseFeature):
    '''
        公式：（最高价 - 最低价） / 成交量
        衡量 每单位成交量的价格变动来衡量价格变动的效率
        类别：
            green(MFI和成交量都之前增加)
            fade(MFI和成交量比以前减少)
            fake(成交量减少和MFI增加)
            squat(成交量增加，但是MFI下降)
    '''
    _n_fields = ['high','low','volume']

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        mfi = (raw['high'] - raw['low'] )/raw['volume']
        mfi_window = mfi.rolling(window = window).mean()
        return mfi_window

class VHF(BaseFeature):
    '''
        vertical horizonal filter 判断处于趋势阶段还是盘整阶段hcp: n期内最高收盘价lcp: n期内最低收盘价分子hcp - lcp,
        分母：n期内的收盘价变动绝对值之和
    '''

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        vertical = raw.rolling(window = window).max() - raw.rolling(window = window).min()
        pct = abs(raw - raw.shift(1))
        horizonal = pct.rolling(window = window).sum()
        vhf_window = vertical / horizonal
        return vhf_window

class PVT(BaseFeature):
    '''
        价量趋势指标，累积成交量: pvi(t) = pvi(t - 1) + （收盘价 - 前期收盘价） / 前期收盘价 * pvi
        pvi为初始交易量
    '''
    _n_fields = ['volume','close']

    @classmethod
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        wgt = raw['close'] / raw['close'].shift(1) -1
        wgt.fillna(1,inplace = True)
        pvt = [sum(raw['volume'][:idx] * wgt[:idx]) for idx in range(window,len(raw)+1)]
        pvt_windowed = pd.Series(pvt,index = raw.index[-len(pvt):])
        return pvt_windowed

class Accumulation(BaseFeature):
    """
        累积 / 派发线与证券市场的背离，变化趋势即将来临 ， Score * 成交量 累积起来 ,类比于pvt 并结合RVI指标
    """
    _n_fields = ['high','close','low','volume']

    @classmethod
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        rvi = RVI.calc_feature(feed)
        accumulation = [sum(raw['volume'][:idx] * rvi[:idx]) for idx in range(window,len(raw)+1)]
        accumulation_windowed = pd.Series(accumulation,index = raw.index[-len(accumulation):])
        return accumulation_windowed

class VCK(BaseFeature):
    '''
        蔡金： 波动性增加表示底部临近，而波动性降低表示顶部临近
        hl平均值 = (high - low)的指数移动平均
        蔡金波动率 = （ hl平均值 - n期前的hl平均值 ） / n期前的hl平均值
    '''
    _n_fields = ['high','low']

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        hl = raw['high'] - raw['low']
        ema_windowed = EMA.calc_feature(hl,window)
        vck = ema_windowed / ema_windowed.shift(window) -1
        return vck

class Rsi(BaseFeature):
    '''
        RSI(相对强弱指数)是通过比较一段时期内的平均收盘涨数和平均收盘跌数来分析市场买沽盘的意向和实力
        1. 根据收盘价格计算价格变动
        2.
        分别筛选gain交易日的价格变动序列gain，和loss交易日的价格变动序列loss
        3.
        分别计算gain和loss的N日移动平均
        4.
        rsi = 100 * gain_mean÷（gain_mean＋loss_mean）

    '''

    @staticmethod
    def _init_rsi(data):
        gain_mean = data[data>0].mean()
        loss_mean = data[data<0].mean()
        rsi = 100 * gain_mean / (gain_mean+loss_mean)
        return rsi

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        raw_diff = raw.diff()
        rs_windowed = raw_diff.rolling(window = window).apply(cls._init_rsi)
        return rs_windowed

class WAD(BaseFeature):
    '''
        威廉姆斯累积 / 派发指标 A / D 指标累积
        TRH = max(preclose, high)  TRL = min(preclose, min) 如果收盘价大于昨日收盘价： A / D = close - TRL
        如果收盘价小于昨日收盘价：A / D = close - TRH
        通过计算平滑移动避免个别极值影响
    '''
    _n_fields = ['high','low','close']

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        raw['preclose'] = raw['close'].shift(1)
        trh = raw['close'] - raw.loc[:,['preclose','high']].max(axis = 1)
        trl = raw['close'] - raw.loc[:,['preclose','low']].min(axis = 1)
        wad = raw['close'].diff()
        wad[wad >0] = trh
        wad[wad<0] = trl
        wad_windowed = wad.rolling(window = window).mean()
        return wad_windowed

class WR(BaseFeature):
    '''
    威廉姆斯 % R：预期价格反转的力量在证券价格达到顶峰的时候转为下跌的时候，提前下跌了；在达到谷底之后转为上涨，并且维持一段时间
    公式：- 100 * (n期内最高价 - 当期收盘价) / (n期最高价 - n期内最低价)
    '''

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        rolling_max = raw.rolling(window = window).max()
        rolling_min = raw.rolling(window = window).min()
        wr_windowed = -100 * (rolling_max - raw) / (rolling_max - rolling_min)
        return wr_windowed

class Aroon(BaseFeature):
    '''
    阿隆指标价格达到近期最高价和最低价以来经过的期间数预测证券价格从趋势性到交易性区域
    阿隆上升线： （n - (n + 1 到达最高价的期间数) ） / n;
    而阿隆下降线： （n - (n + 1 到达最低价的期间数)） / n 平行、交叉穿行
    '''

    @staticmethod
    def measure(data):
        max_point = data.idxmax()
        min_point = data.idxmin()
        if max_point == min_point:
            pos = None
        else:
            pos = (len(data) - max_point)/(max_point - min_point)
        return pos

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        close = raw.pivot(columns = 'code',values = 'close')
        close.index = range(len(close))
        aroon = close.apply(cls.measure)
        return aroon

class DMI(BaseFeature):
    '''
    DMI中的时间区间变化受到价格波动的控制，在平稳的市场较多的时间区间数，在活跃的市场采取较少的时间区间数
    时间区间数 = 14 / 波动性指标 波动性指标 = 收盘价5日标准差除以收盘价5日标准差10日移动平均数
    '''
    _pairwise = True

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        if cls._pairwise and not isinstance(window,(tuple,list)):
            raise TypeError('when pairwise is True,window must be list or tuple of length of 2')
        raw = feed.copy()
        dmi_std = raw.rolling(window = np.array(window).min()).std()
        dmi__std_ma = dmi_std.rolling(window = np.array(window).max()).mean()
        dmi = dmi_std / dmi__std_ma
        return dmi

class VO(BaseFeature):
    '''
    成交量提供给定价格运动的交易密集程度，高成交量水平是顶部特征，在底部之前也会因为恐慌而放大
    VO（volume oscillator)  vol的短期移动平均值与长期移动平均值之差的比率
    '''
    _pairwise = True

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        if cls._pairwise and not isinstance(window,(tuple,list)):
            raise TypeError('when pairwise is True,window must be list or tuple of length of 2')
        raw = feed.copy()
        ma_short = MA.calc_feature(raw,np.array(window).min())
        ma_long = MA.calc_feature(raw,np.array(window).max())
        vo_windowed = ma_short / ma_long
        return vo_windowed

class SO(BaseFeature):
    '''
        随机：共同分布的随机变量的无穷过程。随意摆动指标：收盘价相对于给定时间内价格区间的位置
        公式(今天收盘价 - % K期间内最低价） / （ % K最高价 - % K最低价）
        指定 % K进行处理（MA、EMA、变动移动平均、三角移动等等）
    '''
    _n_fields = ['high','low','close']

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        so_windowed = (raw['close'] - raw['low'].rolling(window = window).min())/\
                      (raw['high'].rolling(window = window).max() - raw['low'].rolling(window = window).min())
        return so_windowed

class WS(BaseFeature):
    '''怀尔德平滑 --- 前期MA + （收盘价 - 前期MA） / 期间数'''

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        ma_windowed = MA.calc_feature(raw,window)
        ws_windowed = ma_windowed + (raw - ma_windowed) / window
        return ws_windowed

class ADX(BaseFeature):
    '''
        N: = 14; REF代表过去, 用法是: REF(X, A), 引用A周期前的X值例如: REF(CLOSE, 1)
        TR1: = SMA(TR, N, 1);
        HD: = HIGH - REF(HIGH, 1);
        LD: = REF(LOW, 1) - LOW;
        DMP: = SMA(IF(HD > 0 AND HD > LD, HD, 0), N, 1);
        DMM: = SMA(IF(LD > 0 AND LD > HD, LD, 0), N, 1);
        PDI: DMP * 100 / TR1;
        MDI: DMM * 100 / TR1;
        ADX: SMA(ABS(MDI - PDI) / (MDI + PDI) * 100, N, 1)

        _windowed = False --- return value  else return list or Series
    '''
    _n_fields = ['high','low']
    _weight = None

    @staticmethod
    def _init_adx(raw,window):
        dmp = raw['high'] - raw['high'].shift(1)
        print('dmp',dmp)
        dmm = raw['low'].shift(1) - raw['low']
        print('dmm',dmm)
        dmp_sign = (dmp > 0) & (dmp > dmm)
        print('dmp_sign',dmp_sign)
        dmm_sign = (dmm > 0 ) & (dmp < dmm)
        print('dmm_sign',dmm_sign)
        dmp[~dmp_sign] = 0
        dmm[~dmm_sign] = 0
        pdi = SMA.calc_feature(dmp,window)
        mdi = SMA.calc_feature(dmm,window)
        return pdi , mdi

    @classmethod
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        SMA._weight = cls._weight
        pdi,mdi = cls._init_adx(raw,window)
        res = 100 * (mdi - pdi)/(mdi + pdi)
        adx = SMA.calc_feature(res,window)
        return adx

class MFI(BaseFeature):
    '''price = （最高价 + 最低价 + 收盘价） / 3  ; 货币流量 = price * 成交量  ; 货币比率 = 正的货币流量 / 负的货币流量
        货币流量指标 = 100（1 - 1 /（1 + 货币比率））
        如果price大于preprice 正流入 ，否则为流出
    '''
    _n_fields = ['high','close','low','volume']

    @staticmethod
    def _init_mfi(data):
        typical = data.loc[:,['high','low','close']].mean(axis =1)
        flow_sign = typical > typical.shift(1)
        postive_flow = typical[flow_sign] * data['volume'][flow_sign]
        negative = typical[~flow_sign] * data['volume'][~flow_sign]
        mfi = postive_flow.sum() / negative.sum()
        return mfi

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        mfi = cls._init_mfi(raw)
        return mfi

class RI(BaseFeature):
    '''期间之间曲线（收盘价之间变化） 与平均期的期间内区域（最高价与最低价）比值 判断先行趋势是否结束结束'''
    _n_fields = ['high', 'low', 'close']

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        close_windowed_ret = raw['close']/ raw['close'].shift(window)
        high_windowed = raw['high'].rolling(window = window).max()
        low_windowed = raw['low'].rolling(window = window).min()
        ri_windowed = close_windowed_ret / (high_windowed - low_windowed)
        return ri_windowed

class TRIX(BaseFeature):
    '''
        1. TR=收盘价的N日指数移动平均；
        2.TRIX=(TR-昨日TR)/昨日TR*100；
    '''

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        ema_windowed = EMA.calc_feature(raw,window)
        trix_windowed = 100 * (ema_windowed / ema_windowed.shift(1) -1 )
        return trix_windowed

class KVO(BaseFeature):
    '''
        成交量动力 = V * abs(2 * DM / CM  - 1 ) *T * 100
        V为成交量，DM每日度量值（最高价减去最低价） CM（DM的累积度量值），
        T趋势（high low close 的均值与前一天的均值比较大于为1 ，否则 - 1）
        KVO : 成交量动力的34指数移动平均线减去55的指数移动平均线

    '''
    _n_fields = ['volume', 'high', 'low']
    _pairwise = True

    @staticmethod
    def _init_vo(data):
        dm = data['high'] - data['low']
        cm = dm.cumsum()
        avgprice = data.loc[:,['high','low','close']].mean(axis =1)
        sign = avgprice > avgprice.shift(1)
        data.loc[:,'sign'] = -1
        data['sign'][sign] = 1
        vo = data['volume'] * abs(2 * dm / cm - 1) * data['sign'] * 100
        return vo

    @classmethod
    def calc_feature(cls,feed,window):
        if cls._pairwise and not isinstance(window,(tuple,list)):
            raise TypeError('when pairwise is True,window must be list or tuple of length of 2')
        super()._validate_fields(feed,cls._n_fields)
        raw = feed.copy()
        vo = cls._init_vo(raw)
        ema_short = EMA.calc_feature(vo,np.array(window).min())
        ema_long = EMA.calc_feature(vo,np.array(window).max())
        kvo_windowed = ema_short - ema_long
        return kvo_windowed

class KDJ(BaseFeature):
    '''
        Stochastics指标又名KDJ
        反映价格走势的强弱和超买超卖现象。主要理论依据是：当价格上涨时，收市价倾向于接近当日价格区间的
        上端；相反，在下降趋势中收市价趋向于接近当日价格区间的下端
        % K的方程式: % K = 100 * (CLOSE - LOW( % K)) / (HIGH( % K) - LOW( % K))
        注释：
        CLOSE — 当天的收盘价格
        LOW( % K) — % K的最低值
        HIGH( % K) — % K的最高值
        根据公式，我们可以计算出 % D的移动平均线:
        % D = SMA( % K, N)
    '''
    _n_fields = ['close','high', 'low']

    @classmethod
    @remove_na
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        k = (raw['close'] - raw['low'].rolling(window = window).min()) /\
            (raw['high'].rolling(window = window).max() - raw['low'].rolling(window = window).min())
        kdj = k.rolling(window = window).mean()
        return kdj

class Macd(BaseFeature):
    '''
    通过macd公式手动计算macd
    param
    fast_period: 快的加权移动均线线, 默认12，即EMA12
    param
    slow_period: 慢的加权移动均线, 默认26，即EMA26
    param
    signal_period: dif的指数移动平均线，默认9

    DIF : MACD称为指数平滑异动移动平均线，是从双指数移动平均线发展而来的，由快的加权移动均线（EMA12）减去慢的加权移动均线（EMA26）
    DEA : DIF - (快线 - 慢线的9日加权移动均线DEA（时间远的赋予小权重，进的赋予大权重））得到MACD柱
    MACD --- 即由快、慢均线的离散、聚合表征当前的多空状态和股价可能的发展变化趋势，
             MACD从负数转向正数，是买的信号。当MACD从正数转向负数，是卖的信号
             MACD以大角度变化，表示快的移动平均线和慢的移动平均线的差距非常迅速的拉开表示转变
    '''
    _triple = True

    @classmethod
    def calc_feature(cls,feed,window):
        raw = feed.copy()
        if cls._triple and not isinstance(window,dict):
            raise TypeError('macd have fast,slow,period three window')
        fast_ema = EMA.calc_feature(raw,window['fast'])
        slow_ema = EMA.calc_feature(raw,window['slow'])
        dif = fast_ema - slow_ema
        dea = EMA.calc_feature(dif,window['period'])
        macd = dif - dea
        return macd


if __name__ == '__main__':

    date = '2019-06-01'
    edate = '2020-02-20'
    asset = '000001'
    window = 100

    # 分为三部分测试 1、close asset ,2、close ,3、ohlcv
    fields = ['close']
    event = Event(date,asset)
    req = GateReq(event,fields,window)
    quandle = Quandle()
    feed = quandle.query_ashare_kline(req)
    feed.index = pd.DatetimeIndex(feed.index)

    tema = TEMA.calc_feature(feed,5)
    print('tema',tema)

    dema = DEMA.calc_feature(feed,5)
    print('dema',dema)

    smma = SMMA.calc_feature(feed,5)
    print('smma',smma)

    ravi = RAVI.calc_feature(feed,[3,5])
    print('ravi',ravi)
    #
    dma = DMA.calc_feature(feed,[3,5])
    print('dma',dma)

    er = ER.calc_feature(feed,5)
    print('er',er)
    #
    ama = AMA.calc_feature(feed,5)
    print('ama',ama)
    #
    vema = VEMA.calc_feature(feed,5)
    print('vema',vema)
    #
    cmo = CMO.calc_feature(feed, 5)
    print('cmo', cmo)
    #
    speed = Speed.calc_feature(feed,5)
    print('speed',speed)
    #
    dpo = Dpo.calc_feature(feed,5)
    print('dpo',dpo)
    #
    mao = MAO.calc_feature(feed,(5,10))
    print('mao',mao)
    #
    vhf = VHF.calc_feature(feed,5)
    print('vhf',vhf)
    #
    rsi = Rsi.calc_feature(feed,5)
    print('rsi',rsi)
    #
    wr = WR.calc_feature(feed,5)
    print('wr',wr)
    #
    dmi = DMI.calc_feature(feed,(5,10))
    print('dmi',dmi)
    #
    ws = WS.calc_feature(feed,5)
    print('ws',ws)
    #
    trix = TRIX.calc_feature(feed,5)
    print('trix',trix)

    macd = Macd.calc_feature(feed, {'fast':5,'slow':10,'period':8})
    print('macd', macd)

    vidya = Vidya.calc_feature(feed,5)
    print('vidya',vidya)

# ------------------------------------------------------------------------------
    fields = ['open','close','high','low','volume']
    event = Event(date,asset)
    req = GateReq(event,fields,window)
    quandle = Quandle()
    feed = quandle.query_ashare_kline(req)
    feed.index = pd.DatetimeIndex(feed.index)

    #逻辑有问题
    jump = Jump.calc_feature(feed,5)
    print('jump',jump)

    massindex = MassIndex.calc_feature(feed,5)
    print('massindex',massindex)

    rvi = RVI.calc_feature(feed)
    print('rvi',rvi)

    smi = SMI.calc_feature(feed,5)
    print('smi',smi)

    rpower = RPower.calc_feature(feed,5)
    print('rpower',rpower)

    dpower = DPower.calc_feature(feed)
    print('dpower',dpower)

    mfi_ = MFI_.calc_feature(feed,5)
    print('mfi_',mfi_)

    pvt = PVT.calc_feature(feed,5)
    print('pvt',pvt)

    accumulate = Accumulation.calc_feature(feed,5)
    print('accumulate',accumulate)

    vck = VCK.calc_feature(feed,5)
    print('vck',vck)

    wad = WAD.calc_feature(feed,5)
    print('wad',wad)

    imi = IMI.calc_feature(feed)
    print('imi',imi)

    so = SO.calc_feature(feed,5)
    print('so',so)

    ri = RI.calc_feature(feed,5)
    print('ri',ri)

    adx = ADX.calc_feature(feed,5)
    print('adx',adx)

    mfi = MFI.calc_feature(feed,5)
    print('mfi',mfi)

    kvo = KVO.calc_feature(feed,(5,10))
    print('kvo',kvo)

    kdj = KDJ.calc_feature(feed,5)
    print('kdj',kdj)

# -------------------------------------------------------
    fields = ['close']
    event = Event(date)
    req = GateReq(event,fields,window)
    quandle = Quandle()
    feed = quandle.query_ashare_kline(req)

    aroon = Aroon.calc_feature(feed,5)
    print('aroon',aroon)

    fields = ['volume']
    event = Event(date,asset)
    req = GateReq(event,fields,window)
    quandle = Quandle()
    feed = quandle.query_ashare_kline(req)

    vo = VO.calc_feature(feed,(5,10))
    print('vo',vo)
