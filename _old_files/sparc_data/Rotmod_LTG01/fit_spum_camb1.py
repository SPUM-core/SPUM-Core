import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import os
import glob
import pandas as pd

# ==================== 常数 ====================
G = 4.30091e-6

# ==================== 读取 SPARC 6列数据（绝对正确） ====================
def load_sparc(filename):
    r, v, err, vg, vd, sb = [], [], [], [], [], []
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if len(p)>=6:
                r.append(float(p[0]))
                v.append(float(p[1]))
                err.append(float(p[2]))
                vg.append(float(p[3]))
                vd.append(float(p[4]))
                sb.append(float(p[5]))
    r = np.array(r)
    v = np.array(v)
    err = np.array(err)
    vg = np.array(vg)
    sb = np.array(sb)
    mask = r>0.01
    return r[mask], v[mask], err[mask], vg[mask], sb[mask]

# ==================== SPUM 模型（你原版物理） ====================
def spum(r, sb, vg, ups, lam, Vb, r0, beta):
    rs = np.maximum(r, 1e-3)
    dr = np.gradient(rs)
    dL = sb * 2*np.pi*rs*dr * 1e6
    Mstar = ups * np.cumsum(dL)
    Mgas = (vg**2 * rs)/G
    Mbar = Mstar + Mgas

    a_grav = G*Mbar/rs**2 * (1+rs/lam)*np.exp(-rs/lam)
    vb = Vb * rs / np.sqrt(rs**2 + r0**2)
    a_base = vb**2 / rs
    sbs = gaussian_filter1d(sb,1)
    a_light = beta * np.abs(np.gradient(sbs,rs))

    a = np.clip(a_grav + a_base + a_light, 1e-6, 1e6)
    return np.sqrt(rs * a)

# ==================== 单星系拟合 ====================
def fit_galaxy(file):
    try:
        r, v, err, vg, sb = load_sparc(file)
        if len(r)<5:
            return None

        def mod(rr, ups, lam, Vb, r0, beta):
            return spum(rr, sb, vg, ups, lam, Vb, r0, beta)

        vmax = v.max()
        p0 = [0.5, 8, 0.7*vmax, 10, 0.01]
        bnds = ([0.2,2,30,2,0], [1.5,30,400,30,0.1])

        popt,_ = curve_fit(mod, r, v, sigma=err, p0=p0, bounds=bnds, maxfev=15000)
        vfit = mod(r,*popt)

        # 画图（强制显示）
        plt.figure()
        plt.errorbar(r,v,yerr=err,fmt='ko',label='obs')
        plt.plot(r,vfit,'r-',linewidth=2,label='SPUM')
        plt.legend()
        plt.title(os.path.basename(file))
        plt.show()  # 强制弹出图

        return {
            'gal':os.path.basename(file).replace('_rotmod.dat',''),
            'chi2':np.sum(((v-vfit)/err)**2)/(len(v)-5)
        }
    except:
        return None

# ==================== 主程序：跑当前目录所有星系 ====================
if __name__ == '__main__':
    files = glob.glob("./*_rotmod.dat")
    print("找到文件数:", len(files))  # 一定会打印！
    
    if not files:
        print("没有找到星系数据！")
    else:
        ress = []
        for f in files:
            print("正在拟合:", f)
            res = fit_galaxy(f)
            if res:
                ress.append(res)
        
        pd.DataFrame(ress).to_csv("SPUM_output.csv", index=False)
        print("✅ 完成！结果已保存到 SPUM_output.csv")
        print("拟合成功星系数:", len(ress))
