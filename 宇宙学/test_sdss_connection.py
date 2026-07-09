"""SDSS FITS structure inspection"""
from astroquery.sdss import SDSS
import warnings
warnings.filterwarnings('ignore')

spec = SDSS.get_spectra(plate=2241, mjd=54156, fiberID=502, timeout=180)
print(f"Return type: {type(spec)}, len={len(spec)}")

sp = spec[0]
print(f"Item type: {type(sp)}")
if isinstance(sp, tuple):
    print(f"Tuple len: {len(sp)}")
    hdu = sp[0]
    meta = sp[1] if len(sp) > 1 else {}
    print(f"Meta: {meta}")
else:
    hdu = sp

print(f"HDU type: {type(hdu)}")
print(f"HDU info:")
hdu.info()

# Check all extensions
for i in range(len(hdu)):
    ext = hdu[i]
    print(f"  Ext {i}: name={ext.name}, type={type(ext.data)}, shape={ext.data.shape if hasattr(ext.data, 'shape') else 'no data'}")
    print(f"     Header keys: RA={ext.header.get('RA', '?')}, DEC={ext.header.get('DEC', '?')}")
    print(f"     COEFF0={ext.header.get('COEFF0', '?')}, COEFF1={ext.header.get('COEFF1', '?')}")
