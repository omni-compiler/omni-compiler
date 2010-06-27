package exc.xcalablemp;

import exc.object.*;

public class XMPshadow {
  // defined in xmp_constant.h
  public final static int SHADOW_NONE	= 400;
  public final static int SHADOW_NORMAL	= 401;
  public final static int SHADOW_FULL	= 402;

  private int _type;
  private Xobject _lo;
  private Xobject _hi;

  public XMPshadow(int type, Xobject lo, Xobject hi) {
    _type = type;
    _lo = lo;
    _hi = hi;
  }

  public int getType() {
    return _type;
  }

  public Xobject getLo() {
    return _lo;
  }

  public Xobject getHi() {
    return _hi;
  }
}
