/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

public class XMPcollective {
  // defined in xmp_constant.h
  public final static int REDUCE_SUM		= 300;
  public final static int REDUCE_PROD		= 301;
  public final static int REDUCE_BAND		= 302;
  public final static int REDUCE_LAND		= 303;
  public final static int REDUCE_BOR		= 304;
  public final static int REDUCE_LOR		= 305;
  public final static int REDUCE_BXOR		= 306;
  public final static int REDUCE_LXOR		= 307;
  public final static int REDUCE_MAX		= 308;
  public final static int REDUCE_MIN		= 309;
  public final static int REDUCE_FIRSTMAX	= 310;
  public final static int REDUCE_FIRSTMIN	= 311;
  public final static int REDUCE_LASTMAX	= 312;
  public final static int REDUCE_LASTMIN	= 313;
  public final static int REDUCE_EQV            = 314;
  public final static int REDUCE_NEQV           = 315;
  public final static int REDUCE_MINUS          = 316;

  public final static int GMOVE_NORMAL	= 400;
  public final static int GMOVE_IN	= 401;
  public final static int GMOVE_OUT	= 402;
}
