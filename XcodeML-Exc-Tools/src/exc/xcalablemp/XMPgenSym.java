/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

public class XMPgenSym {
  private int _num;

  public XMPgenSym() {
    _num = 0;
  }

  public String getStr(String prefix) {
    String newString = new String(prefix + String.valueOf(_num));
    _num++;
    return newString;
  }

}
