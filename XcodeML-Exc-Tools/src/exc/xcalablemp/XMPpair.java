/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

public class XMPpair<T1, T2> {
  private T1 _first;
  private T2 _second;

  public XMPpair(T1 first, T2 second) {
    _first = first;
    _second = second;
  }

  public T1 getFirst() {
    return _first;
  }

  public T2 getSecond() {
    return _second;
  }
}
