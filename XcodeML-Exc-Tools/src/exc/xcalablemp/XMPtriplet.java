/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

public class XMPtriplet<T1, T2, T3> {
  private T1 _first;
  private T2 _second;
  private T3 _third;

  public XMPtriplet(T1 first, T2 second, T3 third) {
    _first = first;
    _second = second;
    _third = third;
  }

  public T1 getFirst() {
    return _first;
  }

  public T2 getSecond() {
    return _second;
  }

  public T3 getThird() {
    return _third;
  }
}
