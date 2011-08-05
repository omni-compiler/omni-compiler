/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

public class XMPquadruplet<T1, T2, T3, T4> {
  private T1 _first;
  private T2 _second;
  private T3 _third;
  private T4 _forth;

  public XMPquadruplet(T1 first, T2 second, T3 third, T4 forth) {
    _first = first;
    _second = second;
    _third = third;
    _forth = forth;
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

  public T4 getForth() {
    return _forth;
  }
}
