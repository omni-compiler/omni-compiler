package exc.xcalablemp;

public class XMPexception extends Exception {
  public XMPexception() {
    super();
  }

  public XMPexception(String msg) {
    super(msg);
  }

  public XMPexception(Throwable cause) {
    super(cause);
  }

  public XMPexception(String msg, Throwable cause) {
    super(msg, cause);
  }
}
