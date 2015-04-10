package exc.xcalablemp;
import exc.object.LineNo;

public class XMPexception extends Exception {
  public XMPexception() {
    super();
  }

  public XMPexception(String msg) {
    super(msg);
  }

  public XMPexception(LineNo lineNo, String msg) {
    super(lineNo + " : " + msg);
  }
}
