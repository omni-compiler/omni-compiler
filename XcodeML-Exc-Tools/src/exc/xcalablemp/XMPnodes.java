package exc.xcalablemp;

import exc.object.*;

public class XMPnodes extends XMPobject {
  public final static int INHERIT_GLOBAL	= 10;
  public final static int INHERIT_EXEC		= 11;
  public final static int INHERIT_NODES		= 12;

  public final static int MAP_UNDEFINED		= 20;
  public final static int MAP_REGULAR		= 21;

  public XMPnodes(int lineNo, String name, int dim, Ident descId) {
    super(XMPobject.NODES, lineNo, name, dim, descId);
  }
}
