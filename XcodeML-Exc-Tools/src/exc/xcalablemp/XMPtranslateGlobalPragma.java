/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.object.*;
import java.util.Vector;

public class XMPtranslateGlobalPragma {
  private XMPglobalDecl		_globalDecl;

  public XMPtranslateGlobalPragma(XMPglobalDecl globalDecl) {
    _globalDecl = globalDecl;
  }

  public void translate(Xobject x) {
    try {
      translatePragma(x);
    } catch (XMPexception e) {
      XMP.error(x.getLineNo(), e.getMessage());
    }
  }

  public void translatePragma(Xobject x) throws XMPexception {
    String pragmaName = x.getArg(0).getString();

    switch (XMPpragma.valueOf(pragmaName)) {
      case NODES:
        { translateNodes(x);		break; }
      case TEMPLATE:
        { translateTemplate(x);		break; }
      case DISTRIBUTE:
        { translateDistribute(x);	break; }
      case ALIGN:
        { translateAlign(x);		break; }
      case SHADOW:
        { translateShadow(x);		break; }
      default:
        throw new XMPexception("'" + pragmaName.toLowerCase() + "' directive is not supported yet");
    }
  }

  private void translateNodes(Xobject nodesPragma) throws XMPexception {
    XMPnodes.translateNodes((XobjList)nodesPragma.getArg(1), _globalDecl, false, null);
  }

  private void translateTemplate(Xobject templatePragma) throws XMPexception {
    XMPtemplate.translateTemplate((XobjList)templatePragma.getArg(1), _globalDecl, false, null);
  }

  private void translateDistribute(Xobject distributePragma) throws XMPexception {
    XMPtemplate.translateDistribute((XobjList)distributePragma.getArg(1), _globalDecl, false, null);
  }

  private void translateAlign(Xobject alignPragma) throws XMPexception {
    XMPalignedArray.translateAlign((XobjList)alignPragma.getArg(1), _globalDecl, false, null);
  }

  private void translateShadow(Xobject shadowPragma) throws XMPexception {
    XMPshadow.translateShadow((XobjList)shadowPragma.getArg(1), _globalDecl, false, null);
  }
}
