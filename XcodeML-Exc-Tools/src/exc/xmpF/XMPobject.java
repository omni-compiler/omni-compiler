/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

import exc.object.*;
import exc.block.*;

/*
 * Base class of XMPnodes and XMPtemplate
 */
public class XMPobject {
  public final static int NODES		= 100;
  public final static int TEMPLATE	= 101;

  int			_kind;
  String		_name;
  int			_dim;
  Ident			_descId;

  public XMPobject(int kind){
    _kind = kind;
  }

  public XMPobject(int kind, String name, int dim, Ident descId) {
    _kind = kind;
    _name = name;
    _dim = dim;
    _descId = descId;
  }

  public int getKind() {
    return _kind;
  }

  public String getName() {
    return _name;
  }

  public int getDim() {
    return _dim;
  }

  public Ident getDescId() {
    return _descId;
  }

  public void buildConstructor(BlockList body, XMPenv def){
    // abstract method
    XMP.fatal("buildConstrutor");
  }

  public void buildDestructor(BlockList body, XMPenv def){
    // abstract method
    XMP.fatal("buildDestrutor");
  }  
}
