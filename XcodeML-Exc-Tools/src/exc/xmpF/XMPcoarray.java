/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

import exc.object.*;
import exc.block.*;
import java.util.Vector;
import java.util.Iterator;

/*
 * Fortran Coarray Object
 */
public class XMPcoarray {

  private XMParray array;
  private Xobject[] cosizeExprs;

  private Ident ident;

  public XMPcoarray(XMParray array) {
    this.array = array;
    this.cosizeExprs = null;
  }

  public XMPcoarray(XMParray array, Xobject[] cosizeExprs) {
    this.array = array;
    this.cosizeExprs = cosizeExprs;
  }

  public XMPcoarray(Ident ident) {
    this.ident = ident;
    //    this.cosizeExprs = ident.type.cosizeExprs;
  }

  public String toString(){
    String s = "{Coarray("+array.getName()+", id="+array.getArrayId()+"):";
    s += "dims="+array.getDim()+",";
    s += "codims="+cosizeExprs.length;
    return s+"}";
  }

  public String getName() {
    return array.getName();
  }

  public Xtype getType() {
    return array.getType();
  }

  public int getDim(){
    return array.getDim();
  }

  /* 
   * Method to translate a coarray declaration
   */
  public static void analyzeCoarray() {}

  /*
(Xobject a, Xobject arrayArgs,
				  Xobject templ, Xobject tempArgs,
				  XMPenv env, PragmaBlock pb){
    XMParray arrayObject = new XMParray();
    arrayObject.parseAlign(a,arrayArgs,templ,tempArgs,env,pb);
    env.declXMParray(arrayObject,pb);
  }
  */

}
