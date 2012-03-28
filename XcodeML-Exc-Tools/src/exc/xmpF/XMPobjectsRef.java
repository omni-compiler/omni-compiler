/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xmpF;

import exc.block.*;
import exc.object.*;

import java.util.Vector;

/* 
 * XMP ObjectsRef object 
 */
public class XMPobjectsRef {

  Ident descId;
  
  String refName;
  XMPobject refObject;	// nodes or template

  Vector<XMPdimInfo> subscripts;

  public XMPobjectsRef() {} // null constructor

  /* 
   * Nodes Reference:
   *  nodes_ref := (nodes-subscript) | nodes_anme(nodes-subscript, ...)
   *    nodes-subsript = int-expr| triplet| *
   *
   *   (LIST null (LIST lower upper stride))
   *   (LIST name (LIST (LIST lower upper stride) ...))
   * 
   * or Template Ref:
   */
  
  public static XMPobjectsRef parseDecl(Xobject decl,XMPenv env, PragmaBlock pb){
    XMPobjectsRef objRef = new XMPobjectsRef();
    objRef.parse(decl,env,pb);
    return objRef;
  }

  public XMPobject getRefObject() { return refObject; }

  public Vector<XMPdimInfo> getSubscripts() { return subscripts; }

  void parse(Xobject decl, XMPenv env, PragmaBlock pb) {
    if (decl.getArg(0) == null) {
      subscripts = new Vector<XMPdimInfo>();
      subscripts.add(XMPdimInfo.parseDecl(decl.getArg(1)));
      refName = "xmp_";
    } else {
      refName = decl.getArg(0).getString();
      refObject = env.getXMPobject(refName,pb);
      if (refObject == null) {
	XMP.error("cannot find objects '" + refName + "'");
	return;
      }
      Xobject subscriptList = decl.getArg(1);
      if(subscriptList != null){
	subscripts = XMPdimInfo.parseSubscripts(subscriptList);
      }
    }
    // allocate DescId
    descId = env.declObjectId(XMP.genSym(refName), pb);
  }

  // make contructor
  public Block buildConstructor(XobjectDef def){
    Block b = Bcons.emptyBlock();
    BasicBlock bb = b.getBasicBlock();
    Ident f = def.declExternIdent(XMP.onref_alloc_f,Xtype.FsubroutineType);
    Xobject args = Xcons.List(descId.Ref(),Xcons.IntConstant(subscripts.size()));
    bb.add(Xcons.List(Xcode.EXPR_STATEMENT,f.Call(args)));

    f = def.declExternIdent(XMP.onref_set_info_f,Xtype.FsubroutineType);
    for(int i = 0; i < subscripts.size(); i++){
      XMPdimInfo info = subscripts.elementAt(i);
      Xobject off = info.getLoopOnRefOffset();
      if(off == null) off = Xcons.IntConstant(0);
      args = Xcons.List(descId.Ref(),Xcons.IntConstant(i),
			Xcons.IntConstant(info.getLoopOnRefIndex()), off);
      bb.add(Xcons.List(Xcode.EXPR_STATEMENT,f.Call(args)));
    }
    
    f = def.declExternIdent(XMP.onref_init_f,Xtype.FsubroutineType);
    bb.add(Xcons.List(Xcode.EXPR_STATEMENT,f.Call(Xcons.List(descId.Ref()))));
    
    return b;
  }

  public Xobject buildLoopTestFuncCall(XobjectDef def, XMPinfo info){
    Ident f = def.declExternIdent(XMP.loop_test_f,Xtype.FlogicalFunctionType);
    Xobject args = Xcons.List(descId);
    for(int i = 0; i < info.getLoopDim(); i++) args.add(info.getLoopIndex(i));
    return f.Call(args);
  }
}