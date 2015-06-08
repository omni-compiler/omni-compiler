/*
 * $TSUKUBA_Release: $
 * $TSUKUBA_Copyright:
 *  $
 */

package exc.xcalablemp;

import exc.block.*;
import exc.object.*;

public class XMPlocalDecl {
  private final static String XMP_SYMBOL_TABLE	= "XCALABLEMP_PROP_LOCAL_XMP_SYMBOL_TABLE";
  private final static String OBJECT_ID_LIST	= "XCALABLEMP_PROP_LOCAL_OBJECT_ID_LIST";
  private final static String CONSTRUCTOR	= "XCALABLEMP_PROP_LOCAL_CONSTRUCTOR";
  private final static String ALLOC		= "XCALABLEMP_PROP_LOCAL_ALLOC";
  private final static String DESTRUCTOR	= "XCALABLEMP_PROP_LOCAL_DESTRUCTOR";

  // FIXME move to this func to XMPglobalDecl
  public static void checkObjectNameCollision(String name, BlockList scopeBL, XMPsymbolTable objectTable) throws XMPexception {
    // check name collision - parameters
    if (scopeBL.findLocalIdent(name) != null)
      throw new XMPexception("'" + name + "' is already declared");

    // check name collision - local object table
    if (objectTable.getXMPobject(name) != null)
      throw new XMPexception("'" + name + "' is already declared");

    // check name collision - descriptor name
    if (scopeBL.findLocalIdent(XMP.DESC_PREFIX_ + name) != null) {
      // FIXME generate unique name
      throw new XMPexception("cannot declare template desciptor, '" + XMP.DESC_PREFIX_ + name + "' is already declared");
    }
  }

  public static FunctionBlock findParentFunctionBlock(Block block) {
    if (block == null) return null;

    for (Block b = block; b != null; b = b.getParentBlock())
      if (b.Opcode() == Xcode.FUNCTION_DEFINITION) return (FunctionBlock)b;

    return null;
  }

  public static Ident findLocalIdent(Block block, String name) {
    if (block == null) return null;

    for (Block b = block; b != null; b = b.getParentBlock()){
      Ident id = b.findVarIdent(name);
      if (id != null) return id;
    }

    return null;
  }

  public static void removeLocalIdent(Block block, String name){
    if (block == null) return;
    for (Block b = block; b != null; b = b.getParentBlock()){
      if (b.removeVarIdent(name)) return;
    }
  }
    
  public static XMPsymbolTable getXMPsymbolTable(Block block) {
    FunctionBlock fb = findParentFunctionBlock(block);
    if (fb == null) return null;
    else return (XMPsymbolTable)fb.getProp(XMP_SYMBOL_TABLE);
  }

  public static XMPsymbolTable declXMPsymbolTable(Block block) {
    FunctionBlock fb = findParentFunctionBlock(block);
    if (fb == null) return null;

    XMPsymbolTable table = (XMPsymbolTable)fb.getProp(XMP_SYMBOL_TABLE);
    if (table == null) {
      table = new XMPsymbolTable();
      fb.setProp(XMP_SYMBOL_TABLE, (Object)table);
    }

    return table;
  }

  public static Ident addObjectId(String objectName, Xtype type, Block block) {
    FunctionBlock fb = findParentFunctionBlock(block);

    XobjList idList = (XobjList)fb.getProp(OBJECT_ID_LIST);
    if (idList == null) {
      idList = Xcons.List(Xcode.LIST);
      fb.setProp(OBJECT_ID_LIST, (Object)idList);
    }

    Ident objectId = Ident.Local(objectName, type);
    idList.add(objectId);

    return objectId;
  }

  public static Ident addObjectId(String objectName, Block block) {
    return addObjectId(objectName, Xtype.voidPtrType, block);
  }

  public static void addConstructorCall(String funcName, Xobject funcArgs, XMPglobalDecl globalDecl, Block block) {
    Block fb = findParentFunctionBlock(block);

    XobjList bodyList = (XobjList)fb.getProp(CONSTRUCTOR);
    if(bodyList == null) {
      bodyList = Xcons.List(Xcode.LIST);
      fb.setProp(CONSTRUCTOR, (Object)bodyList);
    }

    Ident funcId = globalDecl.declExternFunc(funcName);
    bodyList.add(Xcons.List(Xcode.EXPR_STATEMENT, funcId.Call(funcArgs)));
  }

  public static void addAllocCall(String funcName, Xobject funcArgs, XMPglobalDecl globalDecl, Block block) {
    Block fb = findParentFunctionBlock(block);

    XobjList bodyList = (XobjList)fb.getProp(ALLOC);
    if(bodyList == null) {
      bodyList = Xcons.List(Xcode.LIST);
      fb.setProp(ALLOC, (Object)bodyList);
    }

    Ident funcId = globalDecl.declExternFunc(funcName);
    bodyList.add(Xcons.List(Xcode.EXPR_STATEMENT, funcId.Call(funcArgs)));
  }

  public static void insertDestructorCall(String funcName, Xobject funcArgs, XMPglobalDecl globalDecl, Block block) {
    FunctionBlock fb = findParentFunctionBlock(block);

    XobjList bodyList = (XobjList)fb.getProp(DESTRUCTOR);
    if(bodyList == null) {
      bodyList = Xcons.List(Xcode.LIST);
      fb.setProp(DESTRUCTOR, (Object)bodyList);
    }

    Ident funcId = globalDecl.declExternFunc(funcName);
    bodyList.cons(Xcons.List(Xcode.EXPR_STATEMENT, funcId.Call(funcArgs)));
  }

  // public static void setupObjectId(FunctionBlock functionBlock) {
  //   XobjList idList = (XobjList)functionBlock.getProp(OBJECT_ID_LIST);
  //   if (idList != null) {
  //     BlockList bl = functionBlock.getBody().getTail().getBody();
  //     for (XobjArgs i = idList.getArgs(); i != null; i = i.nextArgs()) {
  //       Ident id = (Ident)i.getArg();
  //       bl.addIdent(id);
  //     }
  //   }
  // }

  public static void setupObjectId(FunctionBlock functionBlock){

    topdownBlockIterator iter = new topdownBlockIterator(functionBlock);

    for (iter.init(); !iter.end(); iter.next()) {

      Block b = iter.getBlock();
      BlockList bl = b.getBody();

      XobjList idList = (XobjList)b.getProp(OBJECT_ID_LIST);
      if (idList != null) {
	for (XobjArgs i = idList.getArgs(); i != null; i = i.nextArgs()) {
	  Ident id = (Ident)i.getArg();
	  bl.addIdent(id);
	}
      }

    }

  }

  // public static void setupConstructor(FunctionBlock functionBlock)
  // {
  //   BlockList funcStmtList = functionBlock.getBody().getTail().getBody();

  //   XobjList alloc = (XobjList)functionBlock.getProp(ALLOC);
  //   if (alloc != null) {
  //     funcStmtList.insert(Bcons.buildBlock(Xcons.List(Xcode.COMPOUND_STATEMENT, (Xobject)null, null, alloc)));
  //   }

  //   XobjList constructor = (XobjList)functionBlock.getProp(CONSTRUCTOR);
  //   if (constructor != null) {
  //     funcStmtList.insert(Bcons.buildBlock(Xcons.List(Xcode.COMPOUND_STATEMENT, (Xobject)null, null, constructor)));
  //   }
  // }

  public static void setupConstructor(FunctionBlock functionBlock){

    topdownBlockIterator iter = new topdownBlockIterator(functionBlock);

    for (iter.init(); !iter.end(); iter.next()) {

      Block b = iter.getBlock();
      BlockList bl = b.getBody();

      XobjList alloc = (XobjList)b.getProp(ALLOC);
      if (alloc != null) {
	bl.insert(Bcons.buildBlock(Xcons.List(Xcode.COMPOUND_STATEMENT, (Xobject)null, null, alloc)));
      }

      XobjList constructor = (XobjList)b.getProp(CONSTRUCTOR);
      if (constructor != null) {
	bl.insert(Bcons.buildBlock(Xcons.List(Xcode.COMPOUND_STATEMENT, (Xobject)null, null, constructor)));
      }

    }

  }

  // public static void setupDestructor(FunctionBlock functionBlock) {
  //   XobjList destructor = (XobjList)functionBlock.getProp(DESTRUCTOR);
  //   if (destructor != null) {
  //     Block funcStmts = functionBlock.getBody().getTail();

  //     // insert destructor before return statements
  //     BlockIterator i = new topdownBlockIterator(funcStmts);
  //     for (i.init(); !i.end(); i.next()) {
  //       Block b = i.getBlock();
  //       if (b.Opcode() == Xcode.RETURN_STATEMENT)
  //         b.insert(Bcons.buildBlock(Xcons.List(Xcode.COMPOUND_STATEMENT, (Xobject)null, null, destructor)));
  //     }

  //     // add destructor at the end of the function
  //     funcStmts.getBody().add(Bcons.buildBlock(Xcons.List(Xcode.COMPOUND_STATEMENT, (Xobject)null, null, destructor)));
  //   }
  // }

  public static void setupDestructor(FunctionBlock functionBlock) {

    topdownBlockIterator iter = new topdownBlockIterator(functionBlock);

    for (iter.init(); !iter.end(); iter.next()) {

      Block b = iter.getBlock();
      BlockList bl = b.getBody();

      XobjList destructor = (XobjList)b.getProp(DESTRUCTOR);

      if (destructor != null) {

	// insert destructor before return statements
	BlockIterator i = new topdownBlockIterator(b);
	for (i.init(); !i.end(); i.next()) {
	  Block bb = i.getBlock();
	  if (bb.Opcode() == Xcode.RETURN_STATEMENT)
	    bb.insert(Bcons.buildBlock(Xcons.List(Xcode.COMPOUND_STATEMENT, (Xobject)null, null, destructor)));
	}
	
	// add destructor at the end of the function
	bl.add(Bcons.buildBlock(Xcons.List(Xcode.COMPOUND_STATEMENT, (Xobject)null, null, destructor)));
      }

    }

  }

  // add to the block

  public static XMPsymbolTable getXMPsymbolTable2(Block block) {
    if (block == null) return null;
    else return (XMPsymbolTable)block.getProp(XMP_SYMBOL_TABLE);
  }

  public static XMPsymbolTable declXMPsymbolTable2(Block block) {
    if (block == null) return null;

    XMPsymbolTable table = (XMPsymbolTable)block.getProp(XMP_SYMBOL_TABLE);
    if (table == null) {
      table = new XMPsymbolTable();
      block.setProp(XMP_SYMBOL_TABLE, (Object)table);
    }

    return table;
  }

  public static Ident addObjectId2(String objectName, Xtype type, Block block) {
    XobjList idList = (XobjList)block.getProp(OBJECT_ID_LIST);
    if (idList == null) {
      idList = Xcons.List(Xcode.LIST);
      block.setProp(OBJECT_ID_LIST, (Object)idList);
    }

    Ident objectId = Ident.Local(objectName, type);
    idList.add(objectId);

    return objectId;
  }

  public static Ident addObjectId2(String objectName, Block block) {
    return addObjectId2(objectName, Xtype.voidPtrType, block);
  }

  public static void addConstructorCall2(String funcName, Xobject funcArgs, XMPglobalDecl globalDecl, Block block) {
    XobjList bodyList = (XobjList)block.getProp(CONSTRUCTOR);
    if (bodyList == null){
      bodyList = Xcons.List(Xcode.LIST);
      block.setProp(CONSTRUCTOR, (Object)bodyList);
    }

    Ident funcId = globalDecl.declExternFunc(funcName);
    bodyList.add(Xcons.List(Xcode.EXPR_STATEMENT, funcId.Call(funcArgs)));
  }

  public static void addConstructorCall2_staticDesc(String funcName, Xobject funcArgs, XMPglobalDecl globalDecl, Block block,
						    Ident flagVar, boolean setFlag){
    XobjList bodyList = (XobjList)block.getProp(CONSTRUCTOR);
    if (bodyList == null){
      bodyList = Xcons.List(Xcode.LIST);
      block.setProp(CONSTRUCTOR, (Object)bodyList);
    }

    Ident funcId = globalDecl.declExternFunc(funcName);
    //bodyList.add(Xcons.List(Xcode.EXPR_STATEMENT, funcId.Call(funcArgs)));

    XobjList ifBody = Xcons.List();
    ifBody.add(Xcons.List(Xcode.EXPR_STATEMENT, funcId.Call(funcArgs)));
    if (setFlag) ifBody.add(Xcons.List(Xcode.EXPR_STATEMENT, Xcons.Set(flagVar.Ref(), Xcons.IntConstant(1))));
    Xobject body = Xcons.List(Xcode.IF_STATEMENT,
			      Xcons.unaryOp(Xcode.LOG_NOT_EXPR, flagVar.Ref()),
			      ifBody,
			      null);
    bodyList.add(body);

  }

  public static void addAllocCall2(String funcName, Xobject funcArgs, XMPglobalDecl globalDecl, Block block) {
    XobjList bodyList = (XobjList)block.getProp(ALLOC);
    if (bodyList == null){
      bodyList = Xcons.List(Xcode.LIST);
      block.setProp(ALLOC, (Object)bodyList);
    }

    Ident funcId = globalDecl.declExternFunc(funcName);
    bodyList.add(Xcons.List(Xcode.EXPR_STATEMENT, funcId.Call(funcArgs)));
  }

  public static void insertDestructorCall2(String funcName, Xobject funcArgs, XMPglobalDecl globalDecl, Block block) {
    XobjList bodyList = (XobjList)block.getProp(DESTRUCTOR);
    if(bodyList == null) {
      bodyList = Xcons.List(Xcode.LIST);
      block.setProp(DESTRUCTOR, (Object)bodyList);
    }

    Ident funcId = globalDecl.declExternFunc(funcName);
    bodyList.cons(Xcons.List(Xcode.EXPR_STATEMENT, funcId.Call(funcArgs)));
  }

}
