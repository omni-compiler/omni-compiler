package exc.xcalablemp;

import exc.block.*;
import exc.object.*;
import exc.openacc.ACCpragma;
import java.util.*;

public class XMPrewriteExpr {
  private XMPglobalDecl		_globalDecl;

  public XMPrewriteExpr(XMPglobalDecl globalDecl) {
    _globalDecl = globalDecl;
  }

  public void rewrite(FuncDefBlock def) {
    FunctionBlock fb = def.getBlock();
    if (fb == null) return;

    // get symbol table
    XMPsymbolTable localXMPsymbolTable = XMPlocalDecl.declXMPsymbolTable(fb);

    // rewrite parameters
    rewriteParams(fb, localXMPsymbolTable);

    // rewrite declarations
    rewriteDecls(fb, localXMPsymbolTable);

    rewriteStmts(fb, localXMPsymbolTable);

    // rewrite Function Exprs
    rewriteFuncExprs(fb, localXMPsymbolTable);

    // rewrite OMP pragma
    rewriteOMPpragma(fb, localXMPsymbolTable);
    
    // rewrite ACC pragma
    rewriteACCpragma(fb, localXMPsymbolTable);

    // create local object descriptors, constructors and desctructors
    XMPlocalDecl.setupObjectId(fb);
    XMPlocalDecl.setupConstructor(fb);
    XMPlocalDecl.setupDestructor(fb);

    // add a barrier at the end of the original main
    if (fb.getName() == "main") addBarrier(fb);

    def.Finalize();
  }

  private void rewriteParams(FunctionBlock funcBlock, XMPsymbolTable localXMPsymbolTable) {
    XobjList identList = funcBlock.getBody().getIdentList();
    if (identList == null) {
      return;
    } else {
      for(Xobject x : identList) {
        Ident id = (Ident)x;
        XMPalignedArray alignedArray = localXMPsymbolTable.getXMPalignedArray(id.getName());
        if (alignedArray != null) {
          id.setType(Xtype.Pointer(alignedArray.getType()));
        }
      }
    }
  }

  private void rewriteDecls(FunctionBlock funcBlock, XMPsymbolTable localXMPsymbolTable) {
    topdownBlockIterator iter = new topdownBlockIterator(funcBlock);
    for (iter.init(); !iter.end(); iter.next()) {
      Block b = iter.getBlock();
      BlockList bl = b.getBody();

      if (bl != null) {
        XobjList decls = (XobjList)bl.getDecls();
        if (decls != null) {
          try {
            for (Xobject x : decls) {
              Xobject declInitExpr = x.getArg(1);
	      x.setArg(1, rewriteExpr(declInitExpr, b));
            }
          } catch (XMPexception e) {
            XMP.error(b.getLineNo(), e.getMessage());
          }
        }
      }
    }
  }

  private void rewriteFuncExprs(FunctionBlock funcBlock, XMPsymbolTable localXMPsymbolTable) {
    // insert TASK descripter for cache mechanism.
    //    if(_globalDecl.findVarIdent(funcBlock.getName()).Type().isInline() == false){
      // This decleartion is inserted into the first point of each function.
      //      BlockList taskBody = funcBlock.getBody().getHead().getBody();
      //      Ident taskDescId = taskBody.declLocalIdent("_XMP_TASK_desc", Xtype.voidPtrType, StorageClass.AUTO,
      //                                                 Xcons.Cast(Xtype.voidPtrType, Xcons.IntConstant(0)));
      
      // insert Finalize function into the last point of each function.
      //      XobjList arg = Xcons.List(Xcode.POINTER_REF, taskDescId.Ref());
      //      Ident taskFuncId = _globalDecl.declExternFunc("_XMP_exec_task_NODES_FINALIZE");
      //      taskBody.add(taskFuncId.Call(arg));
      
      // insert Finalize function into the previous point of return statement
      //      BlockIterator i = new topdownBlockIterator(taskBody);
      //      for (i.init(); !i.end(); i.next()) {
      //      	Block b = i.getBlock();
      //      	if (b.Opcode() == Xcode.RETURN_STATEMENT){
      //          b.ainsert(taskFuncId.Call(arg));
      //        }
      //      }
      //    }

    BasicBlockExprIterator iter = new BasicBlockExprIterator(funcBlock);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject expr = iter.getExpr();
      try {
        switch (expr.Opcode()) {
          case ASSIGN_EXPR:
	    iter.setExpr(rewriteAssignExpr(expr, iter.getBasicBlock().getParent(), localXMPsymbolTable, iter));
            break;
          default:
	    iter.setExpr(rewriteExpr(expr, iter.getBasicBlock().getParent()));
            break;
        }
      } catch (XMPexception e) {
        XMP.error(expr.getLineNo(), e.getMessage());
      }
    }
  }

  private Xobject rewriteAssignExpr(Xobject myExpr, Block exprParentBlock, XMPsymbolTable localXMPsymbolTable,
                                    BasicBlockExprIterator iter) throws XMPexception {
    assert myExpr.Opcode() == Xcode.ASSIGN_EXPR;

    Xobject leftExpr = myExpr.getArg(0);
    Xobject rightExpr = myExpr.getArg(1);

    if ((leftExpr.Opcode() == Xcode.CO_ARRAY_REF) && (rightExpr.Opcode() == Xcode.CO_ARRAY_REF)) {   // a:[1] = b[2];   // Fix me
      throw new XMPexception("unknown co-array expression"); 
    } 
    else if ((leftExpr.Opcode() == Xcode.CO_ARRAY_REF) || (rightExpr.Opcode() == Xcode.CO_ARRAY_REF)) {
      return rewriteCoarrayAssignExpr(myExpr, exprParentBlock, localXMPsymbolTable, iter);
    }
    // else if (leftExpr.Opcode() == Xcode.SUB_ARRAY_REF){
    //   return rewriteSubArrayAssignExpr(myExpr, exprParentBlock);
    // }
    else {
      return rewriteExpr(myExpr, exprParentBlock);
    }
  }


  // private Xobject rewriteSubArrayAssignExpr(Xobject assignStmt, Block b) throws XMPexception {

  //   Xobject left = assignStmt.left();

  //   assert left.Opcode() == Xcode.SUB_ARRAY_REF;

  //   List<Ident> varList = new ArrayList<Ident>(XMP.MAX_DIM);
  //   List<Ident> varListTemplate = new ArrayList<Ident>(XMP.MAX_DIM);
  //   for (int i = 0; i < XMP.MAX_DIM; i++) varListTemplate.add(null);
  //   List<Xobject> lbList = new ArrayList<Xobject>(XMP.MAX_DIM);
  //   List<Xobject> lenList = new ArrayList<Xobject>(XMP.MAX_DIM);
  //   List<Xobject> stList = new ArrayList<Xobject>(XMP.MAX_DIM);

  //   //
  //   // convert LHS
  //   //

  //   String arrayName = left.getArg(0).getSym();

  //   Xtype arrayType = null;
  //   Ident arrayId = b.findVarIdent(arrayName);
  //   if (arrayId != null){
  //     arrayType = arrayId.Type();
  //   }
	
  //   if (arrayType == null) throw new XMPexception("array should be declared statically");

  //   Xtype elemType = arrayType.getArrayElementType();
  //   int n = arrayType.getNumDimensions();

  //   XobjList subscripts = (XobjList)left.getArg(1);

  //   for (int i = 0; i < n; i++, arrayType = arrayType.getRef()){

  //     long dimSize = arrayType.getArraySize();
  //     Xobject sizeExpr;
  //     if (dimSize == 0 || arrayType.getKind() == Xtype.POINTER){
  // 	throw new XMPexception("array should be declared statically");
  //     }
  //     else if (dimSize == -1){
  //       sizeExpr = arrayType.getArraySizeExpr();
  //     }
  //     else {
  // 	sizeExpr = Xcons.LongLongConstant(0, dimSize);
  //     }

  //     Xobject sub = subscripts.getArg(i);

  //     Ident var;
  //     Xobject lb, len, st;

  //     if (sub.Opcode() != Xcode.LIST) continue;

  //     var = XMPtranslateLocalPragma.declIdentWithBlock(b, "_XMP_loop_i" + Integer.toString(i), Xtype.intType);
  //     varList.add(var);

  //     lb = ((XobjList)sub).getArg(0);
  //     if (lb == null) lb = Xcons.IntConstant(0);
  //     len = ((XobjList)sub).getArg(1);
  //     if (len == null) len = sizeExpr;
  //     st = ((XobjList)sub).getArg(2);
  //     if (st == null) st = Xcons.IntConstant(1);

  //     lbList.add(lb);
  //     lenList.add(len);
  //     stList.add(st);

  //     Xobject expr;
  //     expr = Xcons.binaryOp(Xcode.MUL_EXPR, var.Ref(), st);
  //     expr = Xcons.binaryOp(Xcode.PLUS_EXPR, expr, lb);

  //     subscripts.setArg(i, expr);

  //   }

  //   Xobject new_left = Xcons.arrayRef(elemType, left.getArg(0), subscripts);

  //   //
  //   // convert RHS
  //   //

  //   // NOTE: Since the top level object cannot be replaced, the following conversion is applied to
  //   //       the whole assignment.
  //   XobjectIterator j = new topdownXobjectIterator(assignStmt);
  //   for (j.init(); !j.end(); j.next()) {
  //     Xobject x = j.getXobject();

  //     if (x.Opcode() != Xcode.SUB_ARRAY_REF) continue;

  //     int k = 0;

  //     String arrayName1 = x.getArg(0).getSym();

  //     XMPalignedArray array1 = _globalDecl.getXMPalignedArray(arrayName1, b);
  //     Xtype arrayType1 = null;
  //     if (array1 != null){
  // 	arrayType1 = array1.getArrayType();
  //     }
  //     else {
  // 	Ident arrayId1 = b.findVarIdent(arrayName1);
  // 	if (arrayId1 != null){
  // 	  arrayType1 = arrayId1.Type();
  // 	}
  //     }
	
  //     if (arrayType1 == null) throw new XMPexception("array should be declared statically");

  //     Xtype elemType1 = arrayType1.getArrayElementType();
  //     int m = arrayType1.getNumDimensions();

  //     XobjList subscripts1 = (XobjList)x.getArg(1);

  //     for (int i = 0; i < m; i++, arrayType1 = arrayType1.getRef()){

  // 	Xobject sub = subscripts1.getArg(i);

  // 	Ident var;
  // 	Xobject lb, st;

  // 	if (sub.Opcode() != Xcode.LIST) continue;

  // 	lb = ((XobjList)sub).getArg(0);
  // 	if (lb == null) lb = Xcons.IntConstant(0);
  // 	st = ((XobjList)sub).getArg(2);
  // 	if (st == null) st = Xcons.IntConstant(1);

  // 	Xobject expr;
  // 	expr = Xcons.binaryOp(Xcode.MUL_EXPR, varList.get(k).Ref(), st);
  // 	expr = Xcons.binaryOp(Xcode.PLUS_EXPR, expr, lb);

  // 	subscripts1.setArg(i, expr);
  // 	k++;
  //     }

  //     Xobject new_x = Xcons.arrayRef(elemType1, x.getArg(0), subscripts1);
  //     j.setXobject(new_x);

  //   }

  //   //
  //   // construct loop
  //   //

  //   BlockList loop = null;

  //   BlockList body = Bcons.emptyBody();
  //   body.add(Xcons.Set(new_left, assignStmt.right()));

  //   for (int i = varList.size() - 1; i >= 0; i--){
  //     loop = Bcons.emptyBody();
  //     loop.add(Bcons.FORall(varList.get(i).Ref(), Xcons.IntConstant(0), lenList.get(i), Xcons.IntConstant(1),
  // 			    Xcode.LOG_LT_EXPR, body));
  //     body = loop;
  //   }

  //   return Bcons.COMPOUND(loop).toXobject();

  // }


  private Xobject createShortcutCoarray(int imageDims, XobjList imageList, String commkind, 
                                        XMPcoarray dstCoarray, XMPcoarray srcCoarray,
                                        Xobject dstCoarrayExpr, Xobject srcCoarrayExpr,
                                        boolean isDstCoarrayOnAcc, boolean isSrcCoarrayOnAcc) throws XMPexception
  // dstCoarray is left expression. srcCoarray is right expression.
  {
    // Set Function Name
    // If image set is 2 dimension and Put operation,
    // function name is "_XMP_shortcut_put_image2"
    boolean isAcc = isDstCoarrayOnAcc || isSrcCoarrayOnAcc;
    String funcName = "_XMP_coarray_shortcut_" + commkind;
    if(isAcc) funcName += "_acc";
    Ident funcId = _globalDecl.declExternFunc(funcName);
    XobjList funcArgs = Xcons.List();

    // Set target image
    XMPcoarray remoteCoarray;
    if(commkind == "put"){
      remoteCoarray = dstCoarray;
    } else{
      remoteCoarray = srcCoarray;
    }
    
    // Distance Image to need increment in each dimension
    // e.g.) a:[4][3][2][*];
    //       remoteImageDistance[0] = 4 * 3 * 2;
    //       remoteImageDistance[1] = 4 * 3;
    //       remoteImageDistance[2] = 4;
    //       remoteImageDistance[3] = 1; // Note: the last dimension must be 1.
    int[] remoteImageDistance = new int[imageDims];
    for(int i=0;i<imageDims-1;i++){
      remoteImageDistance[i] = 1;
      for(int j=0;j<imageDims-1-i;j++){
        remoteImageDistance[i] *= remoteCoarray.getImageAt(j);
      }
    }
    remoteImageDistance[imageDims-1] = 1;

    //    Xobject targetImage = Xcons.binaryOp(Xcode.MINUS_EXPR, imageList.getArg(0), Xcons.IntConstant(1));
    Xobject targetImage = imageList.getArg(0);
    for(int i=1;i<imageDims;i++){
      Xobject tmp = Xcons.binaryOp(Xcode.MUL_EXPR, 
                                   Xcons.binaryOp(Xcode.MINUS_EXPR, imageList.getArg(i), Xcons.IntConstant(1)),
                                   Xcons.IntConstant(remoteImageDistance[imageDims-1-i]));
      targetImage = Xcons.binaryOp(Xcode.PLUS_EXPR, tmp, targetImage);
    }
    funcArgs.add(targetImage);

    // Set Coarray Descriptor
    funcArgs.add(Xcons.SymbolRef(dstCoarray.getDescId()));
    funcArgs.add(Xcons.SymbolRef(srcCoarray.getDescId()));
    
    // Number of elements to need increment in each dimension
    // e.g.) a[3][4][5][6];
    //       xxxCoarrayDistance[0] = 4 * 5 * 6;
    //       xxxCoarrayDistance[1] = 5 * 6;
    //       xxxCoarrayDistance[2] = 6;
    //       xxxCoarrayDistance[3] = 1; // Note: the last dimension must be 1.
    int dstDim = dstCoarray.getVarDim();
    int srcDim = srcCoarray.getVarDim();
    int[] dstCoarrayDistance = new int[dstDim];
    int[] srcCoarrayDistance = new int[srcDim];

    if(dstCoarrayExpr.Opcode() == Xcode.VAR){
      dstCoarrayDistance[0] = 1;
    }
    else{
      for(int i=0;i<dstDim;i++){
        dstCoarrayDistance[i] = 1;
        for(int j=i+1;j<dstDim;j++){
          dstCoarrayDistance[i] *= (int)dstCoarray.getSizeAt(j);
        }
      }
    }

    if(srcCoarrayExpr.Opcode() == Xcode.VAR){
      srcCoarrayDistance[0] = 1;
    }
    else{
      for(int i=0;i<srcDim;i++){
        srcCoarrayDistance[i] = 1;
        for(int j=i+1;j<srcDim;j++){
          srcCoarrayDistance[i] *= (int)srcCoarray.getSizeAt(j);
        }
      }
    }

    // How depth continuous ?
    // e.g.) a[100][100][100]:[*];
    //       a[:][:][:],   xxxCoarrayDepthContinuous = 0
    //       a[2][:][:],   xxxCoarrayDepthContinuous = 1
    //       a[:2][:][:],  xxxCoarrayDepthContinuous = 1
    //       a[2][2][:],   xxxCoarrayDepthContinuous = 2
    //       a[:][:2][:],  xxxCoarrayDepthContinuous = 2
    //       a[2][2][2:2], xxxCoarrayDepthContinuous = 3
    //       a[2][2][2:],  xxxCoarrayDepthContinuous = 3
    //       a[2][2][:],   xxxCoarrayDepthContinuous = 3
    //       a[2][2][1],   xxxCoarrayDepthContinuous = 3

    // dstCoarray
    int dstCoarrayDepthContinuous = dstDim;
    if(dstCoarrayExpr.Opcode() == Xcode.SUB_ARRAY_REF){
      Ident varId = dstCoarray.getVarId();
      XobjList tripletList = (XobjList)dstCoarrayExpr.getArg(1);
      for(int i=dstDim-1;i>=0;i--){
        if(is_all_element(i, tripletList, varId)){
          dstCoarrayDepthContinuous = i;
        }
      }
    }
    // if dstCoarray == Xcode.ARRAY_REF or Xcode.VAR,
    // dstCoarrayDepthContinuous = 1.
    
    int srcCoarrayDepthContinuous = srcDim;
    if(srcCoarrayExpr.Opcode() == Xcode.SUB_ARRAY_REF){
      Ident varId = srcCoarray.getVarId();
      XobjList tripletList = (XobjList)srcCoarrayExpr.getArg(1);
      for(int i=srcDim-1;i>=0;i--){
        if(is_all_element(i, tripletList, varId)){
          srcCoarrayDepthContinuous = i;
        }
      }
    }

    // dst offset
    Xobject position = null;
    if(dstCoarrayExpr.Opcode() == Xcode.SUB_ARRAY_REF){
      for(int i=0;i<dstDim;i++){
        Xobject tripletList = dstCoarrayExpr.getArg(1).getArg(i);
        Xobject tmp_position;
        if(tripletList.isConstant() || tripletList.isVariable()){
          tmp_position = Xcons.binaryOp(Xcode.MUL_EXPR, tripletList, Xcons.IntConstant(dstCoarrayDistance[i]));
        }
        else{
          Xobject start = ((XobjList)tripletList).getArg(0);
          tmp_position = Xcons.binaryOp(Xcode.MUL_EXPR, start, Xcons.IntConstant(dstCoarrayDistance[i]));
        }
        if(i == 0){
          position = tmp_position;
        }
        else{
          position = Xcons.binaryOp(Xcode.PLUS_EXPR, position, tmp_position);
        }
      }
    }
    else if(dstCoarrayExpr.Opcode() == Xcode.ARRAY_REF){
      for(int i=0;i<dstDim;i++){
        Xobject tmp_position = Xcons.binaryOp(Xcode.MUL_EXPR, dstCoarrayExpr.getArg(1).getArg(i),
                                              Xcons.IntConstant(dstCoarrayDistance[i]));
        if(i==0){
          position = tmp_position;
        }
        else{
          position = Xcons.binaryOp(Xcode.PLUS_EXPR, position, tmp_position);
        }
      }
    }
    else if(dstCoarrayExpr.Opcode() == Xcode.VAR){
      position = Xcons.IntConstant(0);
    }
    else{
      throw new XMPexception("Not supported this coarray Syntax");
    }
    Xtype elmtType = dstCoarray.getElmtType();
    position = Xcons.binaryOp(Xcode.MUL_EXPR, position, Xcons.SizeOf(elmtType));
    funcArgs.add(position);

    // src offset
    position = null;
    if(srcCoarrayExpr.Opcode() == Xcode.SUB_ARRAY_REF){
      for(int i=0;i<srcDim;i++){
        Xobject tripletList = srcCoarrayExpr.getArg(1).getArg(i);
        Xobject tmp_position;
        if(tripletList.isConstant() || tripletList.isVariable()){
          tmp_position = Xcons.binaryOp(Xcode.MUL_EXPR, tripletList, Xcons.IntConstant(srcCoarrayDistance[i]));
        }
        else{
          Xobject start = ((XobjList)tripletList).getArg(0);
          tmp_position = Xcons.binaryOp(Xcode.MUL_EXPR, start, Xcons.IntConstant(srcCoarrayDistance[i]));
        }
        if(i == 0){
          position = tmp_position;
        }
        else{
          position = Xcons.binaryOp(Xcode.PLUS_EXPR, position, tmp_position);
        }
      }
    }
    else if(srcCoarrayExpr.Opcode() == Xcode.ARRAY_REF){
      for(int i=0;i<srcDim;i++){
        Xobject tmp_position = Xcons.binaryOp(Xcode.MUL_EXPR, srcCoarrayExpr.getArg(1).getArg(i),
                                              Xcons.IntConstant(srcCoarrayDistance[i]));
        if(i==0){
          position = tmp_position;
        }
        else{
          position = Xcons.binaryOp(Xcode.PLUS_EXPR, position, tmp_position);
        }
      }
    }
    else if(srcCoarrayExpr.Opcode() == Xcode.VAR){
      position = Xcons.IntConstant(0);
    }
    else{
      throw new XMPexception("Not supported this coarray Syntax");
    }
    position = Xcons.binaryOp(Xcode.MUL_EXPR, position, Xcons.SizeOf(elmtType));
    funcArgs.add(position);

    // dst_length
    Xobject dst_length = null;
    if(dstCoarrayExpr.Opcode() == Xcode.SUB_ARRAY_REF){
      if(dstCoarrayDepthContinuous == 0){
        dst_length = Xcons.IntConstant((int)dstCoarray.getSizeAt(0) * dstCoarrayDistance[0]);
      }
      else{
        Xobject tripletList = dstCoarrayExpr.getArg(1).getArg(dstCoarrayDepthContinuous-1);
        if(tripletList.isConstant() || tripletList.isVariable()){
          dst_length = Xcons.IntConstant(dstCoarrayDistance[dstCoarrayDepthContinuous-1]);
        }
        else{
          dst_length = Xcons.binaryOp(Xcode.MUL_EXPR, ((XobjList)tripletList).getArg(1),
                                  Xcons.IntConstant(dstCoarrayDistance[dstCoarrayDepthContinuous-1]));
        }
      }
    }
    else if(dstCoarrayExpr.Opcode() == Xcode.ARRAY_REF || dstCoarrayExpr.Opcode() == Xcode.VAR){
      dst_length = Xcons.IntConstant(1);
    }
    else{
      throw new XMPexception("Not supported this coarray Syntax");
    }
    funcArgs.add(dst_length);

    // src_length
    Xobject src_length = null;
    if(srcCoarrayExpr.Opcode() == Xcode.SUB_ARRAY_REF){
      if(srcCoarrayDepthContinuous == 0){
        src_length = Xcons.IntConstant((int)srcCoarray.getSizeAt(0) * srcCoarrayDistance[0]);
      }
      else{
        Xobject tripletList = srcCoarrayExpr.getArg(1).getArg(srcCoarrayDepthContinuous-1);
        if(tripletList.isConstant() || tripletList.isVariable()){
          src_length = Xcons.IntConstant(srcCoarrayDistance[srcCoarrayDepthContinuous-1]);
        }
        else{
          src_length = Xcons.binaryOp(Xcode.MUL_EXPR, ((XobjList)tripletList).getArg(1),
                                      Xcons.IntConstant(srcCoarrayDistance[srcCoarrayDepthContinuous-1]));
        }
      }
    }
    else if(srcCoarrayExpr.Opcode() == Xcode.ARRAY_REF || srcCoarrayExpr.Opcode() == Xcode.VAR){
      src_length = Xcons.IntConstant(1);
    }
    else{
      throw new XMPexception("Not supported this coarray Syntax");
    }
    funcArgs.add(src_length);

    if(isAcc){
      //add location of coarray (host=0, acc=1)
      funcArgs.add(Xcons.IntConstant(isDstCoarrayOnAcc? 1 : 0));
      funcArgs.add(Xcons.IntConstant(isSrcCoarrayOnAcc? 1 : 0));
    }

    // Create function
    Xobject newExpr = funcId.Call(funcArgs);
    newExpr.setIsRewrittedByXmp(true);
    return newExpr;
  }

  private Xobject rewriteCoarrayAssignExpr(Xobject myExpr, Block exprParentBlock, XMPsymbolTable localXMPsymbolTable, 
                                           BasicBlockExprIterator iter) throws XMPexception {
    assert myExpr.Opcode() == Xcode.ASSIGN_EXPR;

    Xobject leftExpr    = myExpr.getArg(0);
    Xobject rightExpr   = myExpr.getArg(1);
    Xobject coarrayExpr = null;
    Xobject localExpr   = null;
    if(leftExpr.Opcode() == Xcode.CO_ARRAY_REF){  // PUT
      coarrayExpr = leftExpr;
      localExpr   = rightExpr;
    } else{                                       // GET
      coarrayExpr = rightExpr;
      localExpr   = leftExpr;
    }

    String coarrayName = XMPutil.getXobjSymbolName(coarrayExpr.getArg(0));
    XMPcoarray coarray = _globalDecl.getXMPcoarray(coarrayName, exprParentBlock);
    if(coarray == null){
      throw new XMPexception("cannot find coarray '" + coarrayName + "'");
    }

    // Get Coarray Dims
    XobjList funcArgs = Xcons.List();
    int coarrayDims  = coarray.getVarDim();

    // Get Local Dims
    boolean isArray;
    int localDims;
    String localName;
    if(localExpr.Opcode() == Xcode.SUB_ARRAY_REF || localExpr.Opcode() == Xcode.ARRAY_REF){
      isArray = true;
      localName = localExpr.getArg(0).getName();
      Ident varId = localExpr.findVarIdent(localName);
      localDims = varId.Type().getNumDimensions();
    }
    else if(localExpr.Opcode() == Xcode.VAR || localExpr.Opcode() == Xcode.POINTER_REF){
      if(localExpr.Opcode() == Xcode.VAR)
        localName = localExpr.getName();
      else
        localName = localExpr.getArg(0).getName();
      
      isArray = false;
      localDims = 1;
    }
    else if(localExpr.Opcode() == Xcode.ARRAY_ADDR){
      throw new XMPexception("Array pointer is used at coarray Syntax");
    }
    else if(localExpr.isConstant()){  // Fix me
      throw new XMPexception("Not supported a Constant Value at coarray Syntax");
    }
    else{
      throw new XMPexception("Not supported this coarray Syntax");
    }

    // Get image Dims
    XobjList imageList = (XobjList)coarrayExpr.getArg(1);
    int imageDims = coarray.getImageDim();

    // Shortcut Function
    if(isContinuousArray(coarrayExpr, exprParentBlock) &&
       isContinuousArray(localExpr, exprParentBlock) &&
       isCoarray(localExpr, exprParentBlock))
      {
        XMPcoarray remoteCoarray = coarray;
        XMPcoarray localCoarray = _globalDecl.getXMPcoarray(localName, exprParentBlock);
        boolean isRemoteCoarrayUseDevice = isUseDevice(remoteCoarray.getName(), exprParentBlock);
        boolean isLocalCoarrayUseDevice = isUseDevice(localCoarray.getName(), exprParentBlock);
        if(leftExpr.Opcode() == Xcode.CO_ARRAY_REF)
          {  // put a[:]:[1] = b[:];
            return createShortcutCoarray(imageDims, imageList, "put", remoteCoarray, localCoarray,
                                         coarrayExpr.getArg(0), localExpr, isRemoteCoarrayUseDevice, isLocalCoarrayUseDevice);
          }
        else{ // get a[:] = b[:]:[1]
          return createShortcutCoarray(imageDims, imageList, "get", localCoarray, remoteCoarray,
                                       localExpr, coarrayExpr.getArg(0), isRemoteCoarrayUseDevice, isLocalCoarrayUseDevice);
        }
      }

    // Set function _XMP_coarray_rdma_coarray_set_X()
    Ident funcId;
    funcArgs = Xcons.List();
    if(coarrayExpr.getArg(0).Opcode() == Xcode.SUB_ARRAY_REF){
      XobjList tripletList = (XobjList)(coarrayExpr.getArg(0)).getArg(1);
      funcId = _globalDecl.declExternFunc("_XMP_coarray_rdma_coarray_set_" + Integer.toString(tripletList.Nargs()));
      for(int i=0;i<tripletList.Nargs();i++){
        if(tripletList.getArg(i).isConstant() || tripletList.getArg(i).isVariable()){
          funcArgs.add(tripletList.getArg(i)); // start
          funcArgs.add(Xcons.IntConstant(1));  // length
          funcArgs.add(Xcons.IntConstant(1));  // stride
        }
        else{
          for(int j=0;j<3;j++){
            funcArgs.add(tripletList.getArg(i).getArg(j));
          }
        }
      }
    }
    else if(coarrayExpr.getArg(0).Opcode() == Xcode.ARRAY_REF){
      funcId = _globalDecl.declExternFunc("_XMP_coarray_rdma_coarray_set_1");
      XobjList startList = (XobjList)(coarrayExpr.getArg(0)).getArg(1);
      funcId = _globalDecl.declExternFunc("_XMP_coarray_rdma_coarray_set_" + Integer.toString(startList.Nargs()));
      for(int i=0;i<startList.Nargs();i++){
        funcArgs.add(startList.getArg(i));  // start
	funcArgs.add(Xcons.IntConstant(1)); // length
        funcArgs.add(Xcons.IntConstant(1)); // stride
      }
    }
    else if(coarrayExpr.getArg(0).Opcode() == Xcode.VAR){
      funcId = _globalDecl.declExternFunc("_XMP_coarray_rdma_coarray_set_1");
      funcArgs.add(Xcons.IntConstant(0)); // start
      funcArgs.add(Xcons.IntConstant(1)); // length
      funcArgs.add(Xcons.IntConstant(1)); // stride
    }
    else{
      throw new XMPexception("Not supported this coarray Syntax");
    }
    Xobject newExpr = funcId.Call(funcArgs);
    newExpr.setIsRewrittedByXmp(true);
    iter.insertStatement(newExpr);

    // Set function _XMP_coarray_rdma_array_set_X()
    funcArgs = Xcons.List();
    if(isArray){
      String arrayName = localExpr.getArg(0).getName();
      Ident varId = localExpr.findVarIdent(arrayName);
      Xtype varType = varId.Type();
      Xtype elmtType = varType.getArrayElementType();
      int varDim = varType.getNumDimensions();
      funcId = _globalDecl.declExternFunc("_XMP_coarray_rdma_array_set_" + Integer.toString(varDim));
      Integer[] sizeArray = new Integer[varDim];
      Integer[] distanceArray = new Integer[varDim];

      for(int i=0;i<varDim;i++,varType=varType.getRef()){
        int dimSize = (int)varType.getArraySize();
        if((dimSize == 0) || (dimSize == -1)){
          throw new XMPexception("array size should be declared statically");
        }
        sizeArray[i] = dimSize;
      }

      for(int i=0;i<varDim-1;i++){
	int tmp = 1;
	for(int j=i+1;j<varDim;j++){
	  tmp *= sizeArray[j];
	}
	distanceArray[i] = tmp;
      }
      distanceArray[varDim-1] = 1;

      XobjList tripletList = (XobjList)localExpr.getArg(1);
      for(int i=0;i<tripletList.Nargs();i++){
        if(tripletList.getArg(i).isVariable() || tripletList.getArg(i).isIntConstant() ){
          funcArgs.add(tripletList.getArg(i));           // start
          funcArgs.add(Xcons.IntConstant(1));            // length
          funcArgs.add(Xcons.IntConstant(1));            // stride
          funcArgs.add(Xcons.IntConstant(sizeArray[i])); // size
	  funcArgs.add(Xcons.binaryOp(Xcode.MUL_EXPR, Xcons.IntConstant(distanceArray[i]), Xcons.SizeOf(elmtType))); // distance
        }
        else{
          for(int j=0;j<3;j++){
            funcArgs.add(tripletList.getArg(i).getArg(j));
          }
          funcArgs.add(Xcons.IntConstant(sizeArray[i]));     // size
	  funcArgs.add(Xcons.binaryOp(Xcode.MUL_EXPR, Xcons.IntConstant(distanceArray[i]), Xcons.SizeOf(elmtType)));
        }
      }
    }
    else{  // !isArray
      funcId = _globalDecl.declExternFunc("_XMP_coarray_rdma_array_set_1");
      funcArgs.add(Xcons.IntConstant(0)); // start
      funcArgs.add(Xcons.IntConstant(1)); // length
      funcArgs.add(Xcons.IntConstant(1)); // stride
      funcArgs.add(Xcons.IntConstant(1)); // size
      funcArgs.add(Xcons.SizeOf(localExpr.Type()));
    }
    newExpr = funcId.Call(funcArgs);
    newExpr.setIsRewrittedByXmp(true);
    iter.insertStatement(newExpr);

    // Set function _XMP_coarray_rdma_node_image_X()
    funcId = _globalDecl.declExternFunc("_XMP_coarray_rdma_image_set_" + Integer.toString(imageDims));
    funcArgs = Xcons.List();
    for(int i=0;i<imageDims;i++){
      funcArgs.add(imageList.getArg(i));
    }
    newExpr = funcId.Call(funcArgs);
    newExpr.setIsRewrittedByXmp(true);
    iter.insertStatement(newExpr);

    // Set function _XMP_coarray_rdma_do()
    funcArgs = Xcons.List();
    if(leftExpr.Opcode() == Xcode.CO_ARRAY_REF){
      funcArgs.add(Xcons.IntConstant(XMPcoarray.PUT));
    }
    else{
      funcArgs.add(Xcons.IntConstant(XMPcoarray.GET));
    }

    boolean isLocalOnDevice = false;
    boolean isRemoteOnDevice = false;
    
    // Get Coarray Descriptor
    funcArgs.add(Xcons.SymbolRef(coarray.getDescId()));
    isRemoteOnDevice = isUseDevice(coarray.getName(), exprParentBlock);

    // Get Local Pointer Name
    if(localExpr.Opcode() == Xcode.SUB_ARRAY_REF || localExpr.Opcode() == Xcode.ARRAY_REF){
      Xobject varAddr = localExpr.getArg(0);
      String varName = varAddr.getName();
      isLocalOnDevice = isUseDevice(varName, exprParentBlock);
      XMPcoarray localArray = _globalDecl.getXMPcoarray(varName, exprParentBlock);
      if(localArray == null){
	  funcArgs.add(varAddr);
	  Xobject XMP_NULL = Xcons.Cast(Xtype.voidPtrType, Xcons.IntConstant(0));  // ((void *)0)
	  funcArgs.add(XMP_NULL);
      }
      else{
	  funcArgs.add(Xcons.SymbolRef(_globalDecl.findVarIdent(XMP.COARRAY_ADDR_PREFIX_ + varAddr.getName())));
	  funcArgs.add(Xcons.SymbolRef(localArray.getDescId()));
      }
    }
    else if(localExpr.Opcode() == Xcode.VAR || localExpr.Opcode() == Xcode.POINTER_REF){
      String varName;
      if(localExpr.Opcode() == Xcode.VAR)
        varName = localExpr.getName();
      else
        varName = localExpr.getArg(0).getName();
      
      isLocalOnDevice = isUseDevice(varName, exprParentBlock);
      XMPcoarray localVar = _globalDecl.getXMPcoarray(varName, exprParentBlock);
      if(localVar == null){
	Xobject varAddr = Xcons.AddrOf(localExpr);
	funcArgs.add(varAddr);
	Xobject XMP_NULL = Xcons.Cast(Xtype.voidPtrType, Xcons.IntConstant(0));  // ((void *)0)
	funcArgs.add(XMP_NULL);
      }
      else{
	funcArgs.add(Xcons.SymbolRef(_globalDecl.findVarIdent(XMP.COARRAY_ADDR_PREFIX_ + varName)));
	funcArgs.add(Xcons.SymbolRef(localVar.getDescId()));
      }
    }
    else if(localExpr.isConstant()){  // Fix me
      throw new XMPexception("Not supported a Constant Value at coarray Syntax");
    }
    else{
      throw new XMPexception("Not supported this coarray Syntax");
    }

    boolean isAcc = isRemoteOnDevice || isLocalOnDevice;
    if(isAcc){
      funcId = _globalDecl.declExternFunc("_XMP_coarray_rdma_do_acc");
      funcArgs.add(Xcons.IntConstant(isRemoteOnDevice? 1 : 0));
      funcArgs.add(Xcons.IntConstant(isLocalOnDevice? 1 : 0));
    }else{
      funcId = _globalDecl.declExternFunc("_XMP_coarray_rdma_do");
    }
    newExpr = funcId.Call(funcArgs);
    newExpr.setIsRewrittedByXmp(true);
    iter.insertStatement(newExpr);

    return null;
    // Memo: This function translates a coarray syntax (a[1:2:1]:[9] = b) into 4 functions.
    // This function returns null pointer except for shortcut functions. The reason of returning
    // null pointer, when XobjList is returned, an upper process is abort.
    // Therefore this function translates the coarray syntax directly.
  }

  private boolean is_stride_1(int dim, XobjList tripletList)
  {
    if(tripletList.getArg(dim).isConstant() || tripletList.getArg(dim).isVariable()){
        return true;
    } 
    else{
      Xobject stride = tripletList.getArg(dim).getArg(2);
      if(stride.isConstant()){
        if(stride.getInt() == 1){
          return true;
        }
      }
    }

    return false;
  }
  
  private boolean is_start_0(int dim, XobjList tripletList)
  {
    if(tripletList.getArg(dim).isVariable()){
      return false;
    }
    else if(tripletList.getArg(dim).isConstant())
      if(tripletList.getArg(dim).getInt() == 0){
        return true;
      }
      else{
        return false;
      }
    else{
      Xobject start = tripletList.getArg(dim).getArg(0);
      if(start.isConstant()){
        if(start.getInt() == 0){
          return true;
        }
      }
    }
    return false;
  }

  private boolean is_length_all(int dim, XobjList tripletList, Ident varId){
    if(tripletList.getArg(dim).isConstant() || tripletList.getArg(dim).isVariable()){
      return false;
    }

    Xtype varType = varId.Type();

    for(int i=0;i<dim;i++){
      varType = varType.getRef();
    }
    long dimSize = varType.getArraySize();
    Xobject length = tripletList.getArg(dim).getArg(1);

    Xobject arraySizeExpr = Xcons.List(Xcode.MINUS_EXPR, Xtype.intType,
                                       Xcons.IntConstant((int)dimSize),
                                       Xcons.IntConstant(0));

    if(arraySizeExpr.equals(length)){
      return true;
    }
    else if(length.Opcode() ==  Xcode.INT_CONSTANT){
      if(length.getInt() == (int)dimSize){
        return true;
      }
    }

    return false;
  }

  private boolean is_all_element(int dim, XobjList tripletList, Ident varId){
    if(is_start_0(dim, tripletList) && is_length_all(dim, tripletList, varId)){
      return true;
    }
    else{
      return false;
    }
  }

  private boolean is_length_1(int dim, XobjList tripletList)
  {
    if(tripletList.getArg(dim).isVariable() || tripletList.getArg(dim).isConstant()){
      return true;
    }
    else{
      Xobject length = tripletList.getArg(dim).getArg(1);
      if(length.isConstant()){
        if(length.getInt() == 1){
          return true;
        }
      }
    }

    return false;
  }

  private boolean isContinuousArray(Xobject myExpr, Block block) throws XMPexception
  {
    if(myExpr.Opcode() == Xcode.CO_ARRAY_REF)
      myExpr = myExpr.getArg(0);

    if(myExpr.Opcode() == Xcode.VAR || myExpr.Opcode() == Xcode.ARRAY_REF || myExpr.Opcode() == Xcode.POINTER_REF){
      return true;
    }
    else if(myExpr.Opcode() == Xcode.SUB_ARRAY_REF){
      XobjList tripletList = (XobjList)(myExpr).getArg(1);
      String arrayName = myExpr.getArg(0).getName();
      Ident varId = myExpr.findVarIdent(arrayName);
      Xtype varType = varId.Type();
      int varDim = varType.getNumDimensions();

      if(varDim == 1){
        if(!is_stride_1(0, tripletList)){
          return false;
        }
        else{
          return true;
        }
      }
      else if(varDim == 2){
        if(!is_stride_1(0, tripletList) || !is_stride_1(1, tripletList)){
          return false;
        }
        else if(is_all_element(1, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList)){
          return true;
        }
      }
      else if(varDim == 3){
        if(!is_stride_1(0, tripletList) || !is_stride_1(1, tripletList) || !is_stride_1(2, tripletList)){
          return false;
        }
        else if(is_all_element(1, tripletList, varId) && is_all_element(2, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_all_element(2, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_length_1(1, tripletList)){
          return true;
        }
      }
      else if(varDim == 4){
        if(!is_stride_1(0, tripletList) || !is_stride_1(1, tripletList) || !is_stride_1(2, tripletList)
           || !is_stride_1(3, tripletList)){
          return false;
        }
        else if(is_all_element(1, tripletList, varId) && is_all_element(2, tripletList, varId) 
                && is_all_element(3, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_all_element(2, tripletList, varId) && 
                is_all_element(3, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_length_1(1, tripletList) 
                && is_all_element(3, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_length_1(1, tripletList) && is_length_1(2, tripletList)){
          return true;
        }
      }
      else if(varDim == 5){
        if(!is_stride_1(0, tripletList) || !is_stride_1(1, tripletList) || !is_stride_1(2, tripletList)
           || !is_stride_1(3, tripletList) || !is_stride_1(4, tripletList)){
          return false;
        }
        else if(is_all_element(1, tripletList, varId) && is_all_element(2, tripletList, varId) 
                && is_all_element(3, tripletList, varId) && is_all_element(4, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_all_element(2, tripletList, varId) 
                && is_all_element(3, tripletList, varId) && is_all_element(4, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_length_1(1, tripletList)
                && is_all_element(3, tripletList, varId) && is_all_element(4, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_length_1(1, tripletList) && is_length_1(2, tripletList)
                && is_all_element(4, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_length_1(1, tripletList) && is_length_1(2, tripletList)
                && is_length_1(3, tripletList)){
          return true;
        }
      }
      else if(varDim == 6){
        if(!is_stride_1(0, tripletList) || !is_stride_1(1, tripletList) || !is_stride_1(2, tripletList)
           || !is_stride_1(3, tripletList) || !is_stride_1(4, tripletList) || !is_stride_1(5, tripletList)){
          return false;
        }
        else if(is_all_element(1, tripletList, varId) && is_all_element(2, tripletList, varId) 
                && is_all_element(3, tripletList, varId) && is_all_element(4, tripletList, varId) 
                && is_all_element(5, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_all_element(2, tripletList, varId) 
                && is_all_element(3, tripletList, varId) && is_all_element(4, tripletList, varId) 
                && is_all_element(5, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_length_1(1, tripletList) 
                && is_all_element(3, tripletList, varId) && is_all_element(4, tripletList, varId) 
                && is_all_element(5, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_length_1(1, tripletList) && is_length_1(2, tripletList)
                && is_all_element(4, tripletList, varId) && is_all_element(5, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_length_1(1, tripletList) && is_length_1(2, tripletList)
                && is_length_1(3, tripletList) && is_all_element(5, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_length_1(1, tripletList) && is_length_1(2, tripletList)
                && is_length_1(3, tripletList) && is_length_1(4, tripletList)){
          return true;
        }
      }
      else if(varDim == 7){
        if(!is_stride_1(0, tripletList) || !is_stride_1(1, tripletList) || !is_stride_1(2, tripletList)
           || !is_stride_1(3, tripletList) || !is_stride_1(4, tripletList) || !is_stride_1(5, tripletList)
           || !is_stride_1(6, tripletList)){
          return false;
        }
        else if(is_all_element(1, tripletList, varId) && is_all_element(2, tripletList, varId) 
                && is_all_element(3, tripletList, varId) && is_all_element(4, tripletList, varId) 
                && is_all_element(5, tripletList, varId) && is_all_element(6, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_all_element(2, tripletList, varId) 
                && is_all_element(3, tripletList, varId) && is_all_element(4, tripletList, varId) 
                && is_all_element(5, tripletList, varId) && is_all_element(6, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_length_1(1, tripletList) && 
                is_all_element(3, tripletList, varId) && is_all_element(4, tripletList, varId) 
                && is_all_element(5, tripletList, varId) && is_all_element(6, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_length_1(1, tripletList) && is_length_1(2, tripletList)
                && is_all_element(4, tripletList, varId) && is_all_element(5, tripletList, varId) 
                && is_all_element(6, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_length_1(1, tripletList) && is_length_1(2, tripletList)
                && is_length_1(3, tripletList) && is_all_element(5, tripletList, varId)
                && is_all_element(6, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_length_1(1, tripletList) && is_length_1(2, tripletList)
                && is_length_1(3, tripletList) && is_length_1(4, tripletList) 
                && is_all_element(6, tripletList, varId)){
          return true;
        }
        else if(is_length_1(0, tripletList) && is_length_1(1, tripletList) && is_length_1(2, tripletList)
                && is_length_1(3, tripletList) && is_length_1(4, tripletList) && is_length_1(5, tripletList)){
          return true;
        }
      }
    }
    else{
      throw new XMPexception("Not supported this coarray Syntax");
    }

    return false;
  }

  private boolean isCoarray(Xobject myExpr, Block block){
    if(myExpr.Opcode() == Xcode.ARRAY_REF || myExpr.Opcode() == Xcode.SUB_ARRAY_REF ||
       myExpr.Opcode() == Xcode.ADDR_OF){
      myExpr = myExpr.getArg(0);
    }

    if(myExpr.Opcode() == Xcode.POINTER_REF || myExpr.Opcode() == Xcode.ARRAY_REF)
      myExpr = myExpr.getArg(0);

    XMPcoarray coarray = _globalDecl.getXMPcoarray(myExpr.getSym(), block);
    
    if(coarray == null)
      return false;
    else
      return true;
  }

  private void rewriteStmts(FunctionBlock funcBlock, XMPsymbolTable localXMPsymbolTable){

    BasicBlockIterator iter = new BasicBlockIterator(funcBlock);
    for (iter.init(); !iter.end(); iter.next()) {
      BasicBlock bb = iter.getBasicBlock();
      for (Statement s = bb.getHead(); s != null; s = s.getNext()){
	Xobject x = s.getExpr();
	try {
	  if (x.Opcode() == Xcode.ASSIGN_EXPR && x.getArg(0).Opcode() == Xcode.SUB_ARRAY_REF &&
	      x.getArg(1).Opcode() != Xcode.CO_ARRAY_REF){
	    Block b = rewriteSubarrayToLoop(x, bb.getParent());
	    s.insertBlock(b);
	    s.remove();
	  }
	} catch (XMPexception e){
	  XMP.error(x.getLineNo(), e.getMessage());
	}
      }
    }

  }

  private Block rewriteSubarrayToLoop(Xobject assignStmt, Block block) throws XMPexception {

    // NOTE: allmost all of the following code comes from XMPtranslatePragma.convertArrrayToLoop

    Xobject left = assignStmt.left();

    List<Ident> varList = new ArrayList<Ident>(XMP.MAX_DIM);
    List<Xobject> lbList = new ArrayList<Xobject>(XMP.MAX_DIM);
    List<Xobject> lenList = new ArrayList<Xobject>(XMP.MAX_DIM);
    List<Xobject> stList = new ArrayList<Xobject>(XMP.MAX_DIM);

    BlockList body0 = Bcons.emptyBody();

    //
    // convert LHS
    //

    assert left.Opcode() == Xcode.SUB_ARRAY_REF;

    String arrayName = left.getArg(0).getSym();

    Xtype arrayType = null;
    Ident arrayId = block.findVarIdent(arrayName);
    if (arrayId != null){
      arrayType = arrayId.Type();
    }
	
    if (arrayType == null) throw new XMPexception("array should be declared statically");

    Xtype elemType = arrayType.getArrayElementType();
    int n = arrayType.getNumDimensions();

    XobjList subscripts = (XobjList)left.getArg(1);

    for (int i = 0; i < n; i++, arrayType = arrayType.getRef()){

      long dimSize = arrayType.getArraySize();
      Xobject sizeExpr;
      if (dimSize == 0 || arrayType.getKind() == Xtype.POINTER){
	throw new XMPexception("array size should be declared statically");
      }
      else if (dimSize == -1){
        sizeExpr = arrayType.getArraySizeExpr();
      }
      else {
	sizeExpr = Xcons.LongLongConstant(0, dimSize);
      }

      Xobject sub = subscripts.getArg(i);

      Ident var;
      Xobject lb, len, st;

      if (sub.Opcode() != Xcode.LIST) continue;

      var = body0.declLocalIdent("_XMP_loop_i" + Integer.toString(i), Xtype.intType);
      varList.add(var);

      lb = ((XobjList)sub).getArg(0);
      if (lb == null) lb = Xcons.IntConstant(0);
      len = ((XobjList)sub).getArg(1);
      if (len == null) len = sizeExpr;
      st = ((XobjList)sub).getArg(2);
      if (st == null) st = Xcons.IntConstant(1);

      lbList.add(lb);
      lenList.add(len);
      stList.add(st);

      Xobject expr;
      expr = Xcons.binaryOp(Xcode.MUL_EXPR, var.Ref(), st);
      expr = Xcons.binaryOp(Xcode.PLUS_EXPR, expr, lb);

      subscripts.setArg(i, expr);

    }

    Xobject new_left = Xcons.arrayRef(elemType, left.getArg(0), subscripts);

    //
    // convert RHS
    //

    // NOTE: Since the top level object cannot be replaced, the following conversion is applied to
    //       the whole assignment.
    XobjectIterator j = new topdownXobjectIterator(assignStmt);
    for (j.init(); !j.end(); j.next()) {
      Xobject x = j.getXobject();

      if (x.Opcode() != Xcode.SUB_ARRAY_REF) continue;

      int k = 0;

      String arrayName1 = x.getArg(0).getSym();

      Xtype arrayType1 = null;
      Ident arrayId1 = block.findVarIdent(arrayName1);
      if (arrayId1 != null){
	arrayType1 = arrayId1.Type();
      }
	
      if (arrayType1 == null) throw new XMPexception("array should be declared statically");

      Xtype elemType1 = arrayType1.getArrayElementType();
      int m = arrayType1.getNumDimensions();

      XobjList subscripts1 = (XobjList)x.getArg(1);

      for (int i = 0; i < m; i++, arrayType1 = arrayType1.getRef()){

	Xobject sub = subscripts1.getArg(i);

	Xobject lb, st;

	if (sub.Opcode() != Xcode.LIST) continue;

	lb = ((XobjList)sub).getArg(0);
	if (lb == null) lb = Xcons.IntConstant(0);
	st = ((XobjList)sub).getArg(2);
	if (st == null) st = Xcons.IntConstant(1);

	Xobject expr;
	Ident loopVar = varList.get(k);
	if (loopVar == null) XMP.fatal("array on rhs does not conform to that on lhs.");
	expr = Xcons.binaryOp(Xcode.MUL_EXPR, loopVar.Ref(), st);
	expr = Xcons.binaryOp(Xcode.PLUS_EXPR, expr, lb);

	subscripts1.setArg(i, expr);
	k++;
      }

      Xobject new_x = Xcons.arrayRef(elemType1, x.getArg(0), subscripts1);
      j.setXobject(new_x);

    }

    //
    // construct loop
    //

    //BlockList loop = body0;
    BlockList loop = null;

    BlockList body = Bcons.emptyBody();
    body.add(Xcons.Set(new_left, assignStmt.right()));

    for (int i = varList.size() - 1; i >= 0; i--){
      loop = Bcons.emptyBody();
      loop.add(Bcons.FORall(varList.get(i).Ref(), Xcons.IntConstant(0), lenList.get(i), Xcons.IntConstant(1),
			    Xcode.LOG_LT_EXPR, body));
      body = loop;
    }

    loop.setIdentList(body0.getIdentList());
    loop.setDecls(body0.getDecls());

    //return body0.toXobject();
    return Bcons.COMPOUND(loop);

  }
  
  private Xobject rewriteExpr(Xobject expr, Block block) throws XMPexception {
    if (expr == null) {
      return null;
    }
    switch (expr.Opcode()) {
    case ARRAY_REF:
      return rewriteArrayRef(expr, block);
    case VAR:
      return rewriteVarRef(expr, block, true);
    case ARRAY_ADDR:
      return rewriteVarRef(expr, block, false);
    case POINTER_REF:
      return rewritePointerRef(expr, block);
    default:
      {
	topdownXobjectIterator iter = new topdownXobjectIterator(expr);
	for (iter.init(); !iter.end(); iter.next()) {
	  Xobject myExpr = iter.getXobject();
	  if (myExpr == null) {
	    continue;
	  } else if (myExpr.isRewrittedByXmp()) {
	    continue;
	  }
	  switch (myExpr.Opcode()) {
	  case ARRAY_ADDR:
	    iter.setXobject(rewriteArrayAddr(myExpr, block));
	    break;
	  case ARRAY_REF:
	    iter.setXobject(rewriteArrayRef(myExpr, block));
	    break;
	  case SUB_ARRAY_REF:
	    break;
	  case XMP_DESC_OF:
	    iter.setXobject(rewriteXmpDescOf(myExpr, block));
	    break;
	  case VAR:
	    iter.setXobject(rewriteVarRef(myExpr, block, true));
	    break;
	  case POINTER_REF:
	    iter.setXobject(rewritePointerRef(myExpr, block));
	    break;
	  case FUNCTION_CALL:
	    Xobject f = myExpr.getArg(0);
	    if (f.Opcode() == Xcode.FUNC_ADDR && f.getString().equals("xmp_malloc")){
	      Xobject p = iter.getParent();
	      if (p.Opcode() == Xcode.CAST_EXPR){
		Xtype pt = p.Type();
		if (!pt.isPointer()) break;
		Xtype t = pt.getRef();
		if (t.isArray()) t = t.getArrayElementType();
	  	p.setType(Xtype.Pointer(t));
	      }
	    }
            else if (f.Opcode() == Xcode.FUNC_ADDR && f.getString().equals("xmp_atomic_define")){
              iter.setXobject(rewriteXmpAtomicDefine(myExpr, block));
            }
            else if (f.Opcode() == Xcode.FUNC_ADDR && f.getString().equals("xmp_atomic_ref")){
              iter.setXobject(rewriteXmpAtomicRef(myExpr, block));
            }
	  default:
	  }
	}
	return expr;
      }
    }
  }

  private boolean isOneElement(Xobject myExpr) throws XMPexception {
    if(myExpr.Opcode() == Xcode.CO_ARRAY_REF)
      myExpr = myExpr.getArg(0);

    Xcode op = myExpr.Opcode();
    if(op == Xcode.VAR || op == Xcode.ARRAY_REF || op == Xcode.ADDR_OF || op == Xcode.INT_CONSTANT)
      return true;
    else if(op == Xcode.SUB_ARRAY_REF)
      return false;
    else{
      throw new XMPexception("Unexpected value.");
    }
  }

  private Xobject rewriteXmpAtomicRef(Xobject myExpr, Block block) throws XMPexception {
    Xobject localExpr   = myExpr.getArg(1).getArg(0);
    Xobject coarrayExpr = myExpr.getArg(1).getArg(1);

    String coarrayName;
    if(coarrayExpr.Opcode() == Xcode.CO_ARRAY_REF){  // xmp_atomic_ref(&value, atom[1]:[2]);
      coarrayName = XMPutil.getXobjSymbolName(coarrayExpr.getArg(0));
    }
    else if(coarrayExpr.Opcode() == Xcode.VAR){      // xmp_atomic_ref(&value, atom);
      coarrayName = coarrayExpr.getName();
    }
    else if(Xcode.ARRAY_REF == Xcode.ARRAY_REF){     // xmp_atomic_ref(&value, atom[1]);
      coarrayName = coarrayExpr.getArg(0).getName();
    }
    else{
      throw new XMPexception("UNKNOWN DATA TYPE of the 1st argument in xmp_atomic_ref().");
    }
    
    XMPcoarray coarray = _globalDecl.getXMPcoarray(coarrayName, block);
    XobjList funcArgs  = Xcons.List(coarray.getDescId());

    if(coarray == null)
      throw new XMPexception("cannot find coarray '" + coarrayName + "'");

    // Two arguments in xmp_atomic_ref() must be int-type
    if(! localExpr.Type().isPointer())
      throw new XMPexception("The first argument of atomic_ref() must be an int-type pointer.");

    if(coarray.getElmtType() != Xtype.intType)
      throw new XMPexception("The second argument of atomic_ref() must be an int-type.");
    
    // i.e. Only 1 element can be used.
    if(!isOneElement(coarrayExpr) || !isOneElement(localExpr))
      throw new XMPexception("An argument of atomic_ref() must be a scalar coarray or coindexed object.");
      
    // Add offset
    funcArgs.add(getCoarrayOffset(coarrayExpr, coarray));

    // Get Coarray Dims
    int imageDims = (coarrayExpr.Opcode() == Xcode.CO_ARRAY_REF)? coarray.getImageDim() : 0;
    String funcName = "_XMP_atomic_ref_" + imageDims;
    Ident funcId    = _globalDecl.declExternFunc(funcName);
    for(int i=0;i<imageDims;i++)
      funcArgs.add(coarrayExpr.getArg(1).getArg(i));

    // When localExpr is coarray, descriptor of the coarray is added to argument
    if(isCoarray(localExpr, block)){
      localExpr = localExpr.getArg(0);
      if(localExpr.Opcode() == Xcode.VAR){
        funcArgs.add(rewriteVarRef(localExpr, block, false));
      }
      else{  //  else if(localExpr.Opcode() == Xcode.ARRAY_REF){
        funcArgs.add(Xcons.AddrOf(rewriteArrayRef(localExpr, block)));
      }
      String localName = (localExpr.Opcode() == Xcode.VAR)? localExpr.getName() : localExpr.getArg(0).getName();
      funcArgs.add(_globalDecl.getXMPcoarray(localName, block).getDescId());
      funcArgs.add(getCoarrayOffset(localExpr, _globalDecl.getXMPcoarray(localName, block)));
    }
    else{
      funcArgs.add(localExpr);
      funcArgs.add(Xcons.Cast(Xtype.voidPtrType, Xcons.IntConstant(0))); // NULL
      funcArgs.add(Xcons.IntConstant(0));                                // offset (This value is not used)
    }

    // The variable must be an int-type
    funcArgs.add(Xcons.SizeOf(Xtype.intType));
    
    return funcId.Call(funcArgs);
  }
  
  private Xobject rewriteXmpAtomicDefine(Xobject myExpr, Block block) throws XMPexception {
    Xobject coarrayExpr = myExpr.getArg(1).getArg(0);
    Xobject localExpr   = myExpr.getArg(1).getArg(1);
    
    String coarrayName;
    if(coarrayExpr.Opcode() == Xcode.CO_ARRAY_REF){  // xmp_atomic_define(atom[1]:[2]), value);
      coarrayName = XMPutil.getXobjSymbolName(coarrayExpr.getArg(0));
    }
    else if(coarrayExpr.Opcode() == Xcode.VAR){      // xmp_atomic_define(atom, value);
      coarrayName = coarrayExpr.getName();
    }
    else if(Xcode.ARRAY_REF == Xcode.ARRAY_REF){     // xmp_atomic_define(atom[1], value);
      coarrayName = coarrayExpr.getArg(0).getName();
    }
    else{
      throw new XMPexception("UNKNOWN DATA TYPE of the 1st argument in xmp_atomic_define().");
    }

    XMPcoarray coarray = _globalDecl.getXMPcoarray(coarrayName, block);
    XobjList funcArgs  = Xcons.List(coarray.getDescId());

    if(coarray == null)
      throw new XMPexception("cannot find coarray '" + coarrayName + "'");

    // Two arguments in xmp_atomic_define() must be int-type
    if(coarray.getElmtType() != Xtype.intType || localExpr.Type() != Xtype.intType)
      throw new XMPexception("An argument of atomic_define() must be an int-type.");

    // i.e. Only 1 element can be used.
    if(!isOneElement(coarrayExpr) || !isOneElement(localExpr))
      throw new XMPexception("An argument of atomic_define() must be a scalar coarray or coindexed object.");

    // Add offset
    funcArgs.add(getCoarrayOffset(coarrayExpr, coarray));

    // Get Coarray Dims
    int imageDims = (coarrayExpr.Opcode() == Xcode.CO_ARRAY_REF)? coarray.getImageDim() : 0;
    String funcName = "_XMP_atomic_define_" + imageDims;
    Ident funcId    = _globalDecl.declExternFunc(funcName);
    for(int i=0;i<imageDims;i++)
      funcArgs.add(coarrayExpr.getArg(1).getArg(i));

    if(localExpr.Opcode() == Xcode.INT_CONSTANT){
      funcArgs.add(localExpr);
      funcArgs.add(Xcons.Cast(Xtype.voidPtrType, Xcons.IntConstant(0))); // NULL
      funcArgs.add(Xcons.IntConstant(0));                                // offset (This value is not used)
    }
    else{
      // When localExpr is coarray, descriptor of the coarray is added to argument
      if(isCoarray(localExpr, block)){
        if(localExpr.Opcode() == Xcode.VAR){
          funcArgs.add(rewriteVarRef(localExpr, block, true));
        }
        else{ //  else if(localExpr.Opcode() == Xcode.ARRAY_REF){
          funcArgs.add(rewriteArrayRef(localExpr, block));
        }
        String localName = (localExpr.Opcode() == Xcode.VAR)? localExpr.getName() : localExpr.getArg(0).getName();
        funcArgs.add(_globalDecl.getXMPcoarray(localName, block).getDescId());
	funcArgs.add(getCoarrayOffset(localExpr, _globalDecl.getXMPcoarray(localName, block)));
      }
      else{
        funcArgs.add(localExpr);
        funcArgs.add(Xcons.Cast(Xtype.voidPtrType, Xcons.IntConstant(0))); // NULL
	funcArgs.add(Xcons.IntConstant(0));                                // offset (This value is not used)
      }
    }
    
    // The variable must be an int-type
    funcArgs.add(Xcons.SizeOf(Xtype.intType));
    
    return funcId.Call(funcArgs);
  }
  
  private Xobject rewriteXmpDescOf(Xobject myExpr, Block block) throws XMPexception {
    String entityName = myExpr.getArg(0).getName();
    XMPobject entity = _globalDecl.getXMPobject(entityName, block);
    Xobject e = null;

    if(entity != null){
      if(entity.getKind() == XMPobject.TEMPLATE || entity.getKind() == XMPobject.NODES){
	Ident XmpDescOfFuncId = _globalDecl.declExternFunc("_XMP_desc_of", myExpr.Type());
	e = XmpDescOfFuncId.Call(Xcons.List(entity.getDescId().Ref()));
      } 
      else{
	throw new XMPexception("Bad entity name for xmp_desc_of()");
      }
    }
    else{ // When myExpr is a distributed array name.
      String arrayName = myExpr.getArg(0).getSym();
      XMPalignedArray alignedArray =  _globalDecl.getXMPalignedArray(arrayName, block);
      if (alignedArray == null)
	throw new XMPexception(arrayName + " is not aligned global array or tempalte descriptor.");

      Ident XmpDescOfFuncId =  _globalDecl.declExternFunc("_XMP_desc_of", myExpr.Type());
      e = XmpDescOfFuncId.Call(Xcons.List(alignedArray.getDescId().Ref())); 
    }

    return e;
  }

  private Xobject rewriteArrayAddr(Xobject arrayAddr, Block block) throws XMPexception {
    XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(arrayAddr.getSym(), block);
    XMPcoarray coarray = _globalDecl.getXMPcoarray(arrayAddr.getSym(), block);

    if (alignedArray == null && coarray == null) {
      return arrayAddr;
    }

    else if(alignedArray != null && coarray == null){ // only alignedArray
      if (alignedArray.checkRealloc() || (alignedArray.isLocal() && !alignedArray.isParameter()) ||
	  alignedArray.isParameter()){
	Xobject newExpr = alignedArray.getAddrId().Ref();
	newExpr.setIsRewrittedByXmp(true);
	return newExpr;
      }
      else {
      	return arrayAddr;
      }
    } else if(alignedArray == null && coarray != null){  // only coarray
      return rewriteVarRef(arrayAddr, block, false);
    } else{ // no execute
      return arrayAddr;
    }
  }
  
  private Xobject rewriteVarRef(Xobject myExpr, Block block, boolean isVar) throws XMPexception {
    String varName     = myExpr.getSym();
    XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(varName, block);
    XMPcoarray coarray = _globalDecl.getXMPcoarray(varName, block);

    if (alignedArray != null && coarray == null){
      return alignedArray.getAddrId().Ref();
    }
    else if (alignedArray == null && coarray != null){
      Ident coarrayIdent = _globalDecl.getXMPcoarray(varName).getVarId();
      Ident localIdent = XMPlocalDecl.findLocalIdent(block, varName);
      if(coarrayIdent != localIdent){
        // e.g.) When an coarray is declared at global region and 
        //       the same name variable is decleard at local region.
        //
        // int a:[*]
        // void hoge(){
        //   int a;
        //   printf("%d\n", a);  <- "a" should not be changed.
        // }
        return myExpr;
      }
      else{
        Xobject newExpr = _globalDecl.findVarIdent(XMP.COARRAY_ADDR_PREFIX_ + varName).getValue();
        newExpr = Xcons.PointerRef(newExpr);
        if(isVar) // When coarray is NOT pointer,
          newExpr = Xcons.PointerRef(newExpr);
        return newExpr;
      }
    } else{
      return myExpr;
    }
  }
  
  private Xobject rewriteArrayRef(Xobject myExpr, Block block) throws XMPexception {
    Xobject arrayAddr = myExpr.getArg(0);
    String arrayName = arrayAddr.getSym();
    XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(arrayName, block);
    XMPcoarray      coarray      = _globalDecl.getXMPcoarray(arrayName, block);

    if (alignedArray == null && coarray == null) {
      return myExpr;
    } 
    else if(alignedArray != null && coarray == null){  // only alignedArray
      Xobject newExpr = null;
      XobjList arrayRefList = normArrayRefList((XobjList)myExpr.getArg(1), alignedArray);

      if (alignedArray.checkRealloc() || (alignedArray.isLocal() && !alignedArray.isParameter()) ||
	  alignedArray.isParameter()){
	newExpr = rewriteAlignedArrayExpr(arrayRefList, alignedArray);
      } 
      else {
        newExpr = Xcons.arrayRef(myExpr.Type(), arrayAddr, arrayRefList);
      }

      newExpr.setIsRewrittedByXmp(true);
      return newExpr;
    } 
    else if(alignedArray == null && coarray != null){  // only coarray
      Xobject newExpr = translateCoarrayRef(myExpr.getArg(1), coarray);
      if(isAddrCoarray((XobjList)myExpr.getArg(1), coarray) == true){
	return Xcons.AddrOf(newExpr);
      }	else{
	return newExpr;
      }
    } 
    else{  // this statemant must not be executed
      return myExpr;
    }
  }
  
  private Xobject rewritePointerRef(Xobject myExpr, Block block) throws XMPexception
  {
    Xobject addr_expr = myExpr.getArg(0);
    if (addr_expr.Opcode() == Xcode.PLUS_EXPR){

      Xobject pointer = addr_expr.getArg(0);
      Xobject offset = addr_expr.getArg(1);

      if (pointer.Opcode() == Xcode.VAR){
	XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(pointer.getSym(), block);
	XMPcoarray      coarray      = _globalDecl.getXMPcoarray(pointer.getSym(), block);

	if (alignedArray != null && coarray == null){
	  //if (!alignedArray.isParameter())
	    addr_expr.setArg(0, alignedArray.getAddrId().Ref());
	  // NOTE: an aligned pointer is assumed to be a one-dimensional array.
	  addr_expr.setArg(1, getCalcIndexFuncRef(alignedArray, 0, offset)); 
	}
	else if(alignedArray == null && coarray != null){
	  ;
	}
      }
    }
    return myExpr;
  }

  private boolean isAddrCoarray(XobjList myExpr, XMPcoarray coarray){
    if(myExpr.getArgOrNull(coarray.getVarDim()-1) == null){
      return true;
    }
    else{
      return false;
    }
  }

  private Xobject getCoarrayOffset(Xobject myExpr, XMPcoarray coarray){
    // "a[N][M][K]" is defined as a coarray.
    // If a[i][j][k] is referred, this function returns "(i * M * K) + (j * K) + (k)"

    if(myExpr.Opcode() == Xcode.VAR){
      return Xcons.Int(Xcode.INT_CONSTANT, 0);
    }
    else if(myExpr.Opcode() == Xcode.ARRAY_REF){
      myExpr = myExpr.getArg(1);
    }
    else if(myExpr.Opcode() == Xcode.CO_ARRAY_REF){
      if(myExpr.getArg(0).Opcode() == Xcode.VAR)
        return Xcons.Int(Xcode.INT_CONSTANT, 0);
      else  // (myExpr.Opcode() == Xcode.ARRAY_REF)
        myExpr = myExpr.getArg(0).getArg(1);
    }

    Xobject newExpr = null;
    for(int i=0; i<coarray.getVarDim(); i++){
      Xobject tmp = null;
      for(int j=coarray.getVarDim()-1; j>i; j--){
	  int size = (int)coarray.getSizeAt(j);
        if(tmp == null){
          tmp = Xcons.Int(Xcode.INT_CONSTANT, size);
        } else{
          tmp = Xcons.binaryOp(Xcode.MUL_EXPR, Xcons.Int(Xcode.INT_CONSTANT, size), tmp);
        }
      } // end j

      /* Code may be optimized by native compiler when variable(e,g. i, j) is multipled finally. */
      if(myExpr.getArgOrNull(i) == null) break;
      Xobject var = myExpr.getArg(i);

      if(tmp != null){
        var = Xcons.binaryOp(Xcode.MUL_EXPR, tmp, var);
      }

      if(newExpr == null){
        newExpr = var.copy();
      }
      else{
        newExpr = Xcons.binaryOp(Xcode.PLUS_EXPR, newExpr, var);
      }
    }
    return newExpr;
  }

  private Xobject translateCoarrayRef(Xobject myExpr, XMPcoarray coarray){
    // "a[N][M][K]" is defined as a coarray.
    // When "a[i][j][k] = x;" is defined,
    // this function returns "*(_XMP_COARRAY_ADDR_a + (i * M * K) + (j * K) + (k)) = x;".
    Xobject newExpr = getCoarrayOffset(myExpr, coarray);
    
    int offset = -999;  // dummy
    if(newExpr.Opcode() == Xcode.INT_CONSTANT){
      offset = newExpr.getInt();
    }
    
    if(offset == 0){
      Ident tmpExpr = _globalDecl.findVarIdent(XMP.COARRAY_ADDR_PREFIX_ + coarray.getName());
      newExpr = Xcons.PointerRef(tmpExpr.Ref());
    }
    else{
      newExpr = Xcons.binaryOp(Xcode.PLUS_EXPR,
			       _globalDecl.findVarIdent(XMP.COARRAY_ADDR_PREFIX_ + coarray.getName()).Ref(),
			       newExpr);
      newExpr = Xcons.PointerRef(newExpr);
    }
    
    return newExpr;
  }
  
  public static XobjList normArrayRefList(XobjList refExprList, XMPalignedArray alignedArray)
  {
    if (refExprList == null) {
      return null;
    } else {
      XobjList newRefExprList = Xcons.List();
      
      int arrayIndex = 0;
      for (Xobject x : refExprList) {
        Xobject normExpr = alignedArray.getAlignNormExprAt(arrayIndex);
        if (normExpr != null) {
          newRefExprList.add(Xcons.binaryOp(Xcode.PLUS_EXPR, x, normExpr));
        } else {
          newRefExprList.add(x);
        }
        arrayIndex++;
      }

      return newRefExprList;
    }
  }

  private Xobject rewriteAlignedArrayExpr(XobjList refExprList,
                                          XMPalignedArray alignedArray) throws XMPexception {
    int arrayDimCount = 0;
    XobjList args = Xcons.List(alignedArray.getAddrId().Ref());
    if (refExprList != null) {
      for (Xobject x : refExprList) {
	args.add(getCalcIndexFuncRef(alignedArray, arrayDimCount, x));
        arrayDimCount++;
      }
    }

    return createRewriteAlignedArrayFunc(alignedArray, arrayDimCount, args);
  }

  public static Xobject createRewriteAlignedArrayFunc(XMPalignedArray alignedArray, int arrayDimCount,
                                                      XobjList getAddrFuncArgs) throws XMPexception {
    int arrayDim = alignedArray.getDim();
    Ident getAddrFuncId = null;

    if (arrayDim < arrayDimCount) {
      throw new XMPexception("wrong array ref");
    } else if (arrayDim == arrayDimCount) {
      getAddrFuncId = XMP.getMacroId("_XMP_M_GET_ADDR_E_" + arrayDim, Xtype.Pointer(alignedArray.getType()));
      for (int i = 0; i < arrayDim - 1; i++)
        getAddrFuncArgs.add(alignedArray.getAccIdAt(i).Ref());
    } else {
      getAddrFuncId = XMP.getMacroId("_XMP_M_GET_ADDR_" + arrayDimCount, Xtype.Pointer(alignedArray.getType()));
      for (int i = 0; i < arrayDimCount; i++)
        getAddrFuncArgs.add(alignedArray.getAccIdAt(i).Ref());
    }

    Xobject retObj = getAddrFuncId.Call(getAddrFuncArgs);
    if (arrayDim == arrayDimCount) {
      return Xcons.PointerRef(retObj);
    } else {
      return retObj;
    }
  }

  private Xobject getCalcIndexFuncRef(XMPalignedArray alignedArray, int index, Xobject indexRef) throws XMPexception {
    switch (alignedArray.getAlignMannerAt(index)) {
      case XMPalignedArray.NOT_ALIGNED:
      case XMPalignedArray.DUPLICATION:
        return indexRef;
      case XMPalignedArray.BLOCK:
        if (alignedArray.hasShadow()) {
          XMPshadow shadow = alignedArray.getShadowAt(index);
          switch (shadow.getType()) {
            case XMPshadow.SHADOW_NONE:
            case XMPshadow.SHADOW_NORMAL:
              {
                XobjList args = Xcons.List(indexRef, alignedArray.getGtolTemp0IdAt(index).Ref());
                return XMP.getMacroId("_XMP_M_CALC_INDEX_BLOCK").Call(args);
              }
            case XMPshadow.SHADOW_FULL:
              return indexRef;
            default:
              throw new XMPexception("unknown shadow type");
          }
        }
        else {
          XobjList args = Xcons.List(indexRef,
                                     alignedArray.getGtolTemp0IdAt(index).Ref());
          return XMP.getMacroId("_XMP_M_CALC_INDEX_BLOCK").Call(args);
        }
      case XMPalignedArray.CYCLIC:
        if (alignedArray.hasShadow()) {
          XMPshadow shadow = alignedArray.getShadowAt(index);
          switch (shadow.getType()) {
            case XMPshadow.SHADOW_NONE:
              {
                XobjList args = Xcons.List(indexRef,
                                           alignedArray.getGtolTemp0IdAt(index).Ref());
                return XMP.getMacroId("_XMP_M_CALC_INDEX_CYCLIC").Call(args);
              }
            case XMPshadow.SHADOW_FULL:
              return indexRef;
            case XMPshadow.SHADOW_NORMAL:
              throw new XMPexception("only block distribution allows shadow");
            default:
              throw new XMPexception("unknown shadow type");
          }
        }
        else {
          XobjList args = Xcons.List(indexRef, alignedArray.getGtolTemp0IdAt(index).Ref());
          return XMP.getMacroId("_XMP_M_CALC_INDEX_CYCLIC").Call(args);
        }
      case XMPalignedArray.BLOCK_CYCLIC:
        {
          XMPtemplate t = alignedArray.getAlignTemplate();
          int ti = alignedArray.getAlignSubscriptIndexAt(index).intValue();
          XMPnodes n = t.getOntoNodes();
          int ni = t.getOntoNodesIndexAt(ti).getInt();

          if (alignedArray.hasShadow()) {
            XMPshadow shadow = alignedArray.getShadowAt(index);
            switch (shadow.getType()) {
              case XMPshadow.SHADOW_NONE:
                {
                  XobjList args = Xcons.List(indexRef, n.getSizeAt(ni), t.getWidthAt(ti));
                  return XMP.getMacroId("_XMP_M_CALC_INDEX_BLOCK_CYCLIC").Call(args);
                }
              case XMPshadow.SHADOW_FULL:
                return indexRef;
              case XMPshadow.SHADOW_NORMAL:
                throw new XMPexception("only block distribution allows shadow");
              default:
                throw new XMPexception("unknown shadow type");
            }
          }
          else {
            XobjList args = Xcons.List(indexRef, n.getSizeAt(ni), t.getWidthAt(ti));
            return XMP.getMacroId("_XMP_M_CALC_INDEX_BLOCK_CYCLIC").Call(args);
          }
        }
      case XMPalignedArray.GBLOCK:
        // XobjList args = Xcons.List(alignedArray.getDescId().Ref(), Xcons.IntConstant(index), indexRef);
        // Ident f = _globalDecl.declExternFunc("_XMP_lidx_GBLOCK");
        // return f.Call(args);

	if (alignedArray.hasShadow()) {
	  XMPshadow shadow = alignedArray.getShadowAt(index);
	  switch (shadow.getType()) {
	  case XMPshadow.SHADOW_NONE:
	  case XMPshadow.SHADOW_NORMAL:
	    {
	      XobjList args = Xcons.List(indexRef, alignedArray.getGtolTemp0IdAt(index).Ref());
	      return XMP.getMacroId("_XMP_M_CALC_INDEX_GBLOCK").Call(args);
	    }
	  case XMPshadow.SHADOW_FULL:
	    return indexRef;
	  default:
	    throw new XMPexception("unknown shadow type");
	  }
        }        
	else {
	  XobjList args = Xcons.List(indexRef,
				     alignedArray.getGtolTemp0IdAt(index).Ref());
	  return XMP.getMacroId("_XMP_M_CALC_INDEX_GBLOCK").Call(args);
	}
      default:
        throw new XMPexception("unknown align manner for array '" + alignedArray.getName()  + "'");
    }
  }

  public static void rewriteArrayRefInLoop(Xobject expr, XMPglobalDecl globalDecl, Block block) throws XMPexception {

    if (expr == null) return;

    topdownXobjectIterator iter = new topdownXobjectIterator(expr);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject myExpr = iter.getXobject();
      if (myExpr == null) {
        continue;
      } else if (myExpr.isRewrittedByXmp()) {
        continue;
      }
      switch (myExpr.Opcode()) {
        case ARRAY_REF:
          {
            Xobject arrayAddr = myExpr.getArg(0);
            String arrayName = arrayAddr.getSym();
	    XMPalignedArray alignedArray = globalDecl.getXMPalignedArray(arrayName, block);

            if (alignedArray != null) {
              Xobject newExpr = null;
              XobjList arrayRefList = XMPrewriteExpr.normArrayRefList((XobjList)myExpr.getArg(1), alignedArray);
              if (alignedArray.checkRealloc() || (alignedArray.isLocal() && !alignedArray.isParameter()) ||
		  alignedArray.isParameter()){
                newExpr = XMPrewriteExpr.rewriteAlignedArrayExprInLoop(arrayRefList, alignedArray);
              } else {
                newExpr = Xcons.arrayRef(myExpr.Type(), arrayAddr, arrayRefList);
              }
              newExpr.setIsRewrittedByXmp(true);
              iter.setXobject(newExpr);
            }
          } break;
        case POINTER_REF:
	  {
	    Xobject addr_expr = myExpr.getArg(0);
	    if (addr_expr.Opcode() == Xcode.PLUS_EXPR){

	      Xobject pointer = addr_expr.getArg(0);
	      Xobject offset = addr_expr.getArg(1);

	      if (pointer.Opcode() == Xcode.VAR){
		XMPalignedArray alignedArray = globalDecl.getXMPalignedArray(pointer.getSym(), block);
		if (alignedArray != null){
		  XobjList arrayRefList = XMPrewriteExpr.normArrayRefList(Xcons.List(offset), alignedArray);
		  if (alignedArray.checkRealloc() || (alignedArray.isLocal() && !alignedArray.isParameter()) ||
		      alignedArray.isParameter()){
		    Xobject newExpr = XMPrewriteExpr.rewriteAlignedArrayExprInLoop(arrayRefList, alignedArray);
		    newExpr.setIsRewrittedByXmp(true);
		    iter.setXobject(newExpr);
		  }
		  else {
		    addr_expr.setArg(1, arrayRefList.getArg(0));
		  }

		}
	      }
	    }
	    break;
	  }

        default:
      }
    }
  }

  private static Xobject rewriteAlignedArrayExprInLoop(XobjList refExprList,
                                                       XMPalignedArray alignedArray) throws XMPexception {
    int arrayDimCount = 0;
    XobjList args;

    args = Xcons.List(alignedArray.getAddrId().Ref());

    if (refExprList != null) {
      for (Xobject x : refExprList) {
        args.add(x);
        arrayDimCount++;
      }
    }

    return XMPrewriteExpr.createRewriteAlignedArrayFunc(alignedArray, arrayDimCount, args);
  }

  public static void rewriteLoopIndexInLoop(Xobject expr, String loopIndexName, XMPtemplate templateObj,
                                            int templateIndex, XMPglobalDecl globalDecl, Block block) throws XMPexception {
    if (expr == null) return;
    topdownXobjectIterator iter = new topdownXobjectIterator(expr);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject myExpr = iter.getXobject();
      if (myExpr == null) {
        continue;
      }
      else if (myExpr.isRewrittedByXmp()) {
        continue;
      }
      
      switch (myExpr.Opcode()) {
      case VAR:
	{
	  if (loopIndexName.equals(myExpr.getSym())) {
	    iter.setXobject(calcLtoG(templateObj, templateIndex, myExpr));
	  }
	}
        break;
      case ARRAY_REF:
	{
	  XMPalignedArray alignedArray = globalDecl.getXMPalignedArray(myExpr.getArg(0).getSym(), block);
	  if (alignedArray == null) {
	    rewriteLoopIndexVar(templateObj, templateIndex, loopIndexName, myExpr);
	  }
          else {
	    myExpr.setArg(1, rewriteLoopIndexArrayRefList(templateObj, templateIndex, alignedArray,
							  loopIndexName, (XobjList)myExpr.getArg(1)));
	  }
	}
        break;
      case POINTER_REF:
	{
	  XMPalignedArray alignedArray = null;
	  XobjList indexList = Xcons.List();

	  Xobject addr_expr = myExpr;
	  while (addr_expr.Opcode() == Xcode.POINTER_REF){
	    addr_expr = addr_expr.getArg(0);
	    if (addr_expr.Opcode() == Xcode.PLUS_EXPR){
	      indexList.cons(addr_expr.getArg(1));
	      addr_expr = addr_expr.getArg(0);
	    }

	    if (addr_expr.Opcode() == Xcode.VAR){
	      alignedArray = globalDecl.getXMPalignedArray(addr_expr.getSym(), block);
	      break;
	    }

	  }

	  if (alignedArray != null){
	    Xobject newExpr = Xcons.arrayRef(alignedArray.getType(), alignedArray.getAddrId().Ref(),
					     rewriteLoopIndexArrayRefList(templateObj, templateIndex, alignedArray,
									  loopIndexName, indexList));
	    iter.setXobject(newExpr);
	  }

	  break;
	}
      default:
      }
    }
  }

  private static void rewriteLoopIndexVar(XMPtemplate templateObj, int templateIndex,
                                          String loopIndexName, Xobject expr) throws XMPexception
  {
    topdownXobjectIterator iter = new topdownXobjectIterator(expr);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject myExpr = iter.getXobject();
      if (myExpr == null) {
        continue;
      } else if (myExpr.isRewrittedByXmp()) {
        continue;
      }
      switch (myExpr.Opcode()) {
      case VAR:
	{
	  if (loopIndexName.equals(myExpr.getString())) {
	    Xobject newExpr = calcLtoG(templateObj, templateIndex, myExpr);
	    iter.setXobject(newExpr);
	  }
	} break;
      default:
      }
    }
  }

  private static XobjList rewriteLoopIndexArrayRefList(XMPtemplate t, int ti, XMPalignedArray a,
                                                       String loopIndexName, XobjList arrayRefList) throws XMPexception
  {
    if (arrayRefList == null) {
      return null;
    }

    XobjList newArrayRefList = Xcons.List();
    int arrayDimIdx = 0;

    for (Xobject x : arrayRefList) {
      newArrayRefList.add(rewriteLoopIndexArrayRef(t, ti, a, arrayDimIdx, loopIndexName, x));
      arrayDimIdx++;
      x.setIsRewrittedByXmp(true);
    }

    return newArrayRefList;
  }

  private static Xobject rewriteLoopIndexArrayRef(XMPtemplate t, int ti, XMPalignedArray a, int ai,
                                                  String loopIndexName, Xobject arrayRef) throws XMPexception
  {
    if (arrayRef.Opcode() == Xcode.VAR) {
      if (loopIndexName.equals(arrayRef.getString())) {
        return calcShadow(t, ti, a, ai, arrayRef);
      }
      else {
        return arrayRef;
      }
    }

    topdownXobjectIterator iter = new topdownXobjectIterator(arrayRef);
    for (iter.init(); !iter.end(); iter.next()) {
      Xobject myExpr = iter.getXobject();
      if (myExpr == null) {
        continue;
      } else if (myExpr.isRewrittedByXmp()) {
        continue;
      }

      switch (myExpr.Opcode()) {
        case VAR:
          {
            if (loopIndexName.equals(myExpr.getString())) {
              iter.setXobject(calcShadow(t, ti, a, ai, myExpr));
            }
          } break;
        default:
      }
    }
    return arrayRef;
  }

  private static Xobject calcShadow(XMPtemplate t, int ti, XMPalignedArray a, int ai,
                                    Xobject expr) throws XMPexception {
    expr.setIsRewrittedByXmp(true);
    if(a.getAlignSubscriptIndexAt(ai) != null){  // null is an asterisk
      if (ti != a.getAlignSubscriptIndexAt(ai).intValue()) {
	throw new XMPexception("array ref is not consistent with array alignment");
      }
    }

    XMPshadow shadow = a.getShadowAt(ai);
    switch (shadow.getType()) {
      case XMPshadow.SHADOW_NONE:
        return expr;
      case XMPshadow.SHADOW_NORMAL:
        return Xcons.binaryOp(Xcode.PLUS_EXPR, expr, shadow.getLo());
      case XMPshadow.SHADOW_FULL:
        return calcLtoG(t, ti, expr);
      default:
        throw new XMPexception("unknown shadow type");
    }
  }

  public static Xobject calcLtoG(XMPtemplate t, int ti, Xobject expr) throws XMPexception {
    expr.setIsRewrittedByXmp(true);

    if (!t.isDistributed()) {
      return expr;
    }

    XMPnodes n = t.getOntoNodes();
    int ni = -1;
    if (t.getDistMannerAt(ti) != XMPtemplate.DUPLICATION)
      ni = t.getOntoNodesIndexAt(ti).getInt();

    XobjList args = null;
    switch (t.getDistMannerAt(ti)) {
      case XMPtemplate.DUPLICATION:
        return expr;
      case XMPtemplate.BLOCK:
        // _XMP_M_LTOG_TEMPLATE_BLOCK(_l, _m, _N, _P, _p)
        args = Xcons.List(expr, t.getLowerAt(ti), t.getSizeAt(ti), n.getSizeAt(ni), n.getRankAt(ni));
        break;
      case XMPtemplate.CYCLIC:
        // _XMP_M_LTOG_TEMPLATE_CYCLIC(_l, _m, _P, _p)
        args = Xcons.List(expr, t.getLowerAt(ti), n.getSizeAt(ni), n.getRankAt(ni));
        break;
      case XMPtemplate.BLOCK_CYCLIC:
        // _XMP_M_LTOG_TEMPLATE_BLOCK_CYCLIC(_l, _b, _m, _P, _p)
        args = Xcons.List(expr, t.getWidthAt(ti), t.getLowerAt(ti), n.getSizeAt(ni), n.getRankAt(ni));
        break;
      case XMPtemplate.GBLOCK:
        // _XMP_M_LTOG_TEMPLATE_GBLOCK(_l, _x)
	args = Xcons.List(expr, t.getGtolTemp0IdAt(ti).Ref());
	break;
        // _XMP_M_LTOG_TEMPLATE_GBLOCK(_l, _m, _p)
	//args = Xcons.List(expr, t.getDescId().Ref(), Xcons.IntConstant(ti));
	//return XMP.getMacroId("_XMP_L2G_GBLOCK", Xtype.intType).Call(args);
      default:
        throw new XMPexception("unknown distribution manner");
    }
    return XMP.getMacroId("_XMP_M_LTOG_TEMPLATE_" + t.getDistMannerStringAt(ti), Xtype.intType).Call(args);
  }

  /*
   * rewrite OMP pragmas
   */
  private void rewriteOMPpragma(FunctionBlock fb, XMPsymbolTable localXMPsymbolTable)
  {
    topdownBlockIterator iter2 = new topdownBlockIterator(fb);

    for (iter2.init(); !iter2.end(); iter2.next()){
      Block block = iter2.getBlock();
      if (block.Opcode() == Xcode.OMP_PRAGMA){

	PragmaBlock pragmaBlock = (PragmaBlock)block;

	Xobject clauses = pragmaBlock.getClauses();
	if (clauses != null) rewriteOmpClauses(clauses, (PragmaBlock)block, fb, localXMPsymbolTable);

	if (pragmaBlock.getPragma().equals("PARALLEL_FOR")){
	  BlockList body = pragmaBlock.getBody();
	  if (body.getDecls() != null){
	    BlockList newBody = Bcons.emptyBody(body.getIdentList().copy(), body.getDecls().copy());
	    body.setIdentList(null);
	    body.setDecls(null);
	    newBody.add(Bcons.PRAGMA(Xcode.OMP_PRAGMA, pragmaBlock.getPragma(),
				     pragmaBlock.getClauses(), body));
	    pragmaBlock.replace(Bcons.COMPOUND(newBody));
	  }
	}

      }
    }
  }

  /*
   * rewrite OMP clauses
   */
  private void rewriteOmpClauses(Xobject expr, PragmaBlock pragmaBlock, Block block,
				 XMPsymbolTable localXMPsymbolTable)
  {
    bottomupXobjectIterator iter = new bottomupXobjectIterator(expr);
    
    for (iter.init(); !iter.end();iter.next()){
    	
      Xobject x = iter.getXobject();
      if (x == null)  continue;
      if (x.Opcode() == Xcode.VAR){
        try {
          iter.setXobject(rewriteArrayAddr(x, pragmaBlock));
        }
        catch (XMPexception e){
          XMP.error(x.getLineNo(), e.getMessage());
        }
      }
      else if (x.Opcode() == Xcode.LIST){
        if (x.left() != null && x.left().Opcode() == Xcode.STRING &&
            x.left().getString().equals("DATA_PRIVATE")){
          
          if (!pragmaBlock.getPragma().equals("FOR")) continue;
          
          XobjList itemList = (XobjList)x.right();
          
          // find loop variable
          Xobject loop_var = null;
          BasicBlockIterator i = new BasicBlockIterator(pragmaBlock.getBody());
          for (Block b = pragmaBlock.getBody().getHead(); b != null; b = b.getNext()){
            if (b.Opcode() == Xcode.FOR_STATEMENT){
              loop_var = ((ForBlock)b).getInductionVar();
            }
          }
          if (loop_var == null) continue;
          
          // check if the clause has contained the loop variable
          boolean flag = false;
          Iterator<Xobject> j = itemList.iterator();
          while (j.hasNext()){
            Xobject item = j.next();
            if (item.getName().equals(loop_var.getName())){
              flag = true;
            }
          }
          
          // add the loop variable to the clause
          if (!flag){
            itemList.add(loop_var);
          }
        }
      }
    }
  }
  
  /*
   * rewrite ACC pragmas
   */
  private void rewriteACCpragma(FunctionBlock fb, XMPsymbolTable localXMPsymbolTable){
    topdownBlockIterator bIter = new topdownBlockIterator(fb);

    for (bIter.init(); !bIter.end(); bIter.next()){
      Block block = bIter.getBlock();
      if (block.Opcode() == Xcode.ACC_PRAGMA){
	Xobject clauses = ((PragmaBlock)block).getClauses();
	if (clauses != null){
	  BlockList newBody = Bcons.emptyBody();
	  rewriteACCClauses(clauses, (PragmaBlock)block, fb, localXMPsymbolTable, newBody);
	  if(!newBody.isEmpty()){
	    bIter.setBlock(Bcons.COMPOUND(newBody));
    	    newBody.add(Bcons.COMPOUND(Bcons.blockList(block))); //newBody.add(block);
	  }
	}
      }
    }
  }

  /*
   * rewrite ACC clauses
   */
  private void rewriteACCClauses(Xobject expr, PragmaBlock pragmaBlock, Block block,
      XMPsymbolTable localXMPsymbolTable, BlockList body){

    bottomupXobjectIterator iter = new bottomupXobjectIterator(expr);

    for (iter.init(); !iter.end(); iter.next()){
      Xobject x = iter.getXobject();
      if (x == null) continue;

      if (x.Opcode() == Xcode.LIST){
	if (x.left() == null || x.left().Opcode() != Xcode.STRING) continue;
	
	String clauseName = x.left().getString();
	ACCpragma accClause = ACCpragma.valueOf(clauseName); 
	if(accClause != null){
    	  switch(accClause){
    	  case HOST:
    	  case DEVICE:
    	  case USE_DEVICE:
    	  case PRIVATE:
    	  case FIRSTPRIVATE:   
    	  case DEVICE_RESIDENT:
    	    break;
    	  default:
            if(!accClause.isDataClause()) continue;
    	  }
	  
	  XobjList itemList  = (XobjList)x.right();
	  for(int i = 0; i < itemList.Nargs(); i++){
	    Xobject item = itemList.getArg(i);
	    if(item.Opcode() == Xcode.VAR){
	      //item is variable or arrayAddr
	      try{
		itemList.setArg(i, rewriteACCArrayAddr(item, pragmaBlock, body));
	      }catch (XMPexception e){
		XMP.error(x.getLineNo(), e.getMessage());
	      }
	    }else if(item.Opcode() == Xcode.LIST){
	      //item is arrayRef
	      try{
	        itemList.setArg(i, rewriteACCArrayRef(item, pragmaBlock, body));
	      }catch(XMPexception e){
	        XMP.error(x.getLineNo(), e.getMessage());
	      }
	    }
	  }
	}

	if (x.left().getString().equals("PRIVATE")){
	  if (!pragmaBlock.getPragma().contains("LOOP")) continue;

	  XobjList itemList = (XobjList)x.right();

	  // find loop variable
	  Xobject loop_var = null;
	  BasicBlockIterator i = new BasicBlockIterator(pragmaBlock.getBody());
	  for (Block b = pragmaBlock.getBody().getHead();
	      b != null;
	      b = b.getNext()){
	    if (b.Opcode() == Xcode.F_DO_STATEMENT){
	      loop_var = ((FdoBlock)b).getInductionVar();
	    }
	  }
	  if (loop_var == null) continue;

	  // check if the clause has contained the loop variable
	  boolean flag = false;
	  Iterator<Xobject> j = itemList.iterator();
	  while (j.hasNext()){
	    Xobject item = j.next();
	    if (item.getName().equals(loop_var.getName())){
	      flag = true;
	    }
	  }

	  // add the loop variable to the clause
	  if (!flag){
	    itemList.add(loop_var);
	  }
	}
      }
    }
  }
  
  private Xobject rewriteACCArrayAddr(Xobject arrayAddr, Block block, BlockList body) throws XMPexception {
      XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(arrayAddr.getSym(), block);
      XMPcoarray coarray = _globalDecl.getXMPcoarray(arrayAddr.getSym(), block);

      if (alignedArray == null && coarray == null) {
	  return arrayAddr;
      }
      else if(alignedArray != null && coarray == null){ // only alignedArray
	  if (alignedArray.checkRealloc() || (alignedArray.isLocal() && !alignedArray.isParameter()) ||
		  alignedArray.isParameter()){
	      Xobject arrayAddrRef = alignedArray.getAddrId().Ref();
	      Ident descId = alignedArray.getDescId();
	      
	      
	      String arraySizeName = "_ACC_size_" + arrayAddr.getSym();
	      Ident arraySizeId = body.declLocalIdent(arraySizeName, Xtype.unsignedlonglongType);

	      Block getArraySizeFuncCall = _globalDecl.createFuncCallBlock("_XMP_get_array_total_elmts", Xcons.List(descId.Ref()));
	      body.insert(Xcons.Set(arraySizeId.Ref(), getArraySizeFuncCall.toXobject()));
	      
	      XobjList arrayRef = Xcons.List(arrayAddrRef, Xcons.List(Xcons.IntConstant(0), arraySizeId.Ref()));
	      
	      arrayRef.setIsRewrittedByXmp(true);
	      return arrayRef;
	  }
	  else {
	      return arrayAddr;
	  }
      } else if(alignedArray == null && coarray != null){  // only coarray
          Xobject coarrayAddrRef = _globalDecl.findVarIdent(XMP.COARRAY_ADDR_PREFIX_ + arrayAddr.getSym()).Ref();
	  Ident descId = coarray.getDescId();
	  
	  String arraySizeName = "_ACC_size_" + arrayAddr.getSym();
          Ident arraySizeId = body.declLocalIdent(arraySizeName, Xtype.unsignedlonglongType);
	  
          Block getArraySizeFuncCall = _globalDecl.createFuncCallBlock("_XMP_coarray_get_total_elmts", Xcons.List(descId.Ref()));
          body.insert(Xcons.Set(arraySizeId.Ref(), getArraySizeFuncCall.toXobject()));
          
          XobjList arrayRef = Xcons.List(coarrayAddrRef, Xcons.List(Xcons.IntConstant(0), arraySizeId.Ref()));
          
          arrayRef.setIsRewrittedByXmp(true);
          return arrayRef;
      } else{ // no execute
	  return arrayAddr;
      }
  }
  
  private Xobject rewriteACCArrayRef(Xobject arrayRef, Block block, BlockList body) throws XMPexception {
      Xobject arrayAddr = arrayRef.getArg(0);
      
      XMPalignedArray alignedArray = _globalDecl.getXMPalignedArray(arrayAddr.getSym(), block);
      XMPcoarray coarray = _globalDecl.getXMPcoarray(arrayAddr.getSym(), block);
      
      if(alignedArray != null && coarray == null){ //only alignedArray
          Xobject alignedArrayAddrRef = alignedArray.getAddrId().Ref();
          arrayRef.setArg(0, alignedArrayAddrRef);
      }else if(alignedArray == null && coarray != null){ //only coarray
          Xobject coarrayAddrRef = _globalDecl.findVarIdent(XMP.COARRAY_ADDR_PREFIX_ + arrayAddr.getSym()).Ref();
          arrayRef.setArg(0, coarrayAddrRef);
      }
      return arrayRef;
  }

  void rewriteACCPragma(XobjectDef def){
    Xobject x = def.getDef();
    Xobject pragma = x.getArg(0);

    if(! pragma.getString().equals("DECLARE")){
      return;
    }

    XobjList clauses = (XobjList)x.getArg(1);
    for(XobjArgs args = clauses.getArgs(); args != null; args = args.nextArgs()){ //for(Xobject clause : clauses){
      Xobject clause = args.getArg();
      Xobject clauseKind = clause.getArg(0);
      if(! clauseKind.getString().equals("CREATE")){
        XMP.fatal("'" + clauseKind.getString() + "' clause for coarray is not implemented yet");
      }
      XobjList clauseArgs = (XobjList)clause.getArg(1);
      for(XobjArgs args2 = clauseArgs.getArgs(); args2 != null; args2 = args2.nextArgs()){ //for(Xobject var : clauseArgs){
        Xobject var = args2.getArg();
        String varName = var.getName();
        XMPcoarray coarray = _globalDecl.getXMPcoarray(varName);
        if(coarray == null) continue;

        if(coarray.getVarId().getStorageClass() != StorageClass.EXTERN){
          //add func call like "_XMP_coarray_malloc_do_acc(&(_XMP_COARRAY_DESC_a), &(_XMP_COARRAY_ADDR_DEV_a));"
          Xtype elementType = coarray.getElmtType();
          Ident devPointerId = _globalDecl.declStaticIdent(XMP.COARRAY_ADDR_PREFIX_ + "DEV_" + varName, Xtype.Pointer(elementType));
          String funcName = "_XMP_coarray_malloc_do_acc";
          XobjList funcArgs = Xcons.List(coarray.getDescId().getAddr(),
                  devPointerId.getAddr());
          _globalDecl.addGlobalInitFuncCall(funcName, funcArgs);

          //add func call like "acc_map_data(_XMP_COARRAY_ADDR_a, _XMP_COARRAY_ADDR_DEV_a, _XMP_coarray_get_total_elmts(_XMP_COARRAY_DESC_a) * sizeof(int));"
          Ident getTotalElmtFuncId = _globalDecl.declExternFunc("_XMP_coarray_get_total_elmts", Xtype.intType);
          Xobject getTotalElmtFuncCall = getTotalElmtFuncId.Call(Xcons.List(coarray.getDescId().Ref()));
          Ident hostPointerId = _globalDecl.findVarIdent(XMP.COARRAY_ADDR_PREFIX_ + varName);
          funcName = "acc_map_data";
          funcArgs = Xcons.List(hostPointerId.Ref(),
                  devPointerId.Ref(),
                  Xcons.binaryOp(Xcode.MUL_EXPR,
                          Xcons.SizeOf(elementType),
                          getTotalElmtFuncCall));
          _globalDecl.addGlobalInitFuncCall(funcName, funcArgs);
        }


        //remove var from clause
        clauseArgs.removeArgs(args2);
      }
      if(clauseArgs.isEmpty()){
        //remove clause
        clauses.removeArgs(args);
      }
    }

    if(clauses.isEmpty()){
      //delete directive
      def.setDef(Xcons.List(Xcode.TEXT,
              Xcons.String("/* OpenACC declare directive is removed by XcalableACC */")));
    }
  }

  private boolean isUseDevice(String varName, Block block){
    Ident varId = block.findVarIdent(varName);
    if(varId == null) return false;
    
    for(Block b = block; b != null; b = b.getParentBlock()){
      if(b.Opcode() != Xcode.ACC_PRAGMA) continue;
      PragmaBlock pb = (PragmaBlock)b;
      if(! pb.getPragma().equals("HOST_DATA")) continue;

      for(Xobject clause : (XobjList)pb.getClauses()){
        if(! clause.getArg(0).getString().equals("USE_DEVICE")) continue;

        for(Xobject var : (XobjList)clause.getArg(1)){
          if(var.getSym().equals(varName) && b.findVarIdent(varName) == varId){
            return true;
          }
        }
      }
    }

    return false;
  }

  public void rewriteVarDecl(Xobject varDecl, boolean isLocal) {
    assert(varDecl.Opcode() == Xcode.VAR_DECL);

    String varName = varDecl.getArg(0).getName();
    Ident varId = _globalDecl.findVarIdent(varName);

    if (varId.isCoarray()) {
      XobjList codimensions = (XobjList)varId.getCodimensions();

      // normalization of codimensions:
      //  add the last codimension '*' if it is not present
      if (codimensions.getTail() == null ||
          codimensions.getTail().getInt() != XMPcoarray.ASTERISK)
        codimensions.add(Xcons.IntConstant(XMPcoarray.ASTERISK));

      try {
        XMPcoarray.translateCoarray_core(varId, varName, codimensions,
                                         _globalDecl, isLocal);
      } catch (XMPexception e) {
        XMP.error(varDecl.getLineNo(), e.getMessage());
      }
    }
  }

  private void addBarrier(FunctionBlock fb){

    topdownBlockIterator iter = new topdownBlockIterator(fb);

    for (iter.init(); !iter.end(); iter.next()) {

      Block b = iter.getBlock();

      // insert a barrier before each return statement
      if (b.Opcode() == Xcode.RETURN_STATEMENT){
    	Ident f = _globalDecl.declExternFunc("_XMP_barrier_EXEC", Xtype.Function(Xtype.voidType));
    	b.insert(f.Call(Xcons.List()));
      }
	
    }

    // add a barrier at the end of the function
    Ident f = _globalDecl.declExternFunc("_XMP_barrier_EXEC", Xtype.Function(Xtype.voidType));
    BlockList bl = fb.getBody().getHead().getBody();
    bl.add(f.Call(Xcons.List()));

  }

}
