package exc.openacc;

import exc.block.*;
import exc.object.*;
import exc.util.MachineDep;
import exc.util.MachineDepConst;

import java.util.*;


class AccAtomic extends AccDirective {
  private Xtype _type;
  private Xcode _operator;
  private Xobject _operand;
  private Xobject _val;

  AccAtomic(ACCglobalDecl decl, AccInformation info, PragmaBlock pb) {
    super(decl, info, pb);
  }

  @Override
  void analyze() throws ACCexception {
    super.analyze();

    //check attr is one or none
    Set<ACCpragma> allAttrs = new HashSet<ACCpragma>() {{
      add(ACCpragma.READ);
      add(ACCpragma.WRITE);
      add(ACCpragma.UPDATE);
      add(ACCpragma.CAPTURE);
    }};

    ACCpragma attribute = ACCpragma.UPDATE;
    int attributeCount = 0;

    for (ACCpragma attr : allAttrs) {
      if (_info.hasClause(attr)) {
        attribute = attr;
        attributeCount++;
      }
    }

    if (attributeCount > 1) {
      throw new ACCexception("more than one attributes are specified");
    }

    //check unimplemented attribute
    switch (attribute) {
    case UPDATE:
      checkUpdateStatement();
      break;
    case READ:
    case WRITE:
    case CAPTURE:
    default:
      throw new ACCexception("'" + attribute.getName() + "' is not supported");
    }
  }

  @Override
  void generate() throws ACCexception {

  }

  @Override
  void rewrite() throws ACCexception {
    //_pb.replace(makeAtomicBlock());
  }

  @Override
  boolean isAcceptableClause(ACCpragma clauseKind) {
    switch (clauseKind) {
    case READ:
    case WRITE:
    case UPDATE:
    case CAPTURE:
      return true;
    }
    return false;
  }

  private void checkUpdateStatement() throws ACCexception {
    BlockList body = _pb.getBody();
    if (body.isEmpty() || !body.isSingle()) {
      throw new ACCexception("not a single statement");
    }
    Statement st = getSingleStatement(body.getHead());

    Xobject expr = st.getExpr();

    Xcode opcode = expr.Opcode();
    switch (opcode) {
    case POST_DECR_EXPR:
    case POST_INCR_EXPR:
    case PRE_DECR_EXPR:
    case PRE_INCR_EXPR: {
      _operand = expr.getArg(0);
      _val = null;
      _operator = opcode;
    }
    break;

    case ASG_PLUS_EXPR:    // +=
    case ASG_MUL_EXPR:     // *=
    case ASG_MINUS_EXPR:   // -=
    case ASG_DIV_EXPR:     // /=
    case ASG_BIT_AND_EXPR: // &=
    case ASG_BIT_XOR_EXPR: // ^=
    case ASG_BIT_OR_EXPR:  // |=
    case ASG_LSHIFT_EXPR:  // <<=
    case ASG_RSHIFT_EXPR: {// >>=
      _operand = expr.getArg(0);
      _val = expr.getArg(1);
      _operator = opcode;
    }
    break;

    case ASSIGN_EXPR: {
      Xobject lhs = expr.left();
      Xobject rhs = expr.right();
      Xobject rhsLeftOp = rhs.left();
      Xobject rhsRightOp = rhs.right();
      boolean isLeftOpSame = lhs.equals(rhsLeftOp);
      boolean isRightOpSame = lhs.equals(rhsRightOp);

      if (isLeftOpSame && !isRightOpSame) {
        _operand = lhs;
        _val = rhsRightOp;
        _operator = rhs.Opcode();
      } else if (!isLeftOpSame && isRightOpSame) {
        _operand = lhs;
        _val = rhsLeftOp;
        _operator = rhs.Opcode();
        if(_operator == Xcode.MINUS_EXPR || _operator == Xcode.DIV_EXPR){
          throw new ACCexception("'x = expr {-,/} x' is not supported yet");
        }
      } else {
        throw new ACCexception("not vaild");
      }
    }
    break;
    default:
      throw new ACCexception("unsupported statement");
    }

    //type check
    _type = _operand.Type();
    if (!_type.isNumeric()) {
      throw new ACCexception("not numerical");
    }
  }

  Block makeAtomicBlock() throws ACCexception {
    Xobject atomicFuncCall = makeCudaAtomicFuncCall(Xcons.AddrOf(_operand), _val, _operator);
    return Bcons.Statement(Xcons.List(atomicFuncCall));
  }

  private Statement getSingleStatement(Block b) {
    if (b.Opcode() != Xcode.LIST) {
      return null;
    }

    BasicBlock bb = b.getBasicBlock();
    if (bb.isSingle()) {
      return bb.getHead();
    }

    return null;
  }

  private Xobject makeCudaAtomicFuncCall(Xobject addr, Xobject val, Xcode op) throws ACCexception {
    String funcKind;

    if (!addr.Type().isPointer()) {
      throw new ACCexception("not addr");
    }
    Xtype castType = null;
    switch (op) {
    case POST_INCR_EXPR:   // x++
    case PRE_INCR_EXPR:    // ++x
      val = Xcons.IntConstant(1);
    case ASG_PLUS_EXPR:    // +=
    case PLUS_EXPR:        // +
      funcKind = "Add";
      castType = getCudaAtomicAddCastType();
      break;
    case POST_DECR_EXPR:   // x--
    case PRE_DECR_EXPR:    // --x
      val = Xcons.IntConstant(1);
    case ASG_MINUS_EXPR:   // -=
    case MINUS_EXPR:       // -
      funcKind = "Add";
      castType = getCudaAtomicAddCastType();
      val = Xcons.unaryOp(Xcode.UNARY_MINUS_EXPR, val);
      break;
    case ASG_BIT_AND_EXPR: // &=
    case BIT_AND_EXPR:     // &
      funcKind = "And";
      break;
    case ASG_BIT_XOR_EXPR: // ^=
    case BIT_XOR_EXPR:     // ^
      funcKind = "Xor";
      break;
    case ASG_BIT_OR_EXPR:  // |=
    case BIT_OR_EXPR:      // |
      funcKind = "Or";
      break;
    case ASG_MUL_EXPR:     // *=
    case MUL_EXPR:         // *
    case ASG_DIV_EXPR:     // /=
    case DIV_EXPR:         // /
    case ASG_LSHIFT_EXPR:  // <<=
    case LSHIFT_EXPR:      // <<
    case ASG_RSHIFT_EXPR:  // >>=
    case RSHIFT_EXPR:      // >>
      throw new ACCexception("unimplemented operator");
    default:
      throw new ACCexception("unsupported operator");
    }

    if(castType != null){
      addr = Xcons.Cast(Xtype.Pointer(castType), addr);
      val = Xcons.Cast(castType, val);
    }else{
      addr = Xcons.Cast(Xtype.Pointer(_type), addr);
      val = Xcons.Cast(_type, val);
    }
    
    Ident funcId = ACCutil.getMacroFuncId("atomic" + funcKind, castType);
    XobjList args = Xcons.List(addr);
    if (val != null) {
      args.add(val);
    }
    return funcId.Call(args);
  }

  private Xtype getCudaAtomicAddCastType() {
    if(_type.equals(Xtype.longType) || _type.equals(Xtype.unsignedlongType)){
      if(MachineDepConst.SIZEOF_UNSIGNED_INT == MachineDepConst.SIZEOF_UNSIGNED_LONG){
        return (Xtype.unsignedType);
      }else{
        return (Xtype.unsignedlonglongType);
      }
    }else if(_type.equals(Xtype.longlongType)){
      return (Xtype.unsignedlonglongType);
    }
    return null;
  }
}
