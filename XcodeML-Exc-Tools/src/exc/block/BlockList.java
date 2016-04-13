package exc.block;

import xcodeml.util.XmLog;
import xcodeml.util.XmOption;
import exc.object.*;

/**
 * Object to represents a list of blocks. This is usually used for a compound
 * statement.
 */
public class BlockList
{
    Block head, tail;
    Block parent;

    Xobject id_list;
    Xobject decls;

    // construct empty BlockList
    public BlockList()
    {
    }

    // construct empty BlockList
    public BlockList(Block b)
    {
        if(b != null)
            add(b);
    }

    // construct empty for COMPOUND_STATEMENT
    public BlockList(Xobject id_list, Xobject decls)
    {
        this.id_list = id_list;
        this.decls = decls;
    }

    public BlockList(BlockList b_l)
    {
        if(b_l.id_list != null)
            id_list = b_l.id_list.copy();
        if(b_l.decls != null)
            decls = b_l.decls.copy();
        for(Block b = b_l.getHead(); b != null; b = b.getNext())
            this.add(b.copy());
    }

    public BlockList copy()
    {
        return new BlockList(this);
    }

    public Block getTail()
    {
        return tail;
    }

    public Block getHead()
    {
        return head;
    }
    
    public LineNo getHeadLineNo()
    {
        return head != null ? head.getLineNo() : null;
    }

    public Block getParent()
    {
        return parent;
    }

    public BlockList getParentList()
    {
        return parent == null ? null : parent.getParent();
    }

    public void setParent(Block b)
    {
        parent = b;
    }

    // add block at tail
    public void add(Block b)
    {
      if(b == null) return;  // if b is null, do nothing
        if(head == null) {
            head = tail = b;
        } else {
            tail.next = b;
            b.prev = tail;
            tail = b;
        }
        b.setParent(this);
    }

    public void add(BasicBlock bb)
    {
        add(new SimpleBlock(Xcode.LIST, bb));
    }

    public void add(Xobject x)
    {
        add(new SimpleBlock(Xcode.LIST, BasicBlock.Statement(x)));
    }
    
    public void removeFirst()
    {
        if(head == null)
            return;
        head = head.next;
    }

    // insert block before head
    public void insert(Block b)
    {
        if(head == null) {
            head = tail = b;
        } else {
            head.prev = b;
            b.next = head;
            head = b;
        }
        b.setParent(this);
    }

    public void insert(BasicBlock bb)
    {
        insert(new SimpleBlock(Xcode.LIST, bb));
    }

    public void insert(Xobject x)
    {
        insert(new SimpleBlock(Xcode.LIST, BasicBlock.Statement(x)));
    }

    public boolean isEmpty()
    {
        return head == null;
    }

    public boolean isSingle()
    {
        return head != null && head == tail;
    }

    public XobjList getIdentList()
    {
        return (XobjList)id_list;
    }
    
    public void setIdentList(Xobject id_list)
    {
        this.id_list = id_list;
    }
    
    public void setDecls(Xobject decls)
    {
        this.decls = decls;
    }

    public Xobject getDecls()
    {
        return decls;
    }

    public Ident findLocalIdent(Ident id)
    {
        if(id_list == null)
            return null;
        for(Xobject a : (XobjList)id_list) {
            if(a == id)
                return id;
        }
        return null;
    }

    public Ident findLocalIdent(String s)
    {
        if(id_list == null)
            return null;
        for(Xobject a : (XobjList)id_list) {
            Ident id = (Ident)a;
            if(s.equals(id.getName()))
                return id;
        }
        return null;
    }

    public Xobject findLocalDecl(String s)
    {
        if(decls == null)
            return null;
        for(Xobject d : (XobjList)decls) {
            if(d.Opcode() != Xcode.VAR_DECL)
                continue;

            if(s.equals(d.getArg(0).getName()))
                return d;
        }
        return null;
    }

    public Ident findLocalIdent(Xtype type)
    {
        if(id_list == null)
            return null;
        for(Xobject a : (XobjList)id_list) {
            if(a.Type() == type)
                return (Ident)a;
        }
        return null;
    }

    // used by xcalablemp package
    public Ident declLocalIdent(String name, Xtype type) {
      Ident id = this.findLocalIdent(name);
      if (id == null) {
        id = Ident.Local(name, type);
        this.addIdent(id);
      }

      return id;
    }

    // used by xcalablemp package
    public Ident declLocalIdent(String name, Xtype type, StorageClass sclass, Xobject init) {
      Ident id = Ident.Local(name, type);
      id.setStorageClass(sclass);

      XobjList idList = (XobjList)this.getIdentList();
      if (idList == null) {
        idList = Xcons.List();
        this.setIdentList(idList);
      }

      idList.add(id);

      if (init != null) {
        id.setIsDeclared(true);
        XobjList declList = (XobjList)this.getDecls();
        if (declList == null) {
          declList = Xcons.List();
          this.setDecls(declList);
        }

        declList.add(Xcons.List(Xcode.VAR_DECL, id, init, null));
      }

      return id;
    }

    public void addIdent(Ident id)
    {
        if(id_list == null)
            id_list = Xcons.IDList();
        id_list.add(id);
    }

    public Boolean removeIdent(String name){

      Boolean f1 = false, f2 = false;

      if (id_list == null) return false;

      XobjArgs a = id_list.getArgs();
      do {
	if (a.getArg().getName().equals(name)){
	  id_list.removeArgs(a);
      	  f1 = true;
      	  break;
	}	  
	a = a.nextArgs();
      } while (a != null);

      XobjArgs b = decls.getArgs();
      do {
	if (b.getArg().getArg(0).getName().equals(name)){
	  decls.removeArgs(b);
      	  f2 = true;
      	  break;
	}	  
	b = b.nextArgs();
      } while (b != null);

      return (f1 | f2);
    }

    public void removeDeclInit()
    {
        Ident id;

        if(decls == null)
            return;
        boolean exists = false;
        
        // mark variables used in array size.
        for(Xobject d : (XobjList)decls) {
            if(d.Opcode() != Xcode.VAR_DECL)
                continue;
            if((id = findLocalIdent(d.getArg(0).getName())) == null)
                XmLog.fatal("removeDeclInit: no ID");
            if(id.Type().isVariableArray()) {
                for(topdownXobjectIterator ite = id.Type().getArraySizeExpr()
                    .topdownIterator(); !ite.end(); ite.next()) {
                    Xobject x = ite.getXobject();
                    if(x.isVarRef()) {
                        Ident xi = findLocalIdent(x.getName());
                        if(xi != null)
                            xi.setIsUsedInArraySize(true);
                    }
                }
            }
        }
        
        for(Xobject d : (XobjList)decls) {
            if(d.Opcode() == Xcode.VAR_DECL && (d.getArg(1) != null)) {
                exists = true;
                break;
            }
        }
        if(!exists)
            return;
        
        BasicBlock bb = new BasicBlock();
        
        for(Xobject d : (XobjList)decls) {
            if(d.Opcode() != Xcode.VAR_DECL)
                continue;
            if(d.getArg(1) == null)
                continue;
            if((id = findLocalIdent(d.getArg(0).getName())) == null)
                XmLog.fatal("removeDeclInit: no ID");
            if(id.getStorageClass() == StorageClass.STATIC)
                continue;
            if(id.Type().isConst())
                continue;
            if(id.isUsedInArraySize())
                continue;

            XobjArgs init = d.getArgs().nextArgs();
            if(init.getArg().Opcode() == Xcode.LIST
                || init.getArg().Opcode() == Xcode.STRING_CONSTANT)
                expandInitializer(bb, id.Ref(), init);
            else {
                if(StorageClass.REGISTER != id.getStorageClass())
                    bb.add(Xcons.Set(id.Ref(), init.getArg()));
            }

            d.setArg(1, null); // remove!
        }
        insert(bb);
    }

    void expandInitializer(BasicBlock bb, Xobject x, XobjArgs init)
    {
        Xobject v;
        XobjArgs mlist;
        
        if(init == null)
            return;
        
        Xtype type = x.Type();
        
        switch(type.getKind()) {
        case Xtype.BASIC:
        case Xtype.ENUM:
        case Xtype.FUNCTION:
        case Xtype.POINTER:
            v = init.getArg();
            if(v.Opcode() == Xcode.LIST)
                XmLog.fatal("expandInit: bad scalar init");
            bb.add(Xcons.Set(x, v));
            break;
        case Xtype.STRUCT:
            v = init.getArg();
            if(v.Opcode() == Xcode.LIST)
                init = v.getArgs();
            mlist = type.getMemberList().getArgs();
            for(; init != null; init = init.nextArgs()) {
                Ident mid = (Ident)mlist.getArg();
                Xobject memRef = Xcons.memberRef(Xcons.AddrOf(x), mid.getName());
                expandInitializer(bb, memRef, init);
                mlist = mlist.nextArgs();
            }
            break;
        case Xtype.ARRAY:
            v = init.getArg();
            if(v.Opcode() == Xcode.STRING_CONSTANT) {
                Xobject f = Xcons.Symbol(Xcode.FUNC_ADDR, Xtype.Pointer(Xtype
                    .Function(Xtype.voidType)), "strcpy");
                bb.add(Xcons.functionCall(f, Xcons.List(x, v)));
                break;
            }
            if(v.Opcode() == Xcode.LIST)
                init = v.getArgs();
            int i = 0;
            for(; init != null; init = init.nextArgs()) {
                Xobject addr = Xcons.binaryOp(Xcode.PLUS_EXPR, x, Xcons.IntConstant(i++));
                expandInitializer(bb, Xcons.PointerRef(addr), init);
            }
            break;
        case Xtype.UNION:
            v = init.getArg();
            if(v.Nargs() != 1 || v.getArg(0).Opcode() == Xcode.LIST) {
                XmLog.fatal("cannot expand initializer for UNION which is initialized by multiple values or designator.");
            } else {
                Ident mid = (Ident)type.getMemberList().getArg(0);
                Xobject memRef = Xcons.memberRef(Xcons.AddrOf(x), mid.getName());
                expandInitializer(bb, memRef, v.getArgs());
            }
            break;
        default:
            XmLog.fatal("expandInitializer: unknown type");
        }
    }
    
    public Xobject toXobject()
    {
        Xobject v;
        if(head == null)
            return Xcons.statementList();
        if(head == tail && head.Opcode() != Xcode.LIST)
            v = head.toXobject();
        else {
            v = Xcons.statementList();
            for(Block b = head; b != null; b = b.getNext()) {
                if(b.Opcode() != Xcode.LIST) {
                    v.add(b.toXobject());
                    continue;
                }
                    
                for(Statement s : b.getBasicBlock()) {
                    if(s.getExpr() == null)
                        continue;
                    Xobject x = null;
                    if(s.getExpr().Opcode().isFstatement() ||
		       s.getExpr().Opcode() == Xcode.EXPR_STATEMENT) {
                        x = s.getExpr();
                    } else {
                        x = new XobjList(Xcode.EXPR_STATEMENT, s.getExpr());
                        x.setLineNo(s.getLineNo());
                    }
                    v.add(x);
                }
            }
        }
        //        if(id_list == null && decls == null)
        //            return v;
        //        else if(XmOption.isLanguageC())
        //            return Xcons.CompoundStatement(id_list, decls, v);
        //        else
        //            return Xcons.FstatementList(v);
        if(XmOption.isLanguageC()) {
          if(id_list == null && decls == null)
            return v;
          return Xcons.CompoundStatement(id_list, decls, v);
        }
        else
          return Xcons.FstatementList(v);
    }
    
    @Override
    public String toString()
    {
        StringBuilder s = new StringBuilder(256);
        s.append("[BlockList id_list="+id_list);
        int i = 0;
        for(Block b = head; b != null; b = b.getNext()) {
            if(i++ > 0)
                s.append(" ");
            s.append(b.toString());
        }
        s.append("]");
        return s.toString();
    }
}
