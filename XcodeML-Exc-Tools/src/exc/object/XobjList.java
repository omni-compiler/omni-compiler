/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

import java.util.Iterator;

import xcodeml.IXobject;
import exc.block.Block;

/**
 * Xobject which contains the list of Xobjects.
 */

public class XobjList extends Xobject implements Iterable<Xobject>, XobjContainer
{
    XobjArgs args;
    XobjArgs tail; // for add
    LineNo lineno;

    /** Constructor for the empty XobjList with Xcode.LIST */
    public XobjList()
    {
        super(Xcode.LIST, null);
    }

    /** Constructor for the empty XobjList with code and type */
    public XobjList(Xcode code, Xtype type)
    {
        super(code, type);
    }

    /** Constructor for the argument list */
    public XobjList(Xcode code, Xtype type, XobjArgs a)
    {
        super(code, type);
        args = a;
    }

    /** Constructor with one argument */
    public XobjList(Xcode code, Xtype type, Xobject ... a)
    {
        super(code, type);
        
        XobjArgs pa = null;
        
        if(a != null && a.length > 0) {
            for(int i = a.length - 1; i >= 0; --i) {
                pa = new XobjArgs(a[i], pa);
            }
        }
        args = pa;
    }

    /** Constructor with only code */
    public XobjList(Xcode code)
    {
        super(code, null);
    }

    /** Constructor with code and argument */
    public XobjList(Xcode code, XobjArgs a)
    {
        super(code, null);
        args = a;
    }

    /** Constructor with one argument */
    public XobjList(Xcode code, Xobject ... a)
    {
        this(code, null, a);
    }

    public void cons(Xobject a)
    {
        if(args == null) {
            args = new XobjArgs(a, null);
        }
        else {
            XobjArgs newHead = new XobjArgs(a, args);
            args = newHead;
        }
    }

    /** Add object at the end of list */
    @Override
    public void add(Xobject a)
    {
        if(args == null) {
            tail = args = new XobjArgs(a, null);
            return;
        }
        if(tail == null) {
            for(tail = args; tail.nextArgs() != null; tail = tail.nextArgs())
                ;
        }
        tail.next = new XobjArgs(a, null);
        tail = tail.next;
    }

    /** Gets tail of list */
    @Override
    public Xobject getTail()
    {
        if(tail == null)
            return null;
        return tail.getArg();
    }

    /** Inserty object at the first of list */
    @Override
    public void insert(Xobject a)
    {
        if(args == null) {
            tail = args = new XobjArgs(a, null);
            return;
        }
        args = new XobjArgs(a, args);
    }

    /** returns the first argument */
    @Override
    public Xobject operand()
    {
        if(args == null)
            return null;
        return args.getArg();
    }

    /** returns the left argument (the first argument) */
    @Override
    public Xobject left()
    {
        if(args == null)
            return null;
        return args.getArg();
    }

    /** returns the right argment (the second argument) */
    @Override
    public Xobject right()
    {
        if(args == null || args.nextArgs() == null)
            return null;
        return args.nextArgs().getArg();

    }

    /** sets the first argument */
    @Override
    public void setOperand(Xobject x)
    {
        args.setArg(x);
    }

    /** sets the left (first) argument */
    @Override
    public void setLeft(Xobject x)
    {
        args.setArg(x);
    }

    /** sets the right (second) argument */
    @Override
    public void setRight(Xobject x)
    {
        args.nextArgs().setArg(x);
    }

    /** Get the argument list */
    @Override
    public XobjArgs getArgs()
    {
        return args;
    }
    
    /** Sets the argument list */
    @Override
    public void setArgs(XobjArgs l)
    {
        args = l;
        tail = null;
    }

    @Override
    public Xobject removeArgs(XobjArgs a)
    {
        if(args == null)
            return null;
        Xobject x = null;
        for(XobjArgs r = args, p = null; r != null; p = r, r = r.next) {
            if(r != a)
                continue;
            x = r.arg;
            if(p == null)
                args = r.next;
            else
                p.next = r.next;
            if(tail == r)
                tail = p;
        }

        return x;
    }
    
    @Override
    public Xobject removeFirstArgs()
    {
        if(args == null)
            return null;
        Xobject a = args.arg;
        args = args.next;
        if(args == null)
            tail = null;
        return a;
    }

    @Override
    public Xobject removeLastArgs()
    {
        if(tail == null)
            return null;
        Xobject a = tail.arg;
        XobjArgs orgTail = tail;
	if (args == tail){
	    args = null;
	    tail = null;
	    return a;
	}
        tail = args;
        while(tail.next != orgTail) {
            tail = tail.next;
        }
        tail.next = null;
        return a;
    }

    /** remove all arguments */
    public void clear()
    {
        args = tail = null;
    }
    
    /** Get the i-th argument */
    @Override
    public Xobject getArg(int i)
    {
        XobjArgs a = args;
	int j = i;
        while(i > 0) {
            a = a.nextArgs();
            i--;
        }
        if(a == null) {
            throw new NullPointerException(
                "i=" + j + ", args=" + (args != null ? args.toString() : "null"));
        }
        return a.getArg();
    }

    /** Get the i-th argument or null if no i-th argument */
    @Override
    public Xobject getArgOrNull(int i)
    {
        XobjArgs a = args;
        if(a == null)
            return null;
        while(i > 0) {
            a = a.nextArgs();
            if(a == null)
                return null;
            i--;
        }
        return a.getArg();
    }

    /** Get the argument that has the keyword or the i-th argument.
        Null if it is not found or illegal.
     */
    public Xobject getArgWithKeyword(String keyword, int i)
    {
        // select by keyword
        for (XobjArgs a = args; a != null; a = a.nextArgs()) {
          Xobject arg = a.getArg();
          if (arg.Opcode() == Xcode.F_NAMED_VALUE) {
            if (keyword.equalsIgnoreCase(arg.getArg(0).getName()))
              return arg.getArg(1);
          }
        }

        // select by position i
        Xobject arg = getArgOrNull(i);
        if (arg != null && arg.Opcode() == Xcode.F_NAMED_VALUE) {
          // found another keyword at position i
          return null;
        }
        return arg;
    }

    /** Sets the i-th argument */
    @Override
    public void setArg(int i, Xobject x)
    {
        XobjArgs a = args;
        while(i > 0) {
            a = a.nextArgs();
            i--;
        }
        a.setArg(x);
    }

    /** Return the number of arugments. */
    @Override
    public int Nargs()
    {
        int i = 0;
        for(XobjArgs a = args; a != null; a = a.nextArgs())
            i++;
        return i;
    }

    /** set the line number information. */
    @Override
    public void setLineNo(LineNo ln)
    {
        lineno = ln;
    }

    /** get the line number information. */
    @Override
    public LineNo getLineNo()
    {
        return lineno;
    }

    /* returns the copy of this XobjList object. */
    @Override
    public Xobject copy()
    {
        XobjList x = new XobjList(code, type);
        for(XobjArgs a = args; a != null; a = a.nextArgs()) {
            if(a.getArg() == null)
                x.add(null);
            else
                x.add(a.getArg().copy());
        }
        return copyTo(x);
    }

    /** check the equality of two Xobjects. */
    @Override
    public boolean equals(Xobject x)
    {
        if(x == null)
            return false;
        if(!super.equals(x))
            return false;
        XobjArgs a1 = args;
        XobjArgs a2 = x.getArgs();
        while(a1 != null) {
            if(a2 == null)
                return false;
            if(a1.getArg() != a2.getArg()) {
                if(a1.getArg() == null)
                    return false;
                if(!a1.getArg().equals(a2.getArg()))
                    return false;
            }
            a1 = a1.nextArgs();
            a2 = a2.nextArgs();
        }
        if(a2 != null)
            return false;
        return true;
    }
    
    public void reverse()
    {
        if(args == null)
            return;
        
        for(XobjArgs a = args, p = null, n = a.next; a != null;
            p = a, a = n, n = (a != null) ? a.next : null) {
            
            a.next = p;
        }
        
        XobjArgs tmp = args;
        args = tail;
        tail = tmp;
    }
    
    @Override
    public boolean isEmpty()
    {
        return (args == null);
    }

    /** convert to the printable string */
    @Override
    public String toString()
    {
        String s = "(" + OpcodeName();
        for(XobjArgs a = args; a != null; a = a.nextArgs()) {
            if(a.getArg() == null)
                s = s + " ()";
            else
                s = s + " " + a.getArg();
        }
        return s + ")";
    }

    @Override
    public Iterator<Xobject> iterator()
    {
        return new XobjListIterator(this);
    }

    @Override
    public Ident getMember(String name)
    {
        if(Opcode() != Xcode.ID_LIST)
            throw new IllegalStateException("not ID_LIST");
        for(Xobject a : this) {
            if(!(a instanceof Ident))
                throw new IllegalStateException("not Ident : " + a.getClass());
            if(a.getName().equals(name))
                return (Ident)a;
        }
        
        return null;
    }

    @Override
    public void setParentRecursively(IXobject parent)
    {
        super.setParentRecursively(parent);
        for(Xobject a: this) {
            if(a instanceof XobjList)
                ((XobjList)a).setParentRecursively(this);
        }
    }

    @Override
    public Xobject cfold(Block block)
    {
      FconstFolder cfolder = new FconstFolder(this, block);
      Xobject result = cfolder.run();
      if (result == null)
        return this;
      return result;
    }


    @Override
    public Ident find(String name, int find_kind)
    {
        return findIdent(name, find_kind);
    }

    public Ident findIdent(String name, int find_kind)
    {
        XobjList idList = null;
        
        switch(Opcode()) {
        case COMPOUND_STATEMENT:
            idList = (XobjList)getArg(0);
            break;
        case FUNCTION_DEFINITION:
        case F_MODULE_DEFINITION:
            idList = (XobjList)getArg(1);
            break;
        case ID_LIST:
            idList = this;
            break;
        }

        if(idList != null) {
            for(Xobject a : idList) {
                Ident id = (Ident)a;
                if(id.getName().equals(name)) {
                    switch(find_kind) {
                    case IXobject.FINDKIND_ANY:
                        return id;
                    case IXobject.FINDKIND_VAR:
                        if(id.getStorageClass().isVarOrFunc())
                            return id;
                        break;
                    case IXobject.FINDKIND_COMMON:
                        if(id.getStorageClass() == StorageClass.FCOMMON_NAME)
                            return id;
                        break;
                    case IXobject.FINDKIND_TAGNAME:
                        if(id.getStorageClass() == StorageClass.FTYPE_NAME ||
                            id.getStorageClass() == StorageClass.TAGNAME)
                            return id;
                        break;
                    }
                }
            }
        }

        if(parent != null) {
            return (Ident)parent.find(name, find_kind);
        }
        
        return null;
    }
    
    public boolean has(Xobject x)
    {
        for(Xobject xx : this) {
            if(x == xx) {
                return true;
            }
        }
        return false;
    }

    public boolean hasIdent(String name)
    {
        for(Xobject xx : this) {
            if(name.equals(xx.getName())) {
                return true;
            }
        }
        return false;
    }

    public Ident getIdent(String name)
    {
        for(Xobject x : this) {
            if(name.equals(x.getName())) {
                return (Ident)x;
            }
        }
        return null;
    }
    
    public Ident getStructTypeName(Xtype type)
    {
        for(Xobject x : this) {
            Ident id = (Ident)x;
            if(id.getStorageClass() == StorageClass.FTYPE_NAME &&
                id.Type().getBaseRefType() == type.getBaseRefType())
                return id;
        }
        
        return null;
    }

    // used by xcalablemp
    public void mergeList(XobjList l) {
      if (l != null) {
        for (Xobject x : l) {
          this.add(x);
        }
      }
    }

    public boolean hasNullArg() {
      if (args == null)          // empty list
        return false;
      return args.hasNullArg();
    }
}
