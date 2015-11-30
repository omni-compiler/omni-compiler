package exc.object;

import java.util.LinkedList;
import java.util.Iterator;
import java.util.List;

import exc.util.XobjectVisitable;
import exc.util.XobjectVisitor;

import xcodeml.ILineNo;
import xcodeml.IXobject;

public class XobjectDefEnv extends PropObject
    implements Iterable<XobjectDef>, IXobject, XobjectVisitable, XobjContainer
{
    protected Xobject identList;
    protected LinkedList<XobjectDef> defs = new LinkedList<XobjectDef>();
    protected XobjectDefEnv parent;

    public XobjectDefEnv()
    {
    }

    public XobjectDefEnv(Xobject identList)
    {
        this.identList = identList;
    }
    
    public void setIdentList(Xobject identList)
    {
        this.identList = identList;
    }

    public List<XobjectDef> getDefs()
    {
        return defs;
    }

    public void add(XobjectDef s)
    {
        s.setParent(this);
        defs.add(s);
    }

    public void add(Xobject x)
    {
        if(x.Opcode() == Xcode.LIST) {
            for(Xobject a : (XobjList)x)
                add(a);
            return;
        }
        add(new XobjectDef(x, this));
    }

    public void insert(XobjectDef s)
    {
        s.setParent(this);
        defs.addFirst(s);
    }

    public void insertBefore(XobjectDef post, XobjectDef s)
    {
        s.setParent(this);
        int idx = defs.indexOf(post);
        defs.add(idx, s);
    }

    public void addAfter(XobjectDef pre, XobjectDef s)
    {
        s.setParent(this);
        int idx = defs.indexOf(pre);
        defs.add(idx + 1, s);
    }

    public void insert(Xobject x)
    {
        insert(new XobjectDef(x, this));
    }

    //
    // iterator
    // 
    public void iterateDef(XobjectDefVisitor op)
    {
        topdownXobjectDefIterator ite = new topdownXobjectDefIterator(this);
        for(ite.init(); !ite.end(); ite.next())
            op.doDef(ite.getDef());
    }

    public void iterateFuncDef(XobjectDefVisitor op)
    {
        topdownXobjectDefIterator ite = new topdownXobjectDefIterator(this);
        for(ite.init(); !ite.end(); ite.next())
            if(ite.getDef().isFuncDef())
                op.doDef(ite.getDef());
    }

    @Override
    public IXobject find(String name, int kind)
    {
        return findIdent(name, kind);
    }
    
    public Ident findVarIdent(String name)
    {
        return findIdent(name, IXobject.FINDKIND_VAR);
    }

    // search ident name in identList
    public Ident findIdent(String name, int kind)
    {
        if(identList != null) {
            for(Xobject a : (XobjList)identList) {
                Ident id = (Ident)a;
                if(id.getName().equals(name)) {
                    switch(kind) {
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
                    case IXobject.FINDKIND_TYPEDEF_NAME:
                      if(id.getStorageClass() == StorageClass.TYPEDEF_NAME)
                        return id;
                      break;
                    }
                }
            }
        }
        return null;
    }

    // search ident id in identList
    public Ident findIdent(Ident id)
    {
        if(identList != null) {
            for(Xobject a : (XobjList)identList) {
                if(a == id)
                    return id;
            }
        }
        return null;
    }

    @Override
    public Iterator<XobjectDef> iterator()
    {
        return defs.iterator();
    }

    @Override
    public boolean enter(XobjectVisitor visitor)
    {
        return visitor.enter(this);
    }

    @Override
    public ILineNo getLineNo()
    {
        return null;
    }

    /** Retruns the list of the global identifiers in the object file. */
    public Xobject getGlobalIdentList()
    {
        return identList;
    }
    
    public XobjectFile getParentFile()
    {
        if(this instanceof XobjectFile)
            return (XobjectFile)this;
        if(parent == null)
            throw new IllegalStateException();
        if(parent instanceof XobjectFile)
            return (XobjectFile)parent;
        return parent.getParentFile();
    }

    @Override
    public IXobject getParent()
    {
        return parent;
    }

    @Override
    public void setParentRecursively(IXobject parent)
    {
        for(XobjectDef def : this) {
            def.setParentRecursively(this);
        }
    }
}
