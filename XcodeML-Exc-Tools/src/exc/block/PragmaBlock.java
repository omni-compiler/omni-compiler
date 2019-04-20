package exc.block;

import exc.object.*;

//
// for pragma
//
public class PragmaBlock extends CompoundBlock
{
    String pragma;
    Xobject args;

    // statement block with null BasicBlock
    public PragmaBlock(Xcode code, String pragma, Xobject args, BlockList body)
    {
        super(code, body);
        LineNo ln = getLineNo();
        if(ln != null)
            setLineNo(new LineNo(ln.fileName(), ln.lineNo() - 1));
        this.code = code;
        this.pragma = pragma;
        this.args = args;
    }

    /** make clone */
    public PragmaBlock(PragmaBlock b)
    {
      super(b);
      LineNo ln = getLineNo();
      if (ln != null)
	setLineNo(new LineNo(ln.fileName(), ln.lineNo() - 1));
      this.code = b.Opcode();
      this.pragma = b.getPragma();
      this.args = b.getClauses().copy();
    }

    /** make clone */
    @Override
    public Block copy()
    {
        return new PragmaBlock(this);
    }

    public String getPragma()
    {
        return pragma;
    }

    public void setPragma(String pragma)
    {
	this.pragma = pragma;
    }

    public Xobject getClauses()
    {
        return args;
    }

    public void setClauses(Xobject args)
    {
        this.args = args;
    }

    public void addClauses(Xobject args)
    {
	if (this.args == null){
	    this.args = Xcons.List(args);
	}
	else {
	    ((XobjList)this.args).add(args);
	}
    }

    @Override
    public Xobject toXobject()
    {
        Xobject x = new XobjList(Opcode(),
            Xcons.String(pragma), args, super.toXobject());
        x.setLineNo(getLineNo());
        return x;
    }

    @Override
    public String toString()
    {
        StringBuilder s = new StringBuilder(256);
        s.append("(PragmaBlock:"+ Opcode().toString()+" super[");
        s.append(super.toString());
        s.append("])");
        return s.toString();
    }
}
