/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
package exc.object;

import java.util.Stack;

public class bottomupXobjectIterator extends XobjectIterator
{
    Stack<Xobject> obj_stack;
    Stack<XobjArgs> arg_stack;

    XobjArgs prevArgs = null;

    public bottomupXobjectIterator(Xobject x)
    {
        topXobject = x;
    }

    @Override
    public void init()
    {
        init(topXobject);
    }

    @Override
    public void init(Xobject x)
    {
        obj_stack = new Stack<Xobject>();
        arg_stack = new Stack<XobjArgs>();

        currentXobject = x;
        while(currentXobject != null && currentXobject.Opcode() != Xcode.ID_LIST
            && currentXobject instanceof XobjList && currentXobject.getArgs() != null) {
            obj_stack.push(currentXobject);
            arg_stack.push(currentArgs);
            currentArgs = null;
            for(XobjArgs args = currentXobject.getArgs(); args != null; args = args.nextArgs()) {
                arg_stack.push(currentArgs);
                currentArgs = args;
            }
            currentXobject = currentArgs.getArg(); // next
        }
    }

    @Override
    public void next()
    {
        prevArgs = currentArgs;

        if(obj_stack.empty()) {
            currentXobject = null;
            return;
        }
        currentArgs = arg_stack.pop();
        if(currentArgs == null) {
            currentXobject = obj_stack.pop();
            currentArgs = arg_stack.pop();
            return;
        }
        currentXobject = currentArgs.getArg();
        while(currentXobject != null && currentXobject.Opcode() != Xcode.ID_LIST
            && currentXobject instanceof XobjList && currentXobject.getArgs() != null) {
            obj_stack.push(currentXobject);
            arg_stack.push(currentArgs);
            currentArgs = null;
            for(XobjArgs args = currentXobject.getArgs(); args != null; args = args.nextArgs()) {
                arg_stack.push(currentArgs);
                currentArgs = args;
            }
            currentXobject = currentArgs.getArg(); // next
        }
    }

    public void setPrevXobject(Xobject x)
    {
        prevArgs.setArg(x);
    }

    public Xobject getPrevXobject()
    {
        return prevArgs.getArg();
    }

    @Override
    public boolean end()
    {
        return currentXobject == null && obj_stack.empty();
    }

    @Override
    public Xobject getParent()
    {
        if(obj_stack.empty())
            return null;
        else
            return obj_stack.peek();
    }
}
